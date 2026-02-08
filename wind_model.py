"""
Wind Model for Rocket Simulation

Generates time-varying wind with gusts and computes the resulting
roll torque on a spinning rocket. Wind creates asymmetric fin loading
through effective sideslip, which is the key physical motivation for
RL over simple rate-damper PID.

Supports two gust models:
- Legacy sinusoidal: deterministic dual-frequency sinusoids (default)
- Dryden turbulence: MIL-F-8785C forming filters on white noise

Usage:
    from wind_model import WindModel, WindConfig

    # Legacy sinusoidal model (default, backward compatible)
    config = WindConfig(enable=True, base_speed=3.0, max_gust_speed=2.0)
    wind = WindModel(config)
    wind.reset(seed=42)
    speed, direction = wind.get_wind(time=1.0, altitude=50.0)

    # Dryden turbulence model
    config = WindConfig(enable=True, base_speed=3.0, use_dryden=True,
                        turbulence_severity="moderate")
    wind = WindModel(config)
    wind.reset(seed=42)
    speed, direction = wind.get_wind(time=1.0, altitude=50.0)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from airframe import RocketAirframe


# Turbulence intensity lookup (sigma_w / W_20) for MIL-F-8785C
# at reference altitude of 20 ft (6.1 m)
_TURBULENCE_SEVERITY = {
    "light": 0.1,
    "moderate": 0.2,
    "severe": 0.3,
}


@dataclass
class WindConfig:
    """Configuration for wind model.

    Attributes:
        enable: Whether wind is active
        base_speed: Mean wind speed in m/s
        max_gust_speed: Maximum gust amplitude in m/s (legacy sinusoidal model)
        variability: Rate of direction change (higher = more variable)
        altitude_gradient: Speed increase per 100m altitude (legacy linear gradient)
        use_dryden: Use Dryden turbulence model instead of sinusoidal gusts
        turbulence_severity: Dryden severity ("light", "moderate", "severe")
        altitude_profile_alpha: Power-law exponent for altitude wind profile
        reference_altitude: Reference altitude for power-law profile in meters
    """

    enable: bool = False
    base_speed: float = 0.0
    max_gust_speed: float = 0.0
    variability: float = 0.3
    altitude_gradient: float = 0.0

    # Dryden turbulence model
    use_dryden: bool = False
    turbulence_severity: str = "light"
    altitude_profile_alpha: float = 0.14
    reference_altitude: float = 10.0

    # Body shadow factor: fraction of freestream dynamic pressure seen by
    # a fin directly in the body's lee.  The effective roll-torque
    # coefficient K_shadow = (1 - body_shadow_factor).  Default 0.90 gives
    # K_shadow = 0.10, matching the analysis in
    # docs/wind_roll_torque_analysis.md §3.1 (~1:74 wind-to-control ratio
    # for Estes Alpha at q=500 Pa, 1 m/s wind).
    body_shadow_factor: float = 0.90


class DrydenTurbulence:
    """MIL-F-8785C Dryden turbulence model using discrete-time forming filters.

    Implements longitudinal (u_g) and lateral (v_g) turbulence components
    as IIR filters driven by white Gaussian noise. The Dryden spectrum
    has exact rational transfer functions, making it straightforward to
    discretize via the Tustin (bilinear) transform.

    The lateral component (v_g) is the primary driver of roll torque via
    sideslip, while the longitudinal component (u_g) modulates effective
    headwind/tailwind.

    References:
        MIL-F-8785C, "Flying Qualities of Piloted Airplanes"
        MIL-HDBK-1797, "Flying Qualities of Piloted Aircraft"
    """

    def __init__(self, dt: float = 0.01):
        self.dt = dt

        # Filter states (reset per episode)
        # u_g: 1st-order IIR (longitudinal)
        self._u_state = 0.0
        # v_g: 2nd-order IIR (lateral) — needs 2 state variables
        self._v_state1 = 0.0
        self._v_state2 = 0.0

        # Filter coefficients (computed on reset based on flight conditions)
        # u_g: y[n] = a1_u * y[n-1] + b0_u * x[n] + b1_u * x[n-1]
        self._a1_u = 0.0
        self._b0_u = 0.0
        self._b1_u = 0.0
        self._x_prev_u = 0.0  # previous white noise input for u

        # v_g: y[n] = a1_v * y[n-1] + a2_v * y[n-2] + b0_v * x[n]
        #             + b1_v * x[n-1] + b2_v * x[n-2]
        self._a1_v = 0.0
        self._a2_v = 0.0
        self._b0_v = 0.0
        self._b1_v = 0.0
        self._b2_v = 0.0
        self._x_prev_v = 0.0
        self._x_prev2_v = 0.0
        self._y_prev_v = 0.0

        # Current turbulence outputs
        self.u_g = 0.0  # longitudinal gust (m/s)
        self.v_g = 0.0  # lateral gust (m/s)

        # Parameters stored for inspection
        self.sigma_u = 0.0
        self.sigma_v = 0.0
        self.L_u = 0.0
        self.L_v = 0.0

    def compute_parameters(
        self, altitude_m: float, W_20: float, severity: str, V: float
    ) -> None:
        """Compute Dryden parameters and discretize forming filters.

        Args:
            altitude_m: Altitude in meters (used for scale lengths)
            W_20: Wind speed at 20 ft (6.1 m) in m/s
            severity: Turbulence severity ("light", "moderate", "severe")
            V: Rocket airspeed in m/s (for Taylor's frozen turbulence)
        """
        h_ft = max(altitude_m * 3.28084, 10.0)  # Convert to feet, min 10 ft
        V = max(V, 1.0)  # Avoid division by zero

        intensity = _TURBULENCE_SEVERITY.get(severity, 0.1)

        # MIL-F-8785C scale lengths (in feet)
        denom = (0.177 + 0.000823 * h_ft) ** 1.2
        L_u_ft = h_ft / denom
        L_v_ft = L_u_ft  # Same for lateral at low altitude

        # Convert to meters
        self.L_u = L_u_ft * 0.3048
        self.L_v = L_v_ft * 0.3048

        # Turbulence intensities
        sigma_w = intensity * W_20
        sigma_denom = (0.177 + 0.000823 * h_ft) ** 0.4
        self.sigma_u = sigma_w / sigma_denom
        self.sigma_v = self.sigma_u  # Same for lateral at low altitude

        # Discretize longitudinal filter: H_u(s) = sigma_u * sqrt(2*L_u/V) / (V/L_u + s)
        # Rewritten: H_u(s) = sigma_u * sqrt(2*L_u/V) / (1/tau + s) * (1/tau)
        #          = sigma_u * sqrt(2*tau) / (1 + tau*s)
        # where tau = L_u/V
        # Output variance with unit-PSD white noise: G^2/(2*tau) = sigma_u^2  ✓
        tau_u = self.L_u / V
        G_u = self.sigma_u * np.sqrt(2.0 * tau_u)

        # Discrete-time scaling: continuous white noise has PSD=1 (unit^2/Hz),
        # but discrete w[n] with var=1 has PSD=dt. To get correct output variance,
        # scale the forming filter gain by sqrt(pi / dt) — this is the standard
        # Tustin noise scaling factor (sqrt(fs/f_bandwidth) effectively).
        #
        # More precisely: for a first-order filter G/(1+tau*s), output variance
        # with continuous white noise = G^2 / (2*tau). The Tustin-discretized filter
        # driven by discrete white noise (var=1) produces output variance =
        # G_d^2 * (b0^2 + b1^2) / (1 - a1^2). We need to match these, which
        # requires scaling by sqrt(1/dt).
        noise_scale = np.sqrt(1.0 / self.dt)

        # Tustin (bilinear) discretization: s -> (2/dt) * (1 - z^-1) / (1 + z^-1)
        # Denominator pole at a = (2*tau - dt) / (2*tau + dt)
        c_u = 2.0 * tau_u / self.dt
        self._a1_u = (c_u - 1.0) / (c_u + 1.0)
        # Gain for Tustin: G / (1 + c_u) applied to (1 + z^-1), with noise scaling
        gain_u = G_u * noise_scale / (c_u + 1.0)
        self._b0_u = gain_u
        self._b1_u = gain_u

        # Discretize lateral filter: H_v(s) = G_v * (1 + sqrt(3)*tau*s) / (1 + tau*s)^2
        # where G_v = sigma_v * sqrt(tau) and tau = L_v/V
        # Output variance with unit-PSD white noise: G_v^2/tau = sigma_v^2  ✓
        tau_v = self.L_v / V
        G_v = self.sigma_v * np.sqrt(tau_v)
        sqrt3 = np.sqrt(3.0)

        # Continuous: H_v(s) = G_v * (1 + sqrt(3)*tau*s) / (1 + tau*s)^2
        # Denominator: (1 + tau*s)^2 = 1 + 2*tau*s + tau^2*s^2
        # Numerator: 1 + sqrt(3)*tau*s

        # Tustin substitution: s = (2/dt)*(z-1)/(z+1)
        # Let c = 2*tau/dt
        c_v = 2.0 * tau_v / self.dt

        # Denominator of (1+tau*s)^2 after Tustin:
        # (1 + c*(z-1)/(z+1))^2 = ((z+1+c*(z-1))/(z+1))^2
        # = ((1+c)*z + (1-c))^2 / (z+1)^2
        # In z-polynomial: ((1+c)^2 * z^2 + 2*(1-c^2)*z + (1-c)^2) / (z+1)^2
        # But we need the full transfer function...

        # Direct approach: convert to state-space or use coefficient matching.
        # For a 2nd order system, Tustin gives us:
        # With s = (2/dt) * (1-z^-1)/(1+z^-1):
        # Denominator: 1 + 2*tau*s + tau^2*s^2
        # = 1 + 2*tau*(2/dt)*(1-z^-1)/(1+z^-1) + tau^2*(2/dt)^2*((1-z^-1)/(1+z^-1))^2

        # Multiply through by (1+z^-1)^2:
        # (1+z^-1)^2 + 2*tau*(2/dt)*(1-z^-1)*(1+z^-1) + tau^2*(2/dt)^2*(1-z^-1)^2
        # = (1+2z^-1+z^-2) + 4*tau/dt*(1-z^-2) + 4*tau^2/dt^2*(1-2z^-1+z^-2)

        d0 = 1.0 + 4.0 * tau_v / self.dt + 4.0 * tau_v**2 / self.dt**2
        d1 = 2.0 - 8.0 * tau_v**2 / self.dt**2
        d2 = 1.0 - 4.0 * tau_v / self.dt + 4.0 * tau_v**2 / self.dt**2

        # Numerator: G_v * (1 + sqrt(3)*tau*s)
        # After Tustin, multiply by (1+z^-1) from denominator clearing:
        # G_v * ((1+z^-1) + sqrt(3)*tau*(2/dt)*(1-z^-1))
        # = G_v * ((1 + 2*sqrt(3)*tau/dt) + (1 - 2*sqrt(3)*tau/dt)*z^-1)
        # But we cleared (1+z^-1)^2 from denominator, so numerator gets (1+z^-1):
        # G_v * ((1+z^-1) + sqrt(3)*tau*(2/dt)*(1-z^-1)) * (1+z^-1)

        n_a = 1.0 + sqrt3 * 2.0 * tau_v / self.dt
        n_b = 1.0 - sqrt3 * 2.0 * tau_v / self.dt

        # Numerator polynomial: G_v * (n_a + n_b*z^-1) * (1 + z^-1)
        # = G_v * (n_a + (n_a + n_b)*z^-1 + n_b*z^-2)
        # Apply same noise_scale factor as longitudinal filter
        self._b0_v = G_v * noise_scale * n_a / d0
        self._b1_v = G_v * noise_scale * (n_a + n_b) / d0
        self._b2_v = G_v * noise_scale * n_b / d0

        self._a1_v = -d1 / d0
        self._a2_v = -d2 / d0

    def reset(self) -> None:
        """Reset filter states for a new episode."""
        self._u_state = 0.0
        self._v_state1 = 0.0
        self._v_state2 = 0.0
        self._x_prev_u = 0.0
        self._x_prev_v = 0.0
        self._x_prev2_v = 0.0
        self._y_prev_v = 0.0
        self.u_g = 0.0
        self.v_g = 0.0

    def step(self, rng: np.random.Generator) -> Tuple[float, float]:
        """Advance turbulence by one timestep.

        Args:
            rng: Numpy random generator for white noise input

        Returns:
            Tuple of (u_g, v_g) — longitudinal and lateral gust in m/s
        """
        # White noise inputs (unit variance)
        w_u = rng.standard_normal()
        w_v = rng.standard_normal()

        # Longitudinal: 1st-order IIR
        u_new = (
            self._a1_u * self._u_state + self._b0_u * w_u + self._b1_u * self._x_prev_u
        )
        self._x_prev_u = w_u
        self._u_state = u_new
        self.u_g = u_new

        # Lateral: 2nd-order IIR (direct form I)
        v_new = (
            self._a1_v * self._v_state1
            + self._a2_v * self._v_state2
            + self._b0_v * w_v
            + self._b1_v * self._x_prev_v
            + self._b2_v * self._x_prev2_v
        )
        self._x_prev2_v = self._x_prev_v
        self._x_prev_v = w_v
        self._v_state2 = self._v_state1
        self._v_state1 = v_new
        self.v_g = v_new

        return self.u_g, self.v_g


class WindModel:
    """Time-varying wind model with gusts and altitude gradient.

    Supports two gust models:
    - Legacy sinusoidal (use_dryden=False): deterministic dual-frequency gusts
    - Dryden turbulence (use_dryden=True): MIL-F-8785C stochastic turbulence

    The roll torque calculation models the physical effect of crosswind
    on a finned rocket: wind creates an effective sideslip angle
    (v_wind / v_rocket), causing asymmetric lift on the fins. The torque
    varies with sin(wind_direction - roll_angle), creating a periodic
    disturbance that couples with the rocket's spin.
    """

    def __init__(self, config: WindConfig):
        self.config = config
        self._rng = np.random.default_rng()

        # Per-episode random state (legacy sinusoidal model)
        self._episode_speed = 0.0
        self._episode_direction = 0.0
        self._gust_phase = 0.0
        self._gust_freq = 1.0
        self._direction_drift_rate = 0.0

        # Dryden turbulence model
        self._dryden: Optional[DrydenTurbulence] = None
        if config.use_dryden:
            self._dryden = DrydenTurbulence(dt=0.01)
            self._dryden_initialized = False
            self._last_V = 0.0

    def reset(self, seed: Optional[int] = None):
        """Randomize wind conditions for a new episode.

        Args:
            seed: Optional RNG seed for reproducibility
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        cfg = self.config
        # Randomize base speed for this episode: uniform in [0, base_speed]
        self._episode_speed = self._rng.uniform(0.0, max(cfg.base_speed, 0.01))

        # Random initial direction
        self._episode_direction = self._rng.uniform(0.0, 2 * np.pi)

        # Direction drift rate (rad/s)
        self._direction_drift_rate = self._rng.normal(0.0, cfg.variability)

        if self.config.use_dryden and self._dryden is not None:
            self._dryden.reset()
            self._dryden_initialized = False
            self._last_V = 0.0
        else:
            # Legacy sinusoidal gust parameters
            self._gust_phase = self._rng.uniform(0.0, 2 * np.pi)
            self._gust_freq = self._rng.uniform(0.5, 2.0)  # Hz

    def _get_altitude_factor(self, altitude: float) -> float:
        """Compute wind speed scaling factor based on altitude.

        Uses power-law profile when Dryden is enabled, otherwise linear gradient.

        Args:
            altitude: Altitude in meters

        Returns:
            Multiplicative factor for wind speed
        """
        cfg = self.config
        if cfg.use_dryden:
            # Power-law boundary layer profile: V(z) = V_ref * (z / z_ref)^alpha
            z = max(altitude, 1.0)  # Clamp minimum altitude to 1m
            z_ref = max(cfg.reference_altitude, 1.0)
            return (z / z_ref) ** cfg.altitude_profile_alpha
        else:
            # Legacy linear gradient
            return 1.0 + cfg.altitude_gradient * max(altitude, 0.0) / 100.0

    def get_wind(
        self, time: float, altitude: float, rocket_velocity: float = 30.0
    ) -> Tuple[float, float]:
        """Get wind speed and direction at given time and altitude.

        Args:
            time: Simulation time in seconds
            altitude: Altitude in meters
            rocket_velocity: Rocket airspeed in m/s (used by Dryden for
                Taylor's frozen turbulence hypothesis)

        Returns:
            Tuple of (wind_speed_ms, wind_direction_rad)
        """
        if not self.config.enable:
            return 0.0, 0.0

        cfg = self.config

        # Base speed with altitude scaling
        alt_factor = self._get_altitude_factor(altitude)
        speed = self._episode_speed * alt_factor

        if cfg.use_dryden and self._dryden is not None:
            # Dryden turbulence model
            V = max(rocket_velocity, 1.0)

            # Recompute filter coefficients if airspeed changed significantly
            # (>20% change) or on first call
            if (
                not self._dryden_initialized
                or abs(V - self._last_V) > 0.2 * self._last_V
            ):
                W_20 = max(self._episode_speed, 0.1)
                self._dryden.compute_parameters(
                    altitude_m=max(altitude, 3.0),
                    W_20=W_20,
                    severity=cfg.turbulence_severity,
                    V=V,
                )
                self._dryden_initialized = True
                self._last_V = V

            u_g, v_g = self._dryden.step(self._rng)

            # u_g modulates headwind (adds to base speed)
            speed = max(0.0, speed + u_g)

            # v_g modulates crosswind direction
            # Convert lateral gust to direction perturbation
            if speed > 0.01:
                direction_perturbation = np.arctan2(v_g, speed)
            else:
                direction_perturbation = 0.0
        else:
            # Legacy sinusoidal gust model
            gust_amplitude = (
                cfg.max_gust_speed
                * 0.5
                * (
                    np.sin(2 * np.pi * self._gust_freq * time + self._gust_phase)
                    + 0.5
                    * np.sin(
                        2 * np.pi * self._gust_freq * 2.3 * time
                        + self._gust_phase * 1.7
                    )
                )
            )
            speed = max(0.0, speed + gust_amplitude)
            direction_perturbation = 0.0

        # Direction drifts over time
        direction = (
            self._episode_direction
            + self._direction_drift_rate * time
            + direction_perturbation
        )

        return speed, direction

    def get_roll_torque(
        self,
        wind_speed: float,
        wind_direction: float,
        roll_angle: float,
        velocity: float,
        dynamic_pressure: float,
        airframe: RocketAirframe,
        cl_alpha: float = 2.0,
    ) -> float:
        """Calculate roll torque from wind on a spinning rocket.

        Physics: For a symmetric N-fin rocket in uniform crosswind, linear
        theory predicts ZERO net roll torque — opposing fin pairs cancel
        (Barrowman 1967, confirmed by OpenRocket source).  The residual
        torque arises from second-order effects: the rocket body creates an
        asymmetric wake that breaks the perfect cancellation between
        windward and leeward fins.

        The model uses an empirical body-shadow coefficient (K_shadow) that
        captures the combined effect of wake asymmetry, crossflow coupling,
        and potential-flow modification at the fin root.  The torque varies
        as sin(2*(wind_dir - roll_angle)) for a 4-fin rocket because:
          - Zero when wind aligns with any fin (by symmetry)
          - Maximum when wind is at 45° to fin planes
          - Period = pi (factor of 2 from the sin*cos projection)
        For a 3-fin rocket the period is 2*pi/3.

        See docs/wind_roll_torque_analysis.md for full derivation and
        literature references.

        Args:
            wind_speed: Wind speed in m/s
            wind_direction: Wind direction in radians
            roll_angle: Current roll angle in radians
            velocity: Rocket velocity in m/s
            dynamic_pressure: Dynamic pressure in Pa
            airframe: RocketAirframe for geometry
            cl_alpha: Lift curve slope (~2.0 conservative estimate, varies with Mach)

        Returns:
            Roll torque in N*m
        """
        if wind_speed < 0.01 or velocity < 1.0:
            return 0.0

        # Effective sideslip angle from crosswind
        sideslip = np.arctan2(wind_speed, velocity)

        # Fin geometry
        fin_set = None
        for comp in airframe.components:
            if hasattr(comp, "num_fins"):
                fin_set = comp
                break

        if fin_set is None:
            return 0.0

        single_fin_area = fin_set.span * 0.5 * (fin_set.root_chord + fin_set.tip_chord)
        body_radius = airframe.body_diameter / 2.0
        moment_arm = body_radius + fin_set.span / 2.0
        N = fin_set.num_fins
        K_shadow = 1.0 - self.config.body_shadow_factor

        # Periodic forcing at the N-fin angular frequency.
        # For N=4: sin(2*gamma) — period pi, zero at fin-aligned angles.
        # For N=3: sin(3*gamma) — period 2*pi/3.
        # General N-fin: the lowest harmonic that doesn't cancel is N/2
        # for even N (giving factor N/2) or N for odd N.
        relative_angle = wind_direction - roll_angle
        if N % 2 == 0:
            forcing = np.sin((N // 2) * relative_angle)
        else:
            forcing = np.sin(N * relative_angle)

        # Torque = q * A_fin * Cl_alpha * beta * r_moment * K_shadow * forcing
        torque = (
            dynamic_pressure
            * single_fin_area
            * cl_alpha
            * sideslip
            * moment_arm
            * K_shadow
            * forcing
        )

        return torque
