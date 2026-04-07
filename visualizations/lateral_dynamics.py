"""
Lateral Dynamics Model for Trajectory Computation

The core simulation (SpinStabilizedCameraRocket) models 1D vertical motion
plus 1DOF roll.  It has no pitch/yaw or horizontal translation.  This module
adds a physics-based lateral (x-y) dynamics layer that runs alongside the
environment *without* modifying its state or training pipeline.

Physics modelled
----------------
1. **Wind drag force** — wind exerts a lateral aerodynamic force on the
   rocket's side profile (body tube + fins).
   F_drag = 0.5 * rho * Cd_lateral * A_lateral * (v_wind - v_rocket_lateral)^2

2. **Quasi-static pitch/yaw tilt** — the wind normal force on fins and body
   creates a pitching moment.  The rocket's static stability (fin CP behind
   CG) provides a restoring moment.  In quasi-static equilibrium the tilt
   angle is:
       theta_tilt ≈ F_normal / (q * CN_alpha * A_ref * stability_margin)
   where stability_margin = (CP - CG) / d_ref is the caliber-based static
   margin.

3. **Thrust vector tilt** — the tilted rocket has a horizontal thrust
   component T * sin(theta_tilt).  During boost this is the dominant lateral
   force (thrust >> drag for a model rocket).

4. **Spin-tilt coupling** — the tilt direction rotates with the spin.  If
   the rocket spins fast, the horizontal thrust component sweeps around and
   partially cancels.  If the roll controller keeps spin near zero, the tilt
   stays coherent and produces more lateral drift in a single direction.
   This is the mechanism by which controller quality affects lateral
   displacement.

Usage
-----
    tracker = LateralTracker(airframe)
    tracker.reset()

    # Inside episode loop, after env.step():
    tracker.update(info, thrust, mass, dt)

    # After episode:
    x_positions = tracker.x_history
    y_positions = tracker.y_history
"""

import numpy as np
from typing import List, Optional, Tuple

from airframe import RocketAirframe


class LateralTracker:
    """Track lateral (x-y) position using physics-based dynamics.

    This runs alongside the environment without modifying it.  It reads
    the info dict and reconstructs the lateral forces at each timestep.

    Parameters
    ----------
    airframe : RocketAirframe
        Rocket geometry (for lateral area, CP, CG, etc.)
    cd_lateral : float
        Lateral drag coefficient for the rocket body.  A long cylinder
        at moderate Reynolds number has Cd ≈ 1.0–1.2.  Default 1.1.
    cn_alpha_body : float
        Normal force coefficient slope for the body per radian of
        angle of attack.  For a slender body, CN_alpha ≈ 2 per radian.
    cn_alpha_fins : float
        Normal force coefficient slope for the fins per radian.
        For 4 flat-plate fins, CN_alpha ≈ 2*pi * (A_fin/A_ref) per fin
        pair (Barrowman), but we use an effective value.  Default 8.0
        (two fin pairs contributing lift).
    static_margin_calibers : float
        Static margin in calibers (body diameters).  Typical model rocket
        is 1.0–2.0 calibers.  Default 1.5.
    """

    def __init__(
        self,
        airframe: RocketAirframe,
        cd_lateral: float = 1.1,
        cn_alpha_body: float = 2.0,
        cn_alpha_fins: float = 8.0,
        static_margin_calibers: float = 1.5,
    ):
        self.airframe = airframe
        self.cd_lateral = cd_lateral
        self.cn_alpha_body = cn_alpha_body
        self.cn_alpha_fins = cn_alpha_fins
        self.static_margin_calibers = static_margin_calibers

        # Precompute lateral reference area (body side projection)
        self._lateral_area = self._compute_lateral_area()

        # Precompute reference area (frontal) for normal force coefficients
        self._ref_area = airframe.get_frontal_area()
        self._d_ref = airframe.body_diameter

        # State — reset per episode
        self.x_vel = 0.0
        self.y_vel = 0.0
        self.x_pos = 0.0
        self.y_pos = 0.0

        # History
        self.x_history: List[float] = []
        self.y_history: List[float] = []
        self.tilt_history: List[float] = []

    def _compute_lateral_area(self) -> float:
        """Compute the rocket's side-projected area.

        This is the area exposed to crosswind: body tube side area
        plus fin planform area of the two fins facing the wind.
        """
        area = 0.0

        for comp in self.airframe.components:
            if hasattr(comp, "outer_diameter") and hasattr(comp, "length"):
                # Body tube or motor mount — side projection is a rectangle
                area += comp.outer_diameter * comp.length
            elif hasattr(comp, "base_diameter") and hasattr(comp, "length"):
                # Nose cone — side projection is roughly a triangle
                area += 0.5 * comp.base_diameter * comp.length
            elif hasattr(comp, "num_fins"):
                # Fins — only fins facing the wind contribute.
                # For N fins, on average N/2 face the wind, but the
                # effective projected area is reduced by the azimuthal
                # projection.  For 4 fins at 90° spacing, the average
                # effective area ≈ 2 * single_fin_area * (2/pi)
                single_area = comp.fin_area
                area += comp.num_fins / 2 * single_area * (2 / np.pi)

        return max(area, 1e-6)

    def reset(self):
        """Reset for a new episode."""
        self.x_vel = 0.0
        self.y_vel = 0.0
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.x_history = [0.0]
        self.y_history = [0.0]
        self.tilt_history = [0.0]

    def update(
        self,
        info: dict,
        thrust: float,
        mass: float,
        dt: float,
    ):
        """Advance lateral state by one timestep.

        Parameters
        ----------
        info : dict
            The info dict returned by env.step().  Must contain:
            - wind_speed_ms, wind_direction_rad
            - air_density_kg_m3
            - vertical_velocity_ms
            - roll_angle_rad
            - dynamic_pressure_Pa
        thrust : float
            Current motor thrust (N).  0 during coast.
        mass : float
            Current total rocket mass (kg).
        dt : float
            Timestep (s).
        """
        ws = info.get("wind_speed_ms", 0.0)
        wd = info.get("wind_direction_rad", 0.0)
        rho = info.get("air_density_kg_m3", 1.225)
        v_vert = info.get("vertical_velocity_ms", 0.0)
        roll_angle = info.get("roll_angle_rad", 0.0)
        q = info.get("dynamic_pressure_Pa", 0.0)

        # Wind velocity components (inertial frame)
        wx = ws * np.cos(wd)
        wy = ws * np.sin(wd)

        # Relative lateral velocity (wind minus rocket lateral velocity)
        rel_vx = wx - self.x_vel
        rel_vy = wy - self.y_vel
        rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)

        # ----- 1. Lateral aerodynamic drag force -----
        # F = 0.5 * rho * Cd * A_lateral * |v_rel|^2, in direction of v_rel
        if rel_speed > 0.01:
            q_lateral = 0.5 * rho * rel_speed**2
            f_drag = self.cd_lateral * q_lateral * self._lateral_area
            fx_drag = f_drag * rel_vx / rel_speed
            fy_drag = f_drag * rel_vy / rel_speed
        else:
            fx_drag = 0.0
            fy_drag = 0.0

        # ----- 2. Quasi-static pitch tilt from wind -----
        # The wind creates a normal force on the rocket.  The
        # fin-stabilized airframe creates a restoring moment proportional
        # to (CP - CG) * CN_alpha * q * A_ref * alpha.
        #
        # At equilibrium: CN_alpha_total * q * A_ref * alpha = F_normal
        # where F_normal ≈ CN_alpha_total * q * A_ref * beta
        # and beta = arctan(wind_speed / rocket_speed)
        #
        # But the restoring moment is:
        #   M_restore = CN_alpha * q * A_ref * static_margin_m * alpha
        # and the wind moment is:
        #   M_wind = CN_alpha * q * A_ref * beta * L_moment_arm
        #
        # Simplification: quasi-static tilt alpha ≈ beta * L_wind / L_restore
        # For a well-designed rocket with 1.5 cal stability,
        # alpha ≈ beta / (stability_factor)
        #
        # We use a direct approach: the wind sideslip angle beta creates
        # a tilt.  The static margin determines how much the rocket resists
        # tipping.  Higher stability (larger margin) = less tilt.

        v_axial = max(abs(v_vert), 1.0)
        beta = np.arctan2(ws, v_axial)  # sideslip angle

        # Effective stability factor: ratio of restoring moment to
        # disturbing moment.  For a statically stable rocket this is > 1.
        # The restoring moment arm is (CP - CG), the disturbing arm is
        # approximately the distance from the nose to the center of
        # pressure of the normal force distribution (roughly L/3 for body).
        # With static_margin in calibers and a typical L/d ratio of ~15:
        #   stability_factor ≈ static_margin_cal * (CN_fins / CN_total)
        # We simplify: tilt = beta / (1 + static_margin_calibers * cn_ratio)
        cn_total = self.cn_alpha_body + self.cn_alpha_fins
        stability_factor = 1.0 + self.static_margin_calibers * (
            self.cn_alpha_fins / max(cn_total, 0.1)
        )

        # Quasi-static tilt angle (rad)
        # At very low q, the aero restoring force is negligible and the
        # rocket essentially free-falls.  Clamp tilt to physically
        # reasonable values (max ~30 degrees).
        if q > 10.0:
            theta_tilt = beta / stability_factor
        else:
            # Below ~10 Pa dynamic pressure, aero forces are negligible.
            # Tilt is driven by gravity/inertia, not aero.  Assume ~0.
            theta_tilt = 0.0

        theta_tilt = np.clip(theta_tilt, -0.52, 0.52)  # max ~30 deg

        # ----- 3. Tilt direction depends on wind direction + roll angle -----
        # The tilt is *into the wind* (weathercocking), so the horizontal
        # component of thrust points upwind.  But the tilt direction in
        # the body frame is fixed relative to the wind.  In the inertial
        # frame, the tilt direction is the wind direction.
        #
        # For spin-tilt coupling: when the rocket spins, the body-frame
        # tilt direction rotates with the spin.  For a quasi-static tilt,
        # the aero forcing always points the tilt into the wind regardless
        # of roll angle (the fins weathercock in real time).  So the
        # inertial-frame tilt direction tracks the wind direction, not the
        # roll angle.
        #
        # However, during rapid spin, there's a gyroscopic precession
        # effect: the tilt lags behind the wind forcing.  At high spin
        # rates, the rocket resists being tilted because of gyroscopic
        # stiffness.  The effective tilt is reduced by a factor related
        # to (spin_rate / natural_frequency).
        #
        # Gyroscopic reduction factor:
        #   The pitch natural frequency for a rocket with static margin is
        #   omega_n ≈ sqrt(CN_alpha * q * A_ref * margin / I_pitch)
        #   When spin_rate >> omega_n, gyroscopic stiffness dominates and
        #   tilt ≈ theta_tilt * (omega_n / spin_rate)^2
        #   When spin_rate << omega_n, the rocket weathercocks normally.

        roll_rate = np.deg2rad(info.get("roll_rate_deg_s", 0.0))
        abs_spin = abs(roll_rate)

        # Estimate pitch natural frequency
        # I_pitch ≈ (1/12) * m * L^2 (slender rod)
        L = self.airframe.total_length
        I_pitch = (1.0 / 12.0) * mass * L**2
        stability_margin_m = self.static_margin_calibers * self._d_ref

        if q > 10.0 and I_pitch > 0:
            omega_n_sq = cn_total * q * self._ref_area * stability_margin_m / I_pitch
            omega_n = np.sqrt(max(omega_n_sq, 0.01))

            # Gyroscopic suppression: at high spin, tilt is reduced
            if abs_spin > 0.1:
                gyro_factor = 1.0 / (1.0 + (abs_spin / omega_n) ** 2)
            else:
                gyro_factor = 1.0
        else:
            gyro_factor = 1.0

        effective_tilt = theta_tilt * gyro_factor

        # Tilt direction in inertial frame is opposite the wind direction
        # (rocket nose tips INTO the wind).  The horizontal thrust
        # component therefore points UPWIND (opposing wind drift).
        # But: the lateral force from tilt is T * sin(tilt), directed
        # into-wind (upwind), which is OPPOSITE to the wind direction.
        # This actually resists drift — which is correct physics.
        # However, during coast (thrust=0), there's no thrust-vector effect.

        tilt_dir_x = -np.cos(wd)  # Upwind direction (opposite wind)
        tilt_dir_y = -np.sin(wd)

        fx_thrust_tilt = thrust * np.sin(effective_tilt) * tilt_dir_x
        fy_thrust_tilt = thrust * np.sin(effective_tilt) * tilt_dir_y

        # ----- 4. Total force and integration -----
        fx_total = fx_drag + fx_thrust_tilt
        fy_total = fy_drag + fy_thrust_tilt

        # F = ma
        if mass > 0:
            ax = fx_total / mass
            ay = fy_total / mass
        else:
            ax = 0.0
            ay = 0.0

        # Symplectic Euler integration
        self.x_vel += ax * dt
        self.y_vel += ay * dt
        self.x_pos += self.x_vel * dt
        self.y_pos += self.y_vel * dt

        # Store history
        self.x_history.append(self.x_pos)
        self.y_history.append(self.y_pos)
        self.tilt_history.append(np.rad2deg(effective_tilt))

    def get_max_displacement(self) -> float:
        """Maximum radial displacement from launch point (m)."""
        x = np.array(self.x_history)
        y = np.array(self.y_history)
        return float(np.sqrt(x**2 + y**2).max())
