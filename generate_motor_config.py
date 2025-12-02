#!/usr/bin/env python3
"""
Motor Config Generator - FIXED VERSION

This script interfaces with the ThrustCurve.org API to:
1. Search for available rocket motors
2. Download motor thrust curve data
3. Automatically generate training configs with physics-appropriate parameters

Usage:
    # Search for motors
    python generate_motor_config.py search "C6"
    python generate_motor_config.py search --manufacturer Estes
    python generate_motor_config.py search --impulse-class F

    # List popular motors
    python generate_motor_config.py list-popular

    # Generate config from motor
    python generate_motor_config.py generate "Estes C6" --output configs/
    python generate_motor_config.py generate --motor-id <thrustcurve_id> --output configs/

    # Verify a motor exists
    python generate_motor_config.py verify "Aerotech F40"
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import yaml

import numpy as np
from scipy import interpolate, integrate

# ThrustCurve.org API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: 'requests' not installed. API features disabled.")
    print("Install with: pip install requests")


# ============================================================================
# ThrustCurve.org API Interface
# ============================================================================

THRUSTCURVE_API_BASE = "https://www.thrustcurve.org/api/v1"

# Popular/common motors for quick reference
POPULAR_MOTORS = {
    # Low Power (A-D)
    "estes_a8": {"manufacturer": "Estes", "designation": "A8", "common_name": "Estes A8"},
    "estes_b6": {"manufacturer": "Estes", "designation": "B6", "common_name": "Estes B6"},
    "estes_c6": {"manufacturer": "Estes", "designation": "C6", "common_name": "Estes C6"},
    "estes_d12": {"manufacturer": "Estes", "designation": "D12", "common_name": "Estes D12"},

    # Mid Power (E-G)
    "aerotech_e30": {"manufacturer": "AeroTech", "designation": "E30", "common_name": "Aerotech E30"},
    "aerotech_f40": {"manufacturer": "AeroTech", "designation": "F40", "common_name": "Aerotech F40"},
    "aerotech_f52": {"manufacturer": "AeroTech", "designation": "F52", "common_name": "Aerotech F52"},
    "aerotech_g80": {"manufacturer": "AeroTech", "designation": "G80", "common_name": "Aerotech G80"},
    "cesaroni_g79": {"manufacturer": "Cesaroni", "designation": "G79", "common_name": "Cesaroni G79"},

    # High Power (H+)
    "aerotech_h128": {"manufacturer": "AeroTech", "designation": "H128", "common_name": "Aerotech H128"},
    "cesaroni_h133": {"manufacturer": "Cesaroni", "designation": "H133", "common_name": "Cesaroni H133"},
    "aerotech_i284": {"manufacturer": "AeroTech", "designation": "I284", "common_name": "Aerotech I284"},
}


@dataclass
class MotorSearchResult:
    """Result from motor search"""
    motor_id: str
    manufacturer: str
    designation: str
    common_name: str
    diameter: float  # mm
    length: float    # mm
    impulse_class: str
    total_impulse: float  # N·s
    avg_thrust: float     # N
    max_thrust: float     # N
    burn_time: float      # s
    total_mass: float     # g
    prop_mass: float      # g

    def __str__(self):
        return (f"{self.manufacturer} {self.designation} "
                f"({self.impulse_class}-class, {self.total_impulse:.1f} N·s, "
                f"{self.avg_thrust:.1f}N avg, {self.diameter:.0f}mm)")


@dataclass
class MotorData:
    """Complete motor data with thrust curve"""
    motor_id: str
    manufacturer: str
    designation: str
    common_name: str

    # Physical properties
    diameter: float       # m (converted from mm)
    length: float         # m (converted from mm)
    total_mass: float     # kg (converted from g)
    propellant_mass: float  # kg
    case_mass: float      # kg

    # Performance
    total_impulse: float  # N·s
    burn_time: float      # s
    average_thrust: float # N
    max_thrust: float     # N

    # Thrust curve
    time_points: np.ndarray
    thrust_points: np.ndarray

    # Computed
    impulse_class: str = ""
    specific_impulse: float = 0.0

    def __post_init__(self):
        # Calculate impulse class
        impulse = self.total_impulse
        if impulse <= 2.5:
            self.impulse_class = 'A'
        elif impulse <= 5:
            self.impulse_class = 'B'
        elif impulse <= 10:
            self.impulse_class = 'C'
        elif impulse <= 20:
            self.impulse_class = 'D'
        elif impulse <= 40:
            self.impulse_class = 'E'
        elif impulse <= 80:
            self.impulse_class = 'F'
        elif impulse <= 160:
            self.impulse_class = 'G'
        elif impulse <= 320:
            self.impulse_class = 'H'
        elif impulse <= 640:
            self.impulse_class = 'I'
        elif impulse <= 1280:
            self.impulse_class = 'J'
        else:
            self.impulse_class = 'K+'

        # Calculate specific impulse
        if self.propellant_mass > 0:
            self.specific_impulse = self.total_impulse / (self.propellant_mass * 9.81)

        # Create interpolation function for smooth thrust lookup
        self.thrust_interpolator = interpolate.interp1d(
            self.time_points,
            self.thrust_points,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )

    def get_thrust(self, time: float) -> float:
        """Interpolate thrust at given time"""
        if time < 0 or time > self.burn_time:
            return 0.0
        return float(self.thrust_interpolator(time))

    def get_mass(self, time: float) -> float:
        """Get motor mass at given time - assumes linear burn"""
        if time <= 0:
            return self.total_mass
        elif time >= self.burn_time:
            return self.case_mass
        else:
            # Linear approximation - more sophisticated would integrate thrust
            burn_fraction = time / self.burn_time
            return self.case_mass + self.propellant_mass * (1 - burn_fraction)


class ThrustCurveAPI:
    """Interface to ThrustCurve.org API"""

    def __init__(self):
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library required for API access")
        self.base_url = THRUSTCURVE_API_BASE
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RocketSpinControl/1.0 (RL Training Config Generator)'
        })

    def search_motors(
        self,
        query: str = None,
        manufacturer: str = None,
        designation: str = None,
        common_name: str = None,
        impulse_class: str = None,
        diameter: float = None,
        max_results: int = 20,
        availability: str = None
    ) -> List[MotorSearchResult]:
        """
        Search for motors matching criteria.

        Args:
            query: Free text search (applied to results locally)
            manufacturer: Filter by manufacturer (e.g., "Estes", "AeroTech")
            designation: Motor designation (e.g., "C6", "F40")
            common_name: Common name (e.g., "C6", "F40")
            impulse_class: Single letter class (A-O)
            diameter: Motor diameter in mm
            max_results: Maximum results to return
            availability: Filter by availability (e.g., "regular", "available")

        Returns:
            List of matching motors
        """
        # FIXED: Use camelCase for API parameters
        params = {}

        if manufacturer:
            params["manufacturer"] = manufacturer
        if designation:
            params["designation"] = designation
        if common_name:
            params["commonName"] = common_name  # FIXED: was missing
        if impulse_class:
            params["impulseClass"] = impulse_class.upper()  # FIXED: camelCase
        if diameter:
            params["diameter"] = diameter
        if availability:
            params["availability"] = availability
        if max_results:
            params["maxResults"] = max_results  # FIXED: camelCase

        # Build search URL
        url = f"{self.base_url}/search.json"

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return []

        # FIXED: API returns results in 'results' key
        results = []
        for motor in data.get("results", []):
            try:
                # FIXED: Use camelCase field names from API
                result = MotorSearchResult(
                    motor_id=motor.get("motorId", ""),  # FIXED: camelCase
                    manufacturer=motor.get("manufacturer", ""),
                    designation=motor.get("designation", ""),
                    common_name=motor.get("commonName", ""),  # FIXED: camelCase
                    diameter=motor.get("diameter", 0),
                    length=motor.get("length", 0),
                    impulse_class=motor.get("impulseClass", ""),  # FIXED: camelCase
                    total_impulse=motor.get("totImpulseNs", 0),  # FIXED: camelCase
                    avg_thrust=motor.get("avgThrustN", 0),  # FIXED: camelCase
                    max_thrust=motor.get("maxThrustN", 0),  # FIXED: camelCase
                    burn_time=motor.get("burnTimeS", 0),  # FIXED: camelCase
                    total_mass=motor.get("totalWeightG", 0),  # FIXED: camelCase
                    prop_mass=motor.get("propWeightG", 0),  # FIXED: camelCase
                )

                # Filter by query if provided (local filter)
                if query:
                    query_lower = query.lower()
                    searchable = f"{result.manufacturer} {result.designation} {result.common_name}".lower()
                    if query_lower not in searchable:
                        continue

                results.append(result)
            except Exception as e:
                # Skip malformed results
                continue

        return results

    def get_motor_data(self, motor_id: str, search_result: Optional[MotorSearchResult] = None) -> Optional[MotorData]:
        """
        Download complete motor data including thrust curve.

        Args:
            motor_id: ThrustCurve motor ID (24-digit hex string)
            search_result: Optional search result with metadata (recommended)

        Returns:
            MotorData object or None if not found
        """
        # FIXED: Use download endpoint with motorId parameter
        download_url = f"{self.base_url}/download.json"

        # Request with motorId and data=samples to get parsed thrust curve
        params = {
            "motorId": motor_id,  # FIXED: camelCase
            "data": "samples"     # Request parsed samples
        }

        try:
            print(f"Downloading motor data from API...")
            response = self.session.get(download_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"API Response keys: {data.keys()}")
        except requests.RequestException as e:
            print(f"Failed to download motor data: {e}")
            print(f"Request URL: {download_url}")
            print(f"Parameters: {params}")
            return None
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response text: {response.text[:500]}")
            return None

        # FIXED: Response structure - results is an array
        if not data.get("results") or len(data["results"]) == 0:
            print(f"No data found for motor ID: {motor_id}")
            print(f"Full response: {json.dumps(data, indent=2)[:1000]}")
            return None

        motor = data["results"][0]

        # Debug: show what we got
        print(f"Motor data keys: {motor.keys()}")

        # If we have search result metadata, use it (more reliable)
        if search_result:
            print("Using metadata from search result")
            manufacturer = search_result.manufacturer
            designation = search_result.designation
            common_name = search_result.common_name
            diameter = search_result.diameter  # mm
            length = search_result.length      # mm
            total_mass_g = search_result.total_mass
            prop_mass_g = search_result.prop_mass
            total_impulse = search_result.total_impulse
            burn_time = search_result.burn_time
            average_thrust = search_result.avg_thrust
            max_thrust = search_result.max_thrust
        else:
            # Fall back to parsing from download response
            print("Parsing metadata from download response")
            manufacturer = motor.get("manufacturer", "Unknown")
            designation = motor.get("designation", "Unknown")
            common_name = motor.get("commonName", designation)
            diameter = motor.get("diameter", 18)  # mm
            length = motor.get("length", 70)      # mm
            total_mass_g = motor.get("totalWeightG", 0)
            prop_mass_g = motor.get("propWeightG", 0)
            total_impulse = motor.get("totImpulseNs", 0)
            burn_time = motor.get("burnTimeS", 1.0)
            average_thrust = motor.get("avgThrustN", 0)
            max_thrust = motor.get("maxThrustN", 0)

        case_mass_g = total_mass_g - prop_mass_g

        # Parse thrust curve from samples
        samples = motor.get("samples", [])

        if samples and len(samples) > 0:
            # FIXED: Samples are objects with 'time' and 'thrust' keys
            time_points = np.array([s.get("time", 0) for s in samples])
            thrust_points = np.array([s.get("thrust", 0) for s in samples])
        else:
            print(f"Warning: No thrust curve samples for {manufacturer} {designation}")
            print("         Creating synthetic curve from performance data")
            # Create synthetic curve from metadata
            time_points = np.array([0, 0.05, 0.1, burn_time * 0.8, burn_time])
            thrust_points = np.array([0, max_thrust, average_thrust * 1.1, average_thrust * 0.9, 0])

        # Build MotorData
        return MotorData(
            motor_id=motor_id,
            manufacturer=manufacturer,
            designation=designation,
            common_name=common_name,
            diameter=diameter / 1000,  # mm to m
            length=length / 1000,      # mm to m
            total_mass=total_mass_g / 1000,  # g to kg
            propellant_mass=prop_mass_g / 1000,
            case_mass=case_mass_g / 1000,
            total_impulse=total_impulse,
            burn_time=burn_time,
            average_thrust=average_thrust,
            max_thrust=max_thrust,
            time_points=time_points,
            thrust_points=thrust_points,
        )

    def verify_motor(self, query: str) -> Tuple[bool, Optional[MotorSearchResult]]:
        """
        Verify a motor exists and return its info.

        Args:
            query: Motor name like "Estes C6" or "Aerotech F40"

        Returns:
            (exists, motor_info) tuple
        """
        # Try parsing query into manufacturer and designation
        parts = query.split()
        manufacturer = None
        designation = None

        if len(parts) >= 2:
            manufacturer = parts[0]
            designation = parts[1]

        # Search with parsed criteria or just query
        if manufacturer and designation:
            results = self.search_motors(
                manufacturer=manufacturer,
                designation=designation,
                max_results=5
            )
        else:
            # Search by common name
            results = self.search_motors(
                common_name=query,
                max_results=5
            )

        if not results:
            # Try a broader search with just the query
            results = self.search_motors(query=query, max_results=5)

        if not results:
            return False, None

        # Try to find exact match
        query_lower = query.lower().replace(" ", "").replace("-", "")
        for result in results:
            name = f"{result.manufacturer}{result.designation}".lower().replace(" ", "").replace("-", "")
            if query_lower == name:
                return True, result

        # Return first result as best match
        return True, results[0]


# ============================================================================
# Physics-Based Config Generation
# ============================================================================

@dataclass
class PhysicsAnalysis:
    """Results of physics analysis for a motor/rocket combination"""
    motor: MotorData
    dry_mass: float           # kg
    total_mass: float         # kg
    twr: float                # thrust-to-weight ratio
    roll_inertia: float       # kg·m²

    # Control analysis at typical dynamic pressure
    control_accel_per_degree: float  # °/s² per degree of tab deflection
    disturbance_accel_std: float     # °/s² std from disturbance

    # Recommended parameters
    recommended_tab_deflection: float
    recommended_damping_scale: float
    recommended_initial_spin_std: float
    recommended_max_roll_rate: float
    recommended_dt: float

    # Validation
    random_action_safe: bool
    notes: List[str] = field(default_factory=list)


def analyze_motor_physics(
    motor: MotorData,
    dry_mass: float = None,
    diameter: float = None,
    fin_span: float = 0.04,
    fin_chord: float = 0.05,
    disturbance_scale: float = 0.0001,
) -> PhysicsAnalysis:
    """
    Analyze physics to determine appropriate config parameters.

    Args:
        motor: Motor data
        dry_mass: Rocket dry mass (kg), auto-calculated if None
        diameter: Rocket diameter (m), uses motor diameter if None
        fin_span: Fin span (m)
        fin_chord: Fin root chord (m)
        disturbance_scale: Base disturbance scale

    Returns:
        PhysicsAnalysis with recommended parameters
    """
    notes = []

    # Auto-calculate dry mass for TWR ~5 if not provided
    if dry_mass is None:
        target_twr = 5.0
        dry_mass = (motor.average_thrust / (target_twr * 9.81)) - motor.propellant_mass
        dry_mass = max(dry_mass, 0.05)  # Minimum 50g
        notes.append(f"Auto-calculated dry mass for TWR≈{target_twr}")

    # Use motor diameter if not specified
    if diameter is None:
        # Motor mount diameter, rocket body slightly larger
        diameter = motor.diameter * 1.1

    total_mass = dry_mass + motor.propellant_mass
    twr = motor.average_thrust / (total_mass * 9.81)

    if twr < 2.0:
        notes.append(f"WARNING: TWR={twr:.2f} is marginal (recommend >2.0)")

    # Calculate roll inertia (simplified model)
    radius = diameter / 2

    # Body tube (20% of mass at outer radius)
    tube_inertia = 0.2 * total_mass * radius**2

    # Internal components (80% at half radius)
    internal_inertia = 0.5 * 0.8 * total_mass * (radius * 0.5)**2

    # Fins
    fin_thickness = 0.002
    fin_mass_each = fin_span * fin_chord * fin_thickness * 1800  # fiberglass density
    fin_distance = radius + fin_span / 2
    fins_inertia = 4 * fin_mass_each * fin_distance**2

    roll_inertia = tube_inertia + internal_inertia + fins_inertia
    roll_inertia = max(roll_inertia, 1e-6)  # Minimum

    # Calculate control and disturbance at typical dynamic pressure
    # Use q at peak velocity (roughly when rocket reaches max speed)
    # Estimate max velocity from impulse and mass
    estimated_max_velocity = motor.total_impulse / total_mass
    q_typical = 0.5 * 1.225 * (estimated_max_velocity * 0.7)**2  # 70% of max
    q_typical = max(q_typical, 500)  # At least 500 Pa

    # Control torque per degree of deflection
    tab_area = 0.25 * fin_chord * 0.5 * fin_span  # 25% chord, 50% span
    moment_arm = radius + 0.5 * fin_span

    Cl_per_rad = 2 * np.pi  # Thin airfoil theory
    Cl_per_deg = Cl_per_rad * np.pi / 180

    tab_force_per_deg = 0.5 * Cl_per_deg * q_typical * tab_area
    control_torque_per_deg = 2 * tab_force_per_deg * moment_arm  # Two tabs
    control_accel_per_deg = np.rad2deg(control_torque_per_deg / roll_inertia)

    # Disturbance
    size_factor = (diameter / 0.054)**3
    disturbance_torque_std = disturbance_scale * np.sqrt(q_typical) * size_factor
    disturbance_accel_std = np.rad2deg(disturbance_torque_std / roll_inertia)

    # Determine recommended parameters
    # Goal: random actions shouldn't add more than ~50°/s per timestep

    # Start with dt=0.01 (100 Hz)
    dt = 0.01

    # Higher velocities need smaller dt
    if estimated_max_velocity > 100:
        notes.append("High velocity motor - using 100 Hz simulation")

    # Calculate max safe tab deflection
    # Random action effect per step = control_accel_per_deg * max_deflection * dt
    # Want this < 50°/s for training stability
    target_per_step = 50  # °/s
    max_safe_deflection = target_per_step / (control_accel_per_deg * dt)
    max_safe_deflection = min(max_safe_deflection, 15.0)  # Cap at 15°
    max_safe_deflection = max(max_safe_deflection, 1.0)   # Min 1°

    # Round to nice values
    if max_safe_deflection > 10:
        recommended_deflection = 10.0
    elif max_safe_deflection > 5:
        recommended_deflection = 5.0
    elif max_safe_deflection > 3:
        recommended_deflection = 3.0
    else:
        recommended_deflection = round(max_safe_deflection, 1)

    # Damping scale - higher for faster motors
    if estimated_max_velocity > 80:
        damping_scale = 3.0
    elif estimated_max_velocity > 50:
        damping_scale = 2.0
    else:
        damping_scale = 1.5

    # Initial spin std - proportional to control authority
    # With more control authority, we can handle more initial disturbance
    if recommended_deflection >= 5:
        initial_spin_std = 5.0
    else:
        initial_spin_std = 3.0

    # Max roll rate threshold
    # Higher for motors with less control authority (need more margin)
    if recommended_deflection < 5:
        max_roll_rate = 900.0
    elif recommended_deflection < 10:
        max_roll_rate = 720.0
    else:
        max_roll_rate = 540.0

    # Validate: simulate random actions
    random_action_safe = True
    simulated_spin = initial_spin_std
    for _ in range(100):
        random_action = np.random.uniform(-1, 1)
        spin_change = random_action * recommended_deflection * control_accel_per_deg * dt
        simulated_spin += spin_change
        if abs(simulated_spin) > max_roll_rate:
            random_action_safe = False
            break

    if not random_action_safe:
        notes.append("Config may need further tuning for random action survival")

    return PhysicsAnalysis(
        motor=motor,
        dry_mass=dry_mass,
        total_mass=total_mass,
        twr=twr,
        roll_inertia=roll_inertia,
        control_accel_per_degree=control_accel_per_deg,
        disturbance_accel_std=disturbance_accel_std,
        recommended_tab_deflection=recommended_deflection,
        recommended_damping_scale=damping_scale,
        recommended_initial_spin_std=initial_spin_std,
        recommended_max_roll_rate=max_roll_rate,
        recommended_dt=dt,
        random_action_safe=random_action_safe,
        notes=notes,
    )


# ============================================================================
# Config Generation
# ============================================================================

def generate_config(
    motor: MotorData,
    physics: PhysicsAnalysis,
    difficulty: str = "easy",
) -> Dict[str, Any]:
    """
    Generate a training config for the given motor and difficulty.

    Args:
        motor: Motor data
        physics: Physics analysis results
        difficulty: "easy", "medium", or "full"

    Returns:
        Config dictionary ready for YAML export
    """
    # Difficulty scaling
    if difficulty == "easy":
        deflection_mult = 1.0
        damping_mult = 1.0
        spin_std_mult = 1.0
        roll_rate_mult = 1.0
        wind_enabled = False
        wind_speed = 0.0
    elif difficulty == "medium":
        deflection_mult = 2.0  # More control authority
        damping_mult = 0.75
        spin_std_mult = 2.0
        roll_rate_mult = 0.8
        wind_enabled = True
        wind_speed = 3.0
    else:  # full
        deflection_mult = 3.0
        damping_mult = 0.5
        spin_std_mult = 3.0
        roll_rate_mult = 0.5
        wind_enabled = True
        wind_speed = 5.0

    # Calculate scaled parameters
    tab_deflection = min(physics.recommended_tab_deflection * deflection_mult, 15.0)
    damping_scale = max(physics.recommended_damping_scale * damping_mult, 1.0)
    initial_spin_std = physics.recommended_initial_spin_std * spin_std_mult
    max_roll_rate = max(physics.recommended_max_roll_rate * roll_rate_mult, 360.0)

    # Determine appropriate rocket geometry
    if motor.diameter <= 0.020:  # 18mm or smaller
        fin_span = 0.04
        fin_root_chord = 0.05
        fin_tip_chord = 0.025
        rocket_length = 0.40
    elif motor.diameter <= 0.030:  # 24-29mm
        fin_span = 0.05
        fin_root_chord = 0.06
        fin_tip_chord = 0.03
        rocket_length = 0.60
    else:  # 38mm+
        fin_span = 0.06
        fin_root_chord = 0.08
        fin_tip_chord = 0.04
        rocket_length = 0.80

    # Reward scaling based on expected altitude
    # Higher impulse motors go higher, so scale rewards accordingly
    if motor.total_impulse <= 20:  # A-D class
        altitude_reward_scale = 0.01
        max_altitude = 200
    elif motor.total_impulse <= 80:  # E-F class
        altitude_reward_scale = 0.005
        max_altitude = 500
    else:  # G+ class
        altitude_reward_scale = 0.003
        max_altitude = 800

    # Motor name for config
    motor_name = f"{motor.manufacturer}_{motor.designation}".lower().replace(" ", "_").replace("-", "_")

    config = {
        "physics": {
            "dry_mass": round(physics.dry_mass, 4),
            "propellant_mass": round(motor.propellant_mass, 4),
            "diameter": round(motor.diameter * 1.1, 4),  # Rocket slightly larger than motor
            "length": rocket_length,
            "num_fins": 4,
            "fin_span": fin_span,
            "fin_root_chord": fin_root_chord,
            "fin_tip_chord": fin_tip_chord,
            "max_tab_deflection": round(tab_deflection, 1),
            "tab_chord_fraction": 0.25,
            "tab_span_fraction": 0.5,
            "cd_body": 0.5,
            "cd_fins": 0.01,
            "cl_alpha": 2.0,
            "control_effectiveness": 1.0,
            "disturbance_scale": 0.0001,
            "damping_scale": round(damping_scale, 1),
            "initial_spin_std": round(initial_spin_std, 1),
            "max_roll_rate": round(max_roll_rate, 0),
        },
        "motor": {
            "name": motor_name,
            "manufacturer": motor.manufacturer,
            "designation": motor.designation,
            "thrust_multiplier": 1.0,

            # Complete motor specifications
            "diameter_mm": round(motor.diameter * 1000, 1),
            "length_mm": round(motor.length * 1000, 1),
            "total_mass_g": round(motor.total_mass * 1000, 1),
            "propellant_mass_g": round(motor.propellant_mass * 1000, 1),
            "case_mass_g": round(motor.case_mass * 1000, 1),

            # Performance data
            "impulse_class": motor.impulse_class,
            "total_impulse_Ns": round(motor.total_impulse, 1),
            "avg_thrust_N": round(motor.average_thrust, 1),
            "max_thrust_N": round(motor.max_thrust, 1),
            "burn_time_s": round(motor.burn_time, 3),

            # Thrust curve data - the actual data downloaded from ThrustCurve!
            "thrust_curve": {
                "time_s": motor.time_points.tolist(),  # Convert numpy array to list
                "thrust_N": motor.thrust_points.tolist(),
            }
        },
        "environment": {
            "dt": physics.recommended_dt,
            "max_episode_steps": 500,
            "initial_spin_rate_range": [
                -round(initial_spin_std * 2, 1),
                round(initial_spin_std * 2, 1)
            ],
            "initial_tilt_range": [-5.0, 5.0],
            "enable_wind": wind_enabled,
            "max_wind_speed": wind_speed,
            "max_gust_speed": wind_speed * 0.5,
            "wind_variability": 0.3 if wind_enabled else 0.0,
            "max_tilt_angle": 60.0 if difficulty == "easy" else 45.0,
            "min_altitude": -1.0,
            "max_altitude": max_altitude,
            "normalize_observations": True,
            "obs_clip_value": 10.0,
        },
        "reward": {
            "altitude_reward_scale": altitude_reward_scale,
            "spin_penalty_scale": -0.05 if difficulty == "easy" else -0.08 if difficulty == "medium" else -0.1,
            "low_spin_bonus": 1.0,
            "low_spin_threshold": 30.0 if difficulty == "easy" else 20.0 if difficulty == "medium" else 10.0,
            "control_effort_penalty": -0.005,
            "control_smoothness_penalty": -0.02,
            "success_bonus": 100.0,
            "crash_penalty": -20.0 if difficulty == "easy" else -30.0 if difficulty == "medium" else -50.0,
            "use_potential_shaping": difficulty == "full",
            "gamma": 0.99,
        },
        "ppo": {
            "learning_rate": 0.0003 if difficulty == "easy" else 0.0002 if difficulty == "medium" else 0.0001,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2 if difficulty != "full" else 0.15,
            "clip_range_vf": None,
            "ent_coef": 0.01 if difficulty == "easy" else 0.008 if difficulty == "medium" else 0.005,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
            "policy_net_arch": [256, 256],
            "value_net_arch": [256, 256],
            "activation": "tanh",
            "total_timesteps": 500000,
            "n_envs": 8,
            "device": "auto",
        },
        "curriculum": {
            "enabled": difficulty == "full",
            "stages": [] if difficulty != "full" else [
                {"name": "warmup", "initial_spin_range": [-20, 20], "wind_enabled": True, "max_wind_speed": 3.0, "target_reward": 50},
                {"name": "moderate", "initial_spin_range": [-30, 30], "wind_enabled": True, "max_wind_speed": 4.0, "target_reward": 100},
                {"name": "full", "initial_spin_range": [-40, 40], "wind_enabled": True, "max_wind_speed": 5.0, "target_reward": 150},
            ],
            "episodes_to_evaluate": 100,
            "advancement_threshold": 0.8,
        },
        "logging": {
            "log_dir": "logs",
            "save_dir": "models",
            "tensorboard_log": True,
            "save_freq": 10000,
            "keep_checkpoints": 5,
            "eval_freq": 5000,
            "n_eval_episodes": 20,
            "log_episode_freq": 10,
            "experiment_name": f"rocket_{motor_name}_{difficulty}",
            "tags": [motor_name, "spin_control", difficulty],
        },
    }

    return config


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def save_config(config: Dict, filepath: str, motor: MotorData, physics: PhysicsAnalysis, difficulty: str):
    """Save config to YAML file with header comments"""

    # Convert numpy types to native Python types
    config_clean = convert_numpy_types(config)

    header = f"""# {Path(filepath).name}
#
# Auto-generated config for {motor.manufacturer} {motor.designation}
# Difficulty: {difficulty}
#
# Motor: {motor.impulse_class}-class, {motor.total_impulse:.1f} N·s total impulse
#        {motor.average_thrust:.1f}N average thrust, {motor.burn_time:.2f}s burn time
#        Thrust curve: {len(motor.time_points)} data points from ThrustCurve.org
#
# Rocket: {physics.dry_mass*1000:.0f}g dry mass, TWR={physics.twr:.2f}
#
# Physics Analysis:
#   - Control authority: {physics.control_accel_per_degree:.0f} °/s² per degree deflection
#   - Recommended tab deflection: {physics.recommended_tab_deflection:.1f}°
#   - Random action safe: {'Yes' if physics.random_action_safe else 'Needs tuning'}
#
"""
    for note in physics.notes:
        header += f"# NOTE: {note}\n"

    header += "#\n"

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        f.write(header)
        yaml.dump(config_clean, f, default_flow_style=False, sort_keys=False)

    print(f"  Saved: {filepath}")


def generate_motor_configs(
    motor: MotorData,
    output_dir: str = "configs",
    dry_mass: float = None,
    difficulties: List[str] = None,
) -> Dict[str, str]:
    """
    Generate all difficulty configs for a motor.

    Args:
        motor: Motor data
        output_dir: Output directory
        dry_mass: Optional dry mass override
        difficulties: List of difficulties to generate

    Returns:
        Dict mapping difficulty to filepath
    """
    if difficulties is None:
        difficulties = ["easy", "medium", "full"]

    # Analyze physics
    physics = analyze_motor_physics(motor, dry_mass=dry_mass)

    print(f"\n{'='*60}")
    print(f"Motor: {motor.manufacturer} {motor.designation}")
    print(f"{'='*60}")
    print(f"Class: {motor.impulse_class}")
    print(f"Total Impulse: {motor.total_impulse:.1f} N·s")
    print(f"Avg/Max Thrust: {motor.average_thrust:.1f} / {motor.max_thrust:.1f} N")
    print(f"Burn Time: {motor.burn_time:.2f} s")
    print(f"\nRocket Configuration:")
    print(f"  Dry Mass: {physics.dry_mass*1000:.0f} g")
    print(f"  Total Mass: {physics.total_mass*1000:.0f} g")
    print(f"  TWR: {physics.twr:.2f}")
    print(f"\nPhysics Analysis:")
    print(f"  Roll Inertia: {physics.roll_inertia:.2e} kg·m²")
    print(f"  Control: {physics.control_accel_per_degree:.0f} °/s² per deg")
    print(f"  Disturbance: {physics.disturbance_accel_std:.0f} °/s² std")
    print(f"\nRecommended Parameters:")
    print(f"  Tab Deflection: {physics.recommended_tab_deflection:.1f}°")
    print(f"  Damping Scale: {physics.recommended_damping_scale:.1f}")
    print(f"  Initial Spin Std: {physics.recommended_initial_spin_std:.1f}°/s")
    print(f"  Max Roll Rate: {physics.recommended_max_roll_rate:.0f}°/s")
    print(f"  Timestep: {physics.recommended_dt} s")
    print(f"  Random Action Safe: {'✓ Yes' if physics.random_action_safe else '⚠ Needs tuning'}")

    for note in physics.notes:
        print(f"  Note: {note}")

    # Generate configs
    print(f"\nGenerating configs:")

    motor_name = f"{motor.manufacturer}_{motor.designation}".lower().replace(" ", "_").replace("-", "_")
    filepaths = {}

    for difficulty in difficulties:
        config = generate_config(motor, physics, difficulty)
        filepath = os.path.join(output_dir, f"{motor_name}_{difficulty}.yaml")
        save_config(config, filepath, motor, physics, difficulty)
        filepaths[difficulty] = filepath

    return filepaths


# ============================================================================
# Offline Motor Database (for when API is unavailable)
# ============================================================================

def get_offline_motor(motor_key: str) -> Optional[MotorData]:
    """Get motor data from offline database"""

    # Built-in motor database
    motors = {
        "estes_a8": MotorData(
            motor_id="offline_estes_a8",
            manufacturer="Estes",
            designation="A8",
            common_name="Estes A8",
            diameter=0.018,
            length=0.070,
            total_mass=0.0163,
            propellant_mass=0.0031,
            case_mass=0.0132,
            total_impulse=2.5,
            burn_time=0.5,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=np.array([0, 0.02, 0.1, 0.4, 0.5]),
            thrust_points=np.array([0, 10, 6, 4, 0]),
        ),
        "estes_b6": MotorData(
            motor_id="offline_estes_b6",
            manufacturer="Estes",
            designation="B6",
            common_name="Estes B6",
            diameter=0.018,
            length=0.070,
            total_mass=0.0195,
            propellant_mass=0.0056,
            case_mass=0.0139,
            total_impulse=5.0,
            burn_time=0.8,
            average_thrust=6.25,
            max_thrust=12.0,
            time_points=np.array([0, 0.03, 0.15, 0.6, 0.8]),
            thrust_points=np.array([0, 12, 8, 5, 0]),
        ),
        "estes_c6": MotorData(
            motor_id="offline_estes_c6",
            manufacturer="Estes",
            designation="C6",
            common_name="Estes C6",
            diameter=0.018,
            length=0.070,
            total_mass=0.024,
            propellant_mass=0.0123,
            case_mass=0.0117,
            total_impulse=10.0,
            burn_time=1.85,
            average_thrust=5.4,
            max_thrust=14.0,
            time_points=np.array([0.0, 0.04, 0.13, 0.5, 1.0, 1.5, 1.85]),
            thrust_points=np.array([0.0, 14.0, 12.0, 6.0, 5.0, 4.5, 0.0]),
        ),
        "estes_d12": MotorData(
            motor_id="offline_estes_d12",
            manufacturer="Estes",
            designation="D12",
            common_name="Estes D12",
            diameter=0.024,
            length=0.070,
            total_mass=0.044,
            propellant_mass=0.021,
            case_mass=0.023,
            total_impulse=20.0,
            burn_time=1.6,
            average_thrust=12.5,
            max_thrust=30.0,
            time_points=np.array([0, 0.05, 0.2, 0.8, 1.4, 1.6]),
            thrust_points=np.array([0, 30, 16, 10, 8, 0]),
        ),
        "aerotech_f40": MotorData(
            motor_id="offline_aerotech_f40",
            manufacturer="AeroTech",
            designation="F40",
            common_name="Aerotech F40",
            diameter=0.029,
            length=0.124,
            total_mass=0.090,
            propellant_mass=0.039,
            case_mass=0.051,
            total_impulse=80.0,
            burn_time=2.0,
            average_thrust=40.0,
            max_thrust=65.0,
            time_points=np.array([0.0, 0.05, 0.1, 0.5, 1.0, 1.5, 1.9, 2.0]),
            thrust_points=np.array([0.0, 65.0, 55.0, 45.0, 40.0, 35.0, 20.0, 0.0]),
        ),
        "cesaroni_g79": MotorData(
            motor_id="offline_cesaroni_g79",
            manufacturer="Cesaroni",
            designation="G79",
            common_name="Cesaroni G79",
            diameter=0.029,
            length=0.152,
            total_mass=0.149,
            propellant_mass=0.0625,
            case_mass=0.0865,
            total_impulse=130.0,
            burn_time=1.6,
            average_thrust=79.0,
            max_thrust=110.0,
            time_points=np.array([0.0, 0.02, 0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.6]),
            thrust_points=np.array([0.0, 110.0, 95.0, 85.0, 80.0, 75.0, 70.0, 45.0, 0.0]),
        ),
        "aerotech_h128": MotorData(
            motor_id="offline_aerotech_h128",
            manufacturer="AeroTech",
            designation="H128",
            common_name="Aerotech H128",
            diameter=0.029,
            length=0.194,
            total_mass=0.227,
            propellant_mass=0.094,
            case_mass=0.133,
            total_impulse=219.0,
            burn_time=1.7,
            average_thrust=128.0,
            max_thrust=195.0,
            time_points=np.array([0, 0.02, 0.1, 0.5, 1.0, 1.5, 1.7]),
            thrust_points=np.array([0, 195, 150, 130, 110, 80, 0]),
        ),
    }

    # Normalize key
    key = motor_key.lower().replace(" ", "_").replace("-", "_")

    return motors.get(key)


# ============================================================================
# CLI Interface
# ============================================================================

def cmd_search(args):
    """Search for motors"""
    if not REQUESTS_AVAILABLE:
        print("API search requires 'requests' library. Install with: pip install requests")
        print("\nAvailable offline motors:")
        for key, info in POPULAR_MOTORS.items():
            print(f"  {info['common_name']}")
        return

    api = ThrustCurveAPI()

    results = api.search_motors(
        query=args.query if hasattr(args, 'query') and args.query else None,
        manufacturer=args.manufacturer if hasattr(args, 'manufacturer') and args.manufacturer else None,
        impulse_class=args.impulse_class if hasattr(args, 'impulse_class') and args.impulse_class else None,
        max_results=args.max_results if hasattr(args, 'max_results') else 20,
    )

    if not results:
        print("No motors found matching criteria")
        return

    print(f"\nFound {len(results)} motors:\n")
    print(f"{'ID':<25} {'Manufacturer':<15} {'Designation':<10} {'Class':<5} {'Impulse':<10} {'Avg Thrust':<10}")
    print("-" * 90)

    for r in results:
        print(f"{r.motor_id:<25} {r.manufacturer:<15} {r.designation:<10} {r.impulse_class:<5} "
              f"{r.total_impulse:>6.1f} N·s  {r.avg_thrust:>6.1f} N")


def cmd_list_popular(args):
    """List popular/common motors"""
    print("\nPopular Motors (available offline):\n")
    print(f"{'Key':<20} {'Motor':<25} {'Class'}")
    print("-" * 50)

    for key, info in POPULAR_MOTORS.items():
        # Get class from offline data if available
        motor = get_offline_motor(key)
        if motor:
            print(f"{key:<20} {info['common_name']:<25} {motor.impulse_class}")
        else:
            print(f"{key:<20} {info['common_name']:<25}")


def cmd_verify(args):
    """Verify a motor exists"""
    motor_query = args.motor

    # First check offline database
    offline = get_offline_motor(motor_query)
    if offline:
        print(f"✓ Motor found (offline): {offline.manufacturer} {offline.designation}")
        print(f"  Class: {offline.impulse_class}")
        print(f"  Impulse: {offline.total_impulse:.1f} N·s")
        print(f"  Thrust: {offline.average_thrust:.1f} N avg, {offline.max_thrust:.1f} N max")
        return

    # Try API
    if REQUESTS_AVAILABLE:
        api = ThrustCurveAPI()
        exists, result = api.verify_motor(motor_query)

        if exists:
            print(f"✓ Motor found: {result}")
            print(f"  Motor ID: {result.motor_id}")
        else:
            print(f"✗ Motor not found: {motor_query}")
    else:
        print(f"✗ Motor not found in offline database: {motor_query}")
        print("  Install 'requests' for API search: pip install requests")


def cmd_generate(args):
    """Generate config files"""
    motor_query = args.motor

    # Try to get motor data
    motor = None

    # Check if this looks like a motor ID (24-character hex string)
    is_motor_id = (len(motor_query) == 24 and
                   all(c in '0123456789abcdef' for c in motor_query.lower()))

    if is_motor_id and REQUESTS_AVAILABLE:
        # Direct motor ID - fetch directly (no search metadata available)
        print(f"Fetching motor data for ID: {motor_query}")
        api = ThrustCurveAPI()
        motor = api.get_motor_data(motor_query, search_result=None)
    else:
        # Try offline database first
        motor = get_offline_motor(motor_query)

        # Try API search if not found offline
        if motor is None and REQUESTS_AVAILABLE:
            api = ThrustCurveAPI()
            exists, result = api.verify_motor(motor_query)

            if exists:
                print(f"Found motor via API: {result}")
                # Pass search result for better metadata
                motor = api.get_motor_data(result.motor_id, search_result=result)

    if motor is None:
        print(f"Error: Motor not found: {motor_query}")
        print("\nAvailable offline motors:")
        for key in POPULAR_MOTORS.keys():
            print(f"  {key}")
        return 1

    # Parse difficulties
    if args.difficulty == "all":
        difficulties = ["easy", "medium", "full"]
    else:
        difficulties = [args.difficulty]

    # Generate configs
    filepaths = generate_motor_configs(
        motor=motor,
        output_dir=args.output,
        dry_mass=args.dry_mass,
        difficulties=difficulties,
    )

    print(f"\n{'='*60}")
    print("Config Generation Complete!")
    print(f"{'='*60}")
    print("\nGenerated files:")
    for diff, path in filepaths.items():
        print(f"  {diff}: {path}")

    print("\nNext steps:")
    print(f"  1. Review and adjust parameters if needed")
    print(f"  2. Run training: uv run python train_improved.py --config {list(filepaths.values())[0]}")


def main():
    parser = argparse.ArgumentParser(
        description="Motor Config Generator - Create training configs from ThrustCurve.org data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Search for motors
    python generate_motor_config.py search "F40"
    python generate_motor_config.py search --manufacturer AeroTech
    python generate_motor_config.py search --impulse-class G

    # List popular motors (available offline)
    python generate_motor_config.py list-popular

    # Verify a motor exists
    python generate_motor_config.py verify "Estes C6"

    # Generate configs
    python generate_motor_config.py generate estes_c6
    python generate_motor_config.py generate aerotech_f40 --output configs/
    python generate_motor_config.py generate cesaroni_g79 --difficulty easy --dry-mass 0.8
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for motors")
    search_parser.add_argument("query", nargs="?", help="Search query")
    search_parser.add_argument("--manufacturer", "-m", help="Filter by manufacturer")
    search_parser.add_argument("--impulse-class", "-c", help="Filter by impulse class (A-O)")
    search_parser.add_argument("--max-results", "-n", type=int, default=20, help="Max results")

    # List popular command
    list_parser = subparsers.add_parser("list-popular", help="List popular motors")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a motor exists")
    verify_parser.add_argument("motor", help="Motor name or key")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate config files")
    gen_parser.add_argument("motor", help="Motor name or key (e.g., 'estes_c6', 'Aerotech F40')")
    gen_parser.add_argument("--output", "-o", default="configs", help="Output directory")
    gen_parser.add_argument("--difficulty", "-d", default="all",
                           choices=["easy", "medium", "full", "all"],
                           help="Difficulty level(s) to generate")
    gen_parser.add_argument("--dry-mass", type=float, help="Override dry mass (kg)")

    args = parser.parse_args()

    if args.command == "search":
        cmd_search(args)
    elif args.command == "list-popular":
        cmd_list_popular(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
