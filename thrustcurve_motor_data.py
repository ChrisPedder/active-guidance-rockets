import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import xml.etree.ElementTree as ET
import requests
import re
from scipy import integrate, interpolate
import json

@dataclass
class MotorData:
    """Complete motor data from thrustcurve.org"""
    # Basic info
    manufacturer: str
    designation: str
    diameter: float  # mm
    length: float    # mm
    total_mass: float  # g
    propellant_mass: float  # g
    case_mass: float  # g

    # Performance data
    total_impulse: float  # N·s
    burn_time: float  # s
    average_thrust: float  # N
    max_thrust: float  # N

    # Thrust curve data
    time_points: np.ndarray  # seconds
    thrust_points: np.ndarray  # Newtons

    # Optional metadata
    delays: List[float] = None  # Available delay charges
    plugged: bool = False  # Whether motor is plugged (no ejection charge)
    sparky: bool = False  # Whether motor produces sparks
    comments: str = ""

    def __post_init__(self):
        """Validate and compute derived values"""
        # Convert to SI units if needed
        self.diameter /= 1000  # mm to m
        self.length /= 1000    # mm to m
        self.total_mass /= 1000  # g to kg
        self.propellant_mass /= 1000  # g to kg
        self.case_mass /= 1000  # g to kg

        # Compute actual total impulse from curve if not provided
        if len(self.time_points) > 1:
            self.computed_impulse = integrate.simpson(self.thrust_points, self.time_points)

            # Verify against stated impulse
            if abs(self.computed_impulse - self.total_impulse) / self.total_impulse > 0.05:
                print(f"Warning: Computed impulse ({self.computed_impulse:.1f}) differs from "
                      f"stated impulse ({self.total_impulse:.1f}) by more than 5%")

        # Create interpolation function for smooth thrust lookup
        self.thrust_interpolator = interpolate.interp1d(
            self.time_points,
            self.thrust_points,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # Calculate mass flow rate profile
        self._calculate_mass_flow()

    def _calculate_mass_flow(self):
        """Calculate mass flow rate based on thrust curve"""
        # Assuming constant Isp (specific impulse) - typical for solid motors
        # F = dm/dt * Ve, where Ve is exhaust velocity
        # Total impulse = Ve * propellant_mass

        if self.total_impulse > 0 and self.propellant_mass > 0:
            self.exhaust_velocity = self.total_impulse / self.propellant_mass

            # Mass flow rate at each time point
            self.mass_flow_rate = self.thrust_points / self.exhaust_velocity

            # Create interpolator for mass flow
            self.mass_flow_interpolator = interpolate.interp1d(
                self.time_points,
                self.mass_flow_rate,
                kind='linear',
                bounds_error=False,
                fill_value=0.0
            )
        else:
            self.exhaust_velocity = 0
            self.mass_flow_interpolator = lambda t: 0

    def get_thrust(self, time: float) -> float:
        """Get thrust at specific time"""
        return float(self.thrust_interpolator(time))

    def get_mass(self, time: float) -> float:
        """Get current motor mass at specific time"""
        if time <= 0:
            return self.total_mass
        elif time >= self.burn_time:
            return self.case_mass
        else:
            # Integrate mass flow to get propellant consumed
            time_range = np.linspace(0, time, 100)
            mass_consumed = integrate.simpson(
                [self.mass_flow_interpolator(t) for t in time_range],
                time_range
            )
            return self.total_mass - mass_consumed

    def get_cg_shift(self, time: float) -> float:
        """
        Get center of gravity shift as propellant burns.
        Assumes propellant burns from top down (typical for core burners).
        """
        if self.propellant_mass <= 0:
            return 0.0

        prop_fraction = max(0, (self.propellant_mass - self.get_mass(time) + self.case_mass)
                           / self.propellant_mass)

        # CG shifts forward as propellant burns from the top
        # Maximum shift is half the propellant grain length
        max_shift = self.length * 0.3  # Approximate propellant length as 60% of motor
        return max_shift * prop_fraction * 0.5


class ThrustCurveParser:
    """Parser for various thrust curve formats from thrustcurve.org"""

    @staticmethod
    def parse_eng_file(filepath: str) -> MotorData:
        """
        Parse .eng file format (NAR/TRA standard)
        Format: <name> <diam> <len> <delays> <prop_mass> <total_mass> <manufacturer>
        Then: time thrust pairs
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Skip comments
        data_lines = [l.strip() for l in lines if not l.strip().startswith(';')]

        # Parse header
        header = data_lines[0].split()
        designation = header[0]
        diameter = float(header[1])  # mm
        length = float(header[2])    # mm

        # Parse delays (could be multiple like "3-5-7-10")
        delays_str = header[3]
        if delays_str == 'P':
            delays = []
            plugged = True
        else:
            delays = [float(d) for d in delays_str.split('-') if d]
            plugged = False

        propellant_mass = float(header[4])  # g
        total_mass = float(header[5])      # g
        manufacturer = header[6] if len(header) > 6 else "Unknown"

        # Parse thrust curve
        time_points = []
        thrust_points = []

        for line in data_lines[1:]:
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    time_points.append(float(parts[0]))
                    thrust_points.append(float(parts[1]))

        time_array = np.array(time_points)
        thrust_array = np.array(thrust_points)

        # Calculate performance metrics
        total_impulse = integrate.simpson(thrust_array, time_array)
        burn_time = time_array[-1]
        average_thrust = total_impulse / burn_time if burn_time > 0 else 0
        max_thrust = np.max(thrust_array)

        return MotorData(
            manufacturer=manufacturer,
            designation=designation,
            diameter=diameter,
            length=length,
            total_mass=total_mass,
            propellant_mass=propellant_mass,
            case_mass=total_mass - propellant_mass,
            total_impulse=total_impulse,
            burn_time=burn_time,
            average_thrust=average_thrust,
            max_thrust=max_thrust,
            time_points=time_array,
            thrust_points=thrust_array,
            delays=delays,
            plugged=plugged
        )

    @staticmethod
    def parse_rse_file(filepath: str) -> MotorData:
        """Parse RockSim XML format (.rse)"""
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Find engine data
        engine = root.find('.//engine')
        if engine is None:
            raise ValueError("No engine data found in RSE file")

        # Extract data
        manufacturer = engine.find('manufacturer').text
        designation = engine.find('designation').text
        diameter = float(engine.find('diameter').text)  # mm
        length = float(engine.find('length').text)      # mm
        total_mass = float(engine.find('total-mass').text)  # g
        prop_mass = float(engine.find('prop-mass').text)    # g

        # Get thrust curve data
        data_points = engine.find('data')
        time_points = []
        thrust_points = []

        for sample in data_points.findall('sample'):
            time_points.append(float(sample.find('time').text))
            thrust_points.append(float(sample.find('thrust').text))

        time_array = np.array(time_points)
        thrust_array = np.array(thrust_points)

        # Calculate metrics
        total_impulse = integrate.simpson(thrust_array, time_array)
        burn_time = time_array[-1]
        average_thrust = total_impulse / burn_time if burn_time > 0 else 0
        max_thrust = np.max(thrust_array)

        return MotorData(
            manufacturer=manufacturer,
            designation=designation,
            diameter=diameter,
            length=length,
            total_mass=total_mass,
            propellant_mass=prop_mass,
            case_mass=total_mass - prop_mass,
            total_impulse=total_impulse,
            burn_time=burn_time,
            average_thrust=average_thrust,
            max_thrust=max_thrust,
            time_points=time_array,
            thrust_points=thrust_array
        )

    @staticmethod
    def download_from_thrustcurve(motor_id: str, format: str = 'eng') -> MotorData:
        """
        Download motor data directly from thrustcurve.org API

        Example motor_id: "Estes_C6"
        """
        # ThrustCurve.org API endpoint
        api_url = f"https://www.thrustcurve.org/api/v1/motor/{motor_id}/data.{format}"

        response = requests.get(api_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download motor data: {response.status_code}")

        # Save temporarily and parse
        temp_file = f"/tmp/{motor_id}.{format}"
        with open(temp_file, 'w') as f:
            f.write(response.text)

        if format == 'eng':
            return ThrustCurveParser.parse_eng_file(temp_file)
        elif format == 'rse':
            return ThrustCurveParser.parse_rse_file(temp_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
