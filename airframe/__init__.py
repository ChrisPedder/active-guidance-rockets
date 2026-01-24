"""
Rocket Airframe Module

Provides classes for defining rocket airframe geometry and physical properties,
separate from training configuration. Supports importing from OpenRocket .ork files.

Example usage:
    from airframe import RocketAirframe

    # Load from OpenRocket file
    airframe = RocketAirframe.load("my_rocket.ork")

    # Or create programmatically
    airframe = RocketAirframe.estes_alpha()

    # Get physical properties
    print(f"Dry mass: {airframe.dry_mass * 1000:.1f} g")
    print(f"Roll inertia: {airframe.get_roll_inertia():.6f} kg*m^2")
"""

from .airframe import RocketAirframe
from .components import (
    Component,
    NoseCone,
    BodyTube,
    TrapezoidFinSet,
    MotorMount,
    MassObject,
    Material,
    NoseConeShape,
    FinCrossSection,
)
from .openrocket_parser import OpenRocketParser

__all__ = [
    "RocketAirframe",
    "OpenRocketParser",
    "Component",
    "NoseCone",
    "BodyTube",
    "TrapezoidFinSet",
    "MotorMount",
    "MassObject",
    "Material",
    "NoseConeShape",
    "FinCrossSection",
]
