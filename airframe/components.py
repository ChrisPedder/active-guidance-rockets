"""
Rocket component definitions for airframe modeling.

Each component represents a physical part of the rocket with geometry,
material properties, and methods to calculate mass and moment of inertia.

All dimensions are in SI units (meters, kilograms).
"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np


class NoseConeShape(Enum):
    """Nose cone shape types"""
    OGIVE = "ogive"
    CONICAL = "conical"
    ELLIPTICAL = "elliptical"
    PARABOLIC = "parabolic"
    POWER_SERIES = "power_series"
    HAACK = "haack"


class FinCrossSection(Enum):
    """Fin cross-section shapes"""
    SQUARE = "square"
    ROUNDED = "rounded"
    AIRFOIL = "airfoil"
    DOUBLE_WEDGE = "double_wedge"


@dataclass
class Material:
    """Material properties for structural components"""
    name: str
    density: float  # kg/m^3

    @classmethod
    def balsa(cls) -> 'Material':
        return cls("Balsa", 160.0)

    @classmethod
    def plywood_birch(cls) -> 'Material':
        return cls("Birch Plywood", 630.0)

    @classmethod
    def fiberglass(cls) -> 'Material':
        return cls("Fiberglass", 1800.0)

    @classmethod
    def cardboard(cls) -> 'Material':
        return cls("Cardboard", 680.0)

    @classmethod
    def abs_plastic(cls) -> 'Material':
        return cls("ABS Plastic", 1050.0)

    @classmethod
    def from_name(cls, name: str) -> 'Material':
        """Get material by name, with fallback to cardboard"""
        materials = {
            'balsa': cls.balsa(),
            'birch plywood': cls.plywood_birch(),
            'plywood': cls.plywood_birch(),
            'fiberglass': cls.fiberglass(),
            'cardboard': cls.cardboard(),
            'abs plastic': cls.abs_plastic(),
            'abs': cls.abs_plastic(),
            'plastic': cls.abs_plastic(),
        }
        return materials.get(name.lower(), cls.cardboard())


@dataclass
class Component:
    """Base class for rocket components"""
    name: str
    position: float  # Distance from nose tip (m)
    mass_override: Optional[float] = None  # kg, if set overrides calculated mass

    def get_mass(self) -> float:
        """Get component mass (override or calculated)"""
        if self.mass_override is not None:
            return self.mass_override
        return self._calculate_mass()

    def _calculate_mass(self) -> float:
        """Calculate mass from geometry and material. Override in subclasses."""
        return 0.0

    def get_cg_position(self) -> float:
        """Get center of gravity position from nose tip"""
        return self.position

    def get_roll_inertia(self, body_radius: float = 0.0) -> float:
        """
        Get moment of inertia about roll axis.

        Args:
            body_radius: Rocket body radius for parallel axis calculations (m)

        Returns:
            Moment of inertia (kg*m^2)
        """
        return 0.0


@dataclass
class NoseCone(Component):
    """Nose cone component"""
    length: float = 0.07  # m
    base_diameter: float = 0.024  # m (outer diameter at base)
    shape: NoseConeShape = NoseConeShape.OGIVE
    shape_parameter: float = 1.0  # Power for power series, etc.
    thickness: float = 0.002  # m, wall thickness
    material: Material = field(default_factory=Material.abs_plastic)

    def _calculate_mass(self) -> float:
        """Approximate as hollow cone"""
        r = self.base_diameter / 2
        slant_height = np.sqrt(self.length**2 + r**2)
        surface_area = np.pi * r * slant_height
        volume = surface_area * self.thickness
        return volume * self.material.density

    def get_cg_position(self) -> float:
        """CG of hollow cone is approximately 2/3 from tip"""
        return self.position + self.length * 0.67

    def get_roll_inertia(self, body_radius: float = 0.0) -> float:
        """Thin-shell cone about axis: approximately (1/2) * m * r^2"""
        m = self.get_mass()
        r = self.base_diameter / 2
        return 0.5 * m * r**2


@dataclass
class BodyTube(Component):
    """Cylindrical body tube"""
    length: float = 0.30  # m
    outer_diameter: float = 0.024  # m
    inner_diameter: float = 0.022  # m
    material: Material = field(default_factory=Material.cardboard)

    @property
    def wall_thickness(self) -> float:
        return (self.outer_diameter - self.inner_diameter) / 2

    def _calculate_mass(self) -> float:
        """Calculate mass of hollow cylinder"""
        r_out = self.outer_diameter / 2
        r_in = self.inner_diameter / 2
        volume = np.pi * self.length * (r_out**2 - r_in**2)
        return volume * self.material.density

    def get_cg_position(self) -> float:
        """CG at center of tube"""
        return self.position + self.length / 2

    def get_roll_inertia(self, body_radius: float = 0.0) -> float:
        """Thin-walled cylinder: I = m * r_avg^2"""
        m = self.get_mass()
        r_avg = (self.outer_diameter + self.inner_diameter) / 4
        return m * r_avg**2


@dataclass
class TrapezoidFinSet(Component):
    """
    Trapezoidal fin set (most common fin shape).

    Represents a set of identical fins symmetrically arranged around the body.
    """
    num_fins: int = 4
    root_chord: float = 0.05  # m
    tip_chord: float = 0.025  # m
    span: float = 0.04  # m (semi-span, from body surface to tip)
    sweep_length: float = 0.0  # m, leading edge sweep
    thickness: float = 0.003  # m
    cross_section: FinCrossSection = FinCrossSection.SQUARE
    material: Material = field(default_factory=Material.plywood_birch)

    @property
    def fin_area(self) -> float:
        """Area of single fin (trapezoid)"""
        return 0.5 * (self.root_chord + self.tip_chord) * self.span

    @property
    def total_fin_area(self) -> float:
        """Total area of all fins"""
        return self.fin_area * self.num_fins

    def _calculate_mass(self) -> float:
        """Calculate total mass of all fins"""
        volume = self.fin_area * self.thickness * self.num_fins
        return volume * self.material.density

    def get_single_fin_mass(self) -> float:
        """Mass of a single fin"""
        return self.get_mass() / self.num_fins

    def get_cg_position(self) -> float:
        """Approximate CG at 40% of root chord from leading edge"""
        return self.position + self.root_chord * 0.4

    def get_roll_inertia(self, body_radius: float) -> float:
        """
        Roll inertia of fin set using parallel axis theorem.

        I = I_cm + m*d^2 where d is distance from roll axis to fin CG.

        Args:
            body_radius: Rocket body radius (m)

        Returns:
            Total moment of inertia for all fins (kg*m^2)
        """
        single_fin_mass = self.get_single_fin_mass()

        # Fin CG distance from body surface (approximately at 40% of span)
        fin_cg_from_surface = self.span * 0.4

        # Distance from roll axis to fin CG
        d = body_radius + fin_cg_from_surface

        # Fin's own inertia about its CG (thin plate perpendicular to roll axis)
        # I_cm approximately (1/12) * m * span^2
        I_cm = (1/12) * single_fin_mass * self.span**2

        # Total for single fin using parallel axis theorem
        I_single = I_cm + single_fin_mass * d**2

        return self.num_fins * I_single

    def get_control_effectiveness(
        self,
        body_radius: float,
        dynamic_pressure: float,
        tab_chord_fraction: float = 0.25,
        tab_span_fraction: float = 0.5,
        num_controlled_fins: int = 2
    ) -> float:
        """
        Calculate roll torque per radian of tab deflection.

        Args:
            body_radius: Rocket body radius (m)
            dynamic_pressure: q = 0.5 * rho * v^2 (Pa)
            tab_chord_fraction: Fraction of chord that is control tab
            tab_span_fraction: Fraction of span with control tab
            num_controlled_fins: Number of fins with active tabs (typically 2)

        Returns:
            Torque per radian of deflection (N*m/rad)
        """
        # Tab area
        tab_chord = self.root_chord * tab_chord_fraction
        tab_span = self.span * tab_span_fraction
        tab_area = tab_chord * tab_span

        # Lift coefficient slope for thin airfoil: Cl_alpha = 2*pi
        Cl_alpha = 2 * np.pi

        # Moment arm from axis to tab center
        moment_arm = body_radius + self.span * 0.5

        # Force per tab per radian
        force_per_rad = Cl_alpha * dynamic_pressure * tab_area

        # Total torque from differential tabs
        num_active = min(num_controlled_fins, self.num_fins)
        return num_active * force_per_rad * moment_arm

    def get_damping_coefficient(self, body_radius: float) -> float:
        """
        Get roll damping coefficient.

        Damping torque = -C_damp * omega * q / V

        Args:
            body_radius: Rocket body radius (m)

        Returns:
            Damping coefficient (m^4)
        """
        moment_arm = body_radius + self.span / 2
        return self.total_fin_area * moment_arm**2


@dataclass
class MotorMount(Component):
    """Motor mount tube (inner tube for holding the motor)"""
    length: float = 0.07  # m
    outer_diameter: float = 0.020  # m
    inner_diameter: float = 0.018  # m (motor diameter)
    material: Material = field(default_factory=Material.cardboard)

    def _calculate_mass(self) -> float:
        """Calculate mass of hollow cylinder"""
        r_out = self.outer_diameter / 2
        r_in = self.inner_diameter / 2
        volume = np.pi * self.length * (r_out**2 - r_in**2)
        return volume * self.material.density

    def get_cg_position(self) -> float:
        """CG at center of mount"""
        return self.position + self.length / 2

    def get_roll_inertia(self, body_radius: float = 0.0) -> float:
        """Small contribution, mostly near centerline"""
        m = self.get_mass()
        r_avg = (self.outer_diameter + self.inner_diameter) / 4
        return 0.5 * m * r_avg**2


@dataclass
class MassObject(Component):
    """
    Generic mass object (payload, electronics, etc.)

    Used for components where we know the mass but not detailed geometry.
    """
    mass: float = 0.01  # kg
    length: float = 0.02  # m (for CG calculation)
    radius_of_gyration: float = 0.01  # m (for inertia estimation)

    def _calculate_mass(self) -> float:
        return self.mass

    def get_cg_position(self) -> float:
        return self.position + self.length / 2

    def get_roll_inertia(self, body_radius: float = 0.0) -> float:
        """Use radius of gyration: I = m * k^2"""
        return self.mass * self.radius_of_gyration**2
