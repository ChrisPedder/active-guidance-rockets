"""
RocketAirframe - Complete physical rocket definition.

Separates geometry and mass properties from training configuration,
allowing airframes to be defined once and reused across experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
import numpy as np

from .components import (
    Component,
    NoseCone,
    BodyTube,
    TrapezoidFinSet,
    MotorMount,
    MassObject,
    Material,
    NoseConeShape,
)


@dataclass
class RocketAirframe:
    """
    Complete physical definition of a rocket airframe.

    This class represents the geometry, mass distribution, and
    aerodynamic properties of a rocket, independent of training config.

    Attributes:
        name: Descriptive name for the airframe
        description: Optional longer description
        components: List of components from nose to tail
        source_file: Path to source file if loaded from .ork or .yaml
    """

    name: str
    description: str = ""
    components: List[Component] = field(default_factory=list)
    source_file: Optional[str] = None

    # Cached reference dimensions
    _body_diameter: Optional[float] = field(default=None, repr=False)
    _total_length: Optional[float] = field(default=None, repr=False)

    def __post_init__(self):
        """Compute derived properties after initialization"""
        self._compute_reference_dimensions()

    def _compute_reference_dimensions(self):
        """Compute body diameter and length from components"""
        if not self.components:
            return

        # Find body tubes to get diameter
        body_tubes = [c for c in self.components if isinstance(c, BodyTube)]
        if body_tubes and self._body_diameter is None:
            self._body_diameter = max(bt.outer_diameter for bt in body_tubes)

        # Also check nose cones for diameter
        nose_cones = [c for c in self.components if isinstance(c, NoseCone)]
        if nose_cones and self._body_diameter is None:
            self._body_diameter = max(nc.base_diameter for nc in nose_cones)

        # Compute total length
        if self._total_length is None:
            max_extent = 0.0
            for comp in self.components:
                if hasattr(comp, "length"):
                    extent = comp.position + comp.length
                    max_extent = max(max_extent, extent)
            self._total_length = max_extent if max_extent > 0 else None

    @property
    def body_diameter(self) -> float:
        """Reference body diameter (m)"""
        return self._body_diameter or 0.024  # Default 24mm

    @property
    def body_radius(self) -> float:
        """Reference body radius (m)"""
        return self.body_diameter / 2

    @property
    def total_length(self) -> float:
        """Total rocket length (m)"""
        return self._total_length or 0.45  # Default 45cm

    @property
    def dry_mass(self) -> float:
        """Total dry mass of all components (kg)"""
        return sum(c.get_mass() for c in self.components)

    @property
    def cg_position(self) -> float:
        """Center of gravity position from nose tip (m)"""
        total_mass = self.dry_mass
        if total_mass == 0:
            return self.total_length / 2

        moment = sum(c.get_mass() * c.get_cg_position() for c in self.components)
        return moment / total_mass

    def get_fin_set(self) -> Optional[TrapezoidFinSet]:
        """Get the primary fin set (first TrapezoidFinSet found)"""
        for comp in self.components:
            if isinstance(comp, TrapezoidFinSet):
                return comp
        return None

    def get_roll_inertia(self, additional_mass: float = 0.0) -> float:
        """
        Calculate moment of inertia about roll (longitudinal) axis.

        Args:
            additional_mass: Extra mass (e.g., motor, propellant) assumed
                            distributed cylindrically at body radius (kg)

        Returns:
            Roll moment of inertia (kg*m^2)
        """
        I_total = 0.0

        for comp in self.components:
            I_total += comp.get_roll_inertia(self.body_radius)

        # Add contribution from additional mass (motor, propellant)
        # Assume cylindrical distribution at body radius
        if additional_mass > 0:
            I_motor = 0.5 * additional_mass * self.body_radius**2
            I_total += I_motor

        # Ensure minimum inertia for numerical stability
        return max(I_total, 1e-6)

    def get_control_effectiveness(
        self,
        dynamic_pressure: float,
        tab_chord_fraction: float = 0.25,
        tab_span_fraction: float = 0.5,
        num_controlled_fins: int = 2,
        cl_alpha: float = 2 * np.pi,
    ) -> float:
        """
        Get roll control torque per radian of tab deflection.

        Args:
            dynamic_pressure: q = 0.5 * rho * v^2 (Pa)
            tab_chord_fraction: Fraction of fin chord that is control tab
            tab_span_fraction: Fraction of fin span with control tab
            num_controlled_fins: Number of fins with active tabs
            cl_alpha: Lift curve slope (rad^-1), default 2*pi

        Returns:
            Control effectiveness (N*m/rad)
        """
        fin_set = self.get_fin_set()
        if fin_set is None:
            return 0.0

        return fin_set.get_control_effectiveness(
            self.body_radius,
            dynamic_pressure,
            tab_chord_fraction,
            tab_span_fraction,
            num_controlled_fins,
            cl_alpha=cl_alpha,
        )

    def get_aerodynamic_damping_coeff(self, cl_alpha: float = 2 * np.pi) -> float:
        """
        Get roll damping coefficient.

        Damping torque = -C_damp * omega * q / V

        Args:
            cl_alpha: Lift curve slope (rad^-1), scales damping proportionally

        Returns:
            Damping coefficient (m^4)
        """
        fin_set = self.get_fin_set()
        if fin_set is None:
            return 0.0

        return fin_set.get_damping_coefficient(self.body_radius, cl_alpha=cl_alpha)

    def get_frontal_area(self) -> float:
        """Get frontal (cross-sectional) area for drag calculations"""
        return np.pi * self.body_radius**2

    def summary(self) -> str:
        """Return a human-readable summary of the airframe"""
        lines = [
            f"Airframe: {self.name}",
            f"  Length: {self.total_length*1000:.1f} mm",
            f"  Diameter: {self.body_diameter*1000:.1f} mm",
            f"  Dry mass: {self.dry_mass*1000:.1f} g",
            f"  CG position: {self.cg_position*1000:.1f} mm from nose",
            f"  Components: {len(self.components)}",
        ]

        fin_set = self.get_fin_set()
        if fin_set:
            lines.append(f"  Fins: {fin_set.num_fins}x, span={fin_set.span*1000:.1f}mm")

        return "\n".join(lines)

    # Serialization methods

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "components": [self._component_to_dict(c) for c in self.components],
        }

    def _component_to_dict(self, comp: Component) -> Dict[str, Any]:
        """Convert a component to dictionary"""
        data = {
            "type": type(comp).__name__,
            "name": comp.name,
            "position": comp.position,
        }

        if comp.mass_override is not None:
            data["mass_override"] = comp.mass_override

        if isinstance(comp, NoseCone):
            data.update(
                {
                    "length": comp.length,
                    "base_diameter": comp.base_diameter,
                    "shape": comp.shape.value,
                    "thickness": comp.thickness,
                    "material": comp.material.name,
                }
            )
        elif isinstance(comp, BodyTube):
            data.update(
                {
                    "length": comp.length,
                    "outer_diameter": comp.outer_diameter,
                    "inner_diameter": comp.inner_diameter,
                    "material": comp.material.name,
                }
            )
        elif isinstance(comp, TrapezoidFinSet):
            data.update(
                {
                    "num_fins": comp.num_fins,
                    "root_chord": comp.root_chord,
                    "tip_chord": comp.tip_chord,
                    "span": comp.span,
                    "sweep_length": comp.sweep_length,
                    "thickness": comp.thickness,
                    "material": comp.material.name,
                }
            )
        elif isinstance(comp, MotorMount):
            data.update(
                {
                    "length": comp.length,
                    "outer_diameter": comp.outer_diameter,
                    "inner_diameter": comp.inner_diameter,
                    "material": comp.material.name,
                }
            )
        elif isinstance(comp, MassObject):
            data.update(
                {
                    "mass": comp.mass,
                    "length": comp.length,
                    "radius_of_gyration": comp.radius_of_gyration,
                }
            )

        return data

    def save_yaml(self, path: str):
        """Save airframe definition to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str) -> "RocketAirframe":
        """Load airframe from YAML file"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data, source_file=str(path))

    @classmethod
    def load(cls, path: str) -> "RocketAirframe":
        """
        Load airframe from file (auto-detect format).

        Supports .ork (OpenRocket) and .yaml/.yml files.

        Args:
            path: Path to airframe file

        Returns:
            RocketAirframe instance
        """
        path = Path(path)

        if path.suffix.lower() == ".ork":
            from .openrocket_parser import OpenRocketParser

            return OpenRocketParser.parse(str(path))
        elif path.suffix.lower() in (".yaml", ".yml"):
            return cls.load_yaml(str(path))
        else:
            raise ValueError(f"Unsupported airframe file format: {path.suffix}")

    @classmethod
    def _from_dict(
        cls, data: Dict[str, Any], source_file: str = None
    ) -> "RocketAirframe":
        """Create airframe from dictionary"""
        components = []

        for comp_data in data.get("components", []):
            comp_data = dict(comp_data)  # Copy to avoid mutation
            comp_type = comp_data.pop("type")

            # Get material if specified
            material_name = comp_data.pop("material", None)
            if material_name:
                material = Material.from_name(material_name)
            else:
                material = None

            if comp_type == "NoseCone":
                shape_str = comp_data.pop("shape", "ogive")
                shape = NoseConeShape(shape_str)
                if material:
                    comp_data["material"] = material
                components.append(NoseCone(**comp_data, shape=shape))

            elif comp_type == "BodyTube":
                if material:
                    comp_data["material"] = material
                components.append(BodyTube(**comp_data))

            elif comp_type == "TrapezoidFinSet":
                if material:
                    comp_data["material"] = material
                components.append(TrapezoidFinSet(**comp_data))

            elif comp_type == "MotorMount":
                if material:
                    comp_data["material"] = material
                components.append(MotorMount(**comp_data))

            elif comp_type == "MassObject":
                components.append(MassObject(**comp_data))

        return cls(
            name=data.get("name", "Unnamed Airframe"),
            description=data.get("description", ""),
            components=components,
            source_file=source_file,
        )

    # Factory methods for common rockets

    @classmethod
    def estes_alpha(cls) -> "RocketAirframe":
        """
        Create Estes Alpha III airframe.

        Classic beginner rocket designed for Estes C6 motors.
        Approximately 31cm long, 24mm diameter.
        """
        return cls(
            name="Estes Alpha III",
            description="Classic beginner rocket for C6 motors",
            components=[
                NoseCone(
                    name="Nose Cone",
                    position=0.0,
                    length=0.07,
                    base_diameter=0.024,
                    shape=NoseConeShape.OGIVE,
                    thickness=0.002,
                    material=Material.abs_plastic(),
                ),
                BodyTube(
                    name="Body Tube",
                    position=0.07,
                    length=0.24,
                    outer_diameter=0.024,
                    inner_diameter=0.022,
                    material=Material.cardboard(),
                ),
                TrapezoidFinSet(
                    name="Fins",
                    position=0.24,
                    num_fins=4,
                    root_chord=0.05,
                    tip_chord=0.025,
                    span=0.04,
                    thickness=0.002,
                    material=Material.balsa(),
                ),
                MotorMount(
                    name="Motor Mount",
                    position=0.24,
                    length=0.07,
                    outer_diameter=0.020,
                    inner_diameter=0.018,
                    material=Material.cardboard(),
                ),
            ],
        )

    @classmethod
    def high_power_minimum_diameter(
        cls, motor_diameter: float = 0.038
    ) -> "RocketAirframe":
        """
        Create a minimum-diameter high-power rocket.

        Args:
            motor_diameter: Motor diameter in meters (default 38mm)
        """
        body_od = motor_diameter + 0.003  # Small clearance
        body_id = motor_diameter + 0.001

        return cls(
            name=f"Min-D {motor_diameter*1000:.0f}mm",
            description=f"Minimum diameter rocket for {motor_diameter*1000:.0f}mm motors",
            components=[
                NoseCone(
                    name="Nose Cone",
                    position=0.0,
                    length=0.15,
                    base_diameter=body_od,
                    shape=NoseConeShape.OGIVE,
                    thickness=0.003,
                    material=Material.fiberglass(),
                ),
                BodyTube(
                    name="Body Tube",
                    position=0.15,
                    length=0.60,
                    outer_diameter=body_od,
                    inner_diameter=body_id,
                    material=Material.fiberglass(),
                ),
                TrapezoidFinSet(
                    name="Fins",
                    position=0.60,
                    num_fins=4,
                    root_chord=0.10,
                    tip_chord=0.05,
                    span=0.06,
                    thickness=0.003,
                    material=Material.fiberglass(),
                ),
            ],
        )
