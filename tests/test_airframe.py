"""
Tests for airframe module - rocket airframe and components.
"""

import pytest
import numpy as np
from pathlib import Path


class TestMaterial:
    """Tests for Material class."""

    def test_predefined_materials(self):
        """Test predefined material factory methods."""
        from airframe import Material

        balsa = Material.balsa()
        assert balsa.name == "Balsa"
        assert balsa.density == 160.0

        fiberglass = Material.fiberglass()
        assert fiberglass.name == "Fiberglass"
        assert fiberglass.density == 1800.0

        cardboard = Material.cardboard()
        assert cardboard.density == 680.0

    def test_from_name(self):
        """Test creating material by name."""
        from airframe import Material

        balsa = Material.from_name("balsa")
        assert balsa.density == 160.0

        # Case insensitive
        balsa2 = Material.from_name("BALSA")
        assert balsa2.density == 160.0

        # Unknown defaults to cardboard
        unknown = Material.from_name("unknown_material")
        assert unknown.density == 680.0


class TestNoseCone:
    """Tests for NoseCone component."""

    def test_default_values(self):
        """Test NoseCone default values."""
        from airframe import NoseCone

        nc = NoseCone(name="Test", position=0.0)

        assert nc.length == 0.07
        assert nc.base_diameter == 0.024
        assert nc.thickness == 0.002

    def test_mass_calculation(self):
        """Test NoseCone mass calculation."""
        from airframe import NoseCone, Material

        nc = NoseCone(
            name="Test",
            position=0.0,
            length=0.07,
            base_diameter=0.024,
            thickness=0.002,
            material=Material.abs_plastic(),
        )

        mass = nc.get_mass()
        assert mass > 0
        assert mass < 0.1  # Reasonable upper bound

    def test_mass_override(self):
        """Test that mass_override is used when set."""
        from airframe import NoseCone

        nc = NoseCone(name="Test", position=0.0, mass_override=0.05)

        assert nc.get_mass() == 0.05

    def test_cg_position(self):
        """Test CG position calculation."""
        from airframe import NoseCone

        nc = NoseCone(name="Test", position=0.1, length=0.1)

        # CG should be approximately 2/3 from the tip
        cg = nc.get_cg_position()
        assert 0.15 < cg < 0.2

    def test_roll_inertia(self):
        """Test roll inertia calculation."""
        from airframe import NoseCone

        nc = NoseCone(name="Test", position=0.0, mass_override=0.01)

        inertia = nc.get_roll_inertia()
        assert inertia > 0
        assert inertia < 1e-4  # Small value for nose cone


class TestBodyTube:
    """Tests for BodyTube component."""

    def test_wall_thickness(self):
        """Test wall thickness calculation."""
        from airframe import BodyTube

        bt = BodyTube(
            name="Test", position=0.0, outer_diameter=0.024, inner_diameter=0.022
        )

        assert bt.wall_thickness == pytest.approx(0.001, rel=0.01)

    def test_mass_calculation(self):
        """Test body tube mass calculation."""
        from airframe import BodyTube, Material

        bt = BodyTube(
            name="Test",
            position=0.0,
            length=0.3,
            outer_diameter=0.024,
            inner_diameter=0.022,
            material=Material.cardboard(),
        )

        mass = bt.get_mass()
        assert mass > 0
        # Cardboard tube should be light
        assert mass < 0.05

    def test_cg_at_center(self):
        """Test that CG is at center of tube."""
        from airframe import BodyTube

        bt = BodyTube(name="Test", position=0.1, length=0.2)

        cg = bt.get_cg_position()
        assert cg == pytest.approx(0.2, rel=0.01)


class TestTrapezoidFinSet:
    """Tests for TrapezoidFinSet component."""

    def test_fin_area_calculation(self):
        """Test single fin area calculation."""
        from airframe import TrapezoidFinSet

        fins = TrapezoidFinSet(
            name="Test",
            position=0.0,
            num_fins=4,
            root_chord=0.05,
            tip_chord=0.025,
            span=0.04,
        )

        # Trapezoid area = 0.5 * (root + tip) * height
        expected_area = 0.5 * (0.05 + 0.025) * 0.04
        assert fins.fin_area == pytest.approx(expected_area, rel=0.01)

    def test_total_fin_area(self):
        """Test total fin area calculation."""
        from airframe import TrapezoidFinSet

        fins = TrapezoidFinSet(
            name="Test",
            position=0.0,
            num_fins=4,
            root_chord=0.05,
            tip_chord=0.025,
            span=0.04,
        )

        assert fins.total_fin_area == pytest.approx(4 * fins.fin_area, rel=0.01)

    def test_roll_inertia(self):
        """Test roll inertia calculation with parallel axis theorem."""
        from airframe import TrapezoidFinSet

        fins = TrapezoidFinSet(
            name="Test",
            position=0.0,
            num_fins=4,
            span=0.04,
            mass_override=0.01,  # 10g total for all fins
        )

        body_radius = 0.012  # 24mm diameter
        inertia = fins.get_roll_inertia(body_radius)

        assert inertia > 0
        # Should be significant due to fins being far from axis
        assert inertia > 1e-6

    def test_control_effectiveness(self):
        """Test control effectiveness calculation."""
        from airframe import TrapezoidFinSet

        fins = TrapezoidFinSet(
            name="Test", position=0.0, num_fins=4, root_chord=0.05, span=0.04
        )

        body_radius = 0.012
        dynamic_pressure = 1000.0  # Pa

        effectiveness = fins.get_control_effectiveness(
            body_radius,
            dynamic_pressure,
            tab_chord_fraction=0.25,
            tab_span_fraction=0.5,
            num_controlled_fins=2,
        )

        assert effectiveness > 0

    def test_damping_coefficient(self):
        """Test damping coefficient calculation."""
        from airframe import TrapezoidFinSet

        fins = TrapezoidFinSet(
            name="Test",
            position=0.0,
            num_fins=4,
            span=0.04,
            root_chord=0.05,
            tip_chord=0.025,
        )

        body_radius = 0.012
        damping = fins.get_damping_coefficient(body_radius)

        assert damping > 0


class TestMotorMount:
    """Tests for MotorMount component."""

    def test_mass_calculation(self):
        """Test motor mount mass calculation."""
        from airframe import MotorMount

        mount = MotorMount(
            name="Test",
            position=0.0,
            length=0.07,
            outer_diameter=0.020,
            inner_diameter=0.018,
        )

        mass = mount.get_mass()
        assert mass > 0
        assert mass < 0.01  # Should be very light


class TestMassObject:
    """Tests for MassObject component."""

    def test_mass_returned(self):
        """Test that specified mass is returned."""
        from airframe import MassObject

        obj = MassObject(name="Payload", position=0.1, mass=0.05)

        assert obj.get_mass() == 0.05

    def test_roll_inertia_from_radius_of_gyration(self):
        """Test roll inertia uses radius of gyration."""
        from airframe import MassObject

        obj = MassObject(name="Test", position=0.0, mass=0.1, radius_of_gyration=0.01)

        inertia = obj.get_roll_inertia()
        expected = 0.1 * 0.01**2
        assert inertia == pytest.approx(expected, rel=0.01)


class TestRocketAirframe:
    """Tests for RocketAirframe class."""

    def test_estes_alpha_factory(self, estes_alpha_airframe):
        """Test Estes Alpha factory method."""
        airframe = estes_alpha_airframe

        assert airframe.name == "Estes Alpha III"
        assert len(airframe.components) >= 3

    def test_dry_mass(self, estes_alpha_airframe):
        """Test dry mass calculation."""
        airframe = estes_alpha_airframe

        mass = airframe.dry_mass
        assert mass > 0
        # Estes Alpha is typically 30-50g empty
        assert 0.01 < mass < 0.1

    def test_body_diameter(self, estes_alpha_airframe):
        """Test body diameter property."""
        airframe = estes_alpha_airframe

        # Estes Alpha uses 24mm body tubes
        assert airframe.body_diameter == pytest.approx(0.024, rel=0.01)

    def test_total_length(self, estes_alpha_airframe):
        """Test total length calculation."""
        airframe = estes_alpha_airframe

        length = airframe.total_length
        assert length > 0
        # Should be around 30-40cm
        assert 0.2 < length < 0.5

    def test_cg_position(self, estes_alpha_airframe):
        """Test center of gravity position."""
        airframe = estes_alpha_airframe

        cg = airframe.cg_position
        assert cg > 0
        assert cg < airframe.total_length

    def test_get_fin_set(self, estes_alpha_airframe):
        """Test getting fin set from airframe."""
        airframe = estes_alpha_airframe

        fins = airframe.get_fin_set()
        assert fins is not None
        assert fins.num_fins == 4

    def test_roll_inertia(self, estes_alpha_airframe):
        """Test roll inertia calculation."""
        airframe = estes_alpha_airframe

        inertia = airframe.get_roll_inertia()
        assert inertia > 0

        # With additional mass (motor)
        inertia_with_motor = airframe.get_roll_inertia(additional_mass=0.024)
        assert inertia_with_motor > inertia

    def test_frontal_area(self, estes_alpha_airframe):
        """Test frontal area calculation."""
        airframe = estes_alpha_airframe

        area = airframe.get_frontal_area()
        expected = np.pi * (0.024 / 2) ** 2
        assert area == pytest.approx(expected, rel=0.01)

    def test_save_and_load_yaml(self, estes_alpha_airframe, tmp_path):
        """Test saving and loading airframe from YAML."""
        airframe = estes_alpha_airframe
        yaml_path = tmp_path / "airframe.yaml"

        airframe.save_yaml(str(yaml_path))
        assert yaml_path.exists()

        from airframe import RocketAirframe

        loaded = RocketAirframe.load_yaml(str(yaml_path))

        assert loaded.name == airframe.name
        assert len(loaded.components) == len(airframe.components)
        assert loaded.dry_mass == pytest.approx(airframe.dry_mass, rel=0.01)

    def test_summary(self, estes_alpha_airframe):
        """Test summary output."""
        airframe = estes_alpha_airframe

        summary = airframe.summary()
        assert "Estes Alpha III" in summary
        assert "Length:" in summary
        assert "Diameter:" in summary
        assert "Dry mass:" in summary


class TestAirframeLoad:
    """Tests for loading airframes from files."""

    def test_load_yaml(self, tmp_path):
        """Test loading from YAML file."""
        from airframe import RocketAirframe

        yaml_content = """
name: Test Rocket
description: A test rocket
components:
- type: NoseCone
  name: Nose
  position: 0.0
  length: 0.05
  base_diameter: 0.024
  shape: ogive
  thickness: 0.002
  material: ABS Plastic
- type: BodyTube
  name: Body
  position: 0.05
  length: 0.2
  outer_diameter: 0.024
  inner_diameter: 0.022
  material: Cardboard
"""
        yaml_file = tmp_path / "test_rocket.yaml"
        yaml_file.write_text(yaml_content)

        airframe = RocketAirframe.load(str(yaml_file))

        assert airframe.name == "Test Rocket"
        assert len(airframe.components) == 2

    def test_load_unsupported_format(self, tmp_path):
        """Test that unsupported formats raise error."""
        from airframe import RocketAirframe

        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("invalid")

        with pytest.raises(ValueError, match="Unsupported"):
            RocketAirframe.load(str(bad_file))


class TestHighPowerAirframe:
    """Tests for high power rocket airframe."""

    def test_minimum_diameter_factory(self):
        """Test minimum diameter airframe factory."""
        from airframe import RocketAirframe

        airframe = RocketAirframe.high_power_minimum_diameter(motor_diameter=0.038)

        assert airframe.name == "Min-D 38mm"
        # Body should be slightly larger than motor
        assert airframe.body_diameter > 0.038
        assert airframe.body_diameter < 0.045
