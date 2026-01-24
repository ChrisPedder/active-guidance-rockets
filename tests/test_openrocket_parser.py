"""
Tests for OpenRocket .ork file parser.
"""

import pytest
import zipfile
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET


class TestOpenRocketParserBasics:
    """Tests for basic OpenRocketParser functionality."""

    def test_parser_exists(self):
        """Test that OpenRocketParser class exists."""
        from airframe.openrocket_parser import OpenRocketParser

        assert OpenRocketParser is not None

    def test_parse_simple_ork(self, tmp_path):
        """Test parsing a simple .ork file."""
        from airframe.openrocket_parser import OpenRocketParser

        # Create a simple ork file (ZIP with rocket.ork XML inside)
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<openrocket version="1.8">
  <rocket>
    <name>Test Rocket</name>
    <subcomponents>
      <stage>
        <name>Sustainer</name>
        <subcomponents>
          <nosecone>
            <name>Nose Cone</name>
            <length>0.07</length>
            <aftouterdiameter>0.024</aftouterdiameter>
            <thickness>0.002</thickness>
            <shape>ogive</shape>
          </nosecone>
          <bodytube>
            <name>Body Tube</name>
            <length>0.30</length>
            <outerdiameter>0.024</outerdiameter>
            <innerdiameter>0.022</innerdiameter>
            <subcomponents>
              <trapezoidfinset>
                <name>Fins</name>
                <fincount>4</fincount>
                <rootchord>0.05</rootchord>
                <tipchord>0.025</tipchord>
                <height>0.04</height>
                <thickness>0.003</thickness>
              </trapezoidfinset>
            </subcomponents>
          </bodytube>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
        ork_file = tmp_path / "test_rocket.ork"

        # Create ZIP file with XML content
        with zipfile.ZipFile(ork_file, "w") as zf:
            zf.writestr("rocket.ork", xml_content)

        # Parse the file
        airframe = OpenRocketParser.parse(str(ork_file))

        assert airframe.name == "Test Rocket"
        assert len(airframe.components) >= 2  # At least nose cone and body tube

    def test_parse_uncompressed_ork(self, tmp_path):
        """Test parsing plain XML .ork file (not zipped)."""
        from airframe.openrocket_parser import OpenRocketParser

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<openrocket version="1.8">
  <rocket>
    <name>Plain XML Rocket</name>
    <subcomponents>
      <stage>
        <subcomponents>
          <nosecone>
            <name>Nose</name>
            <length>0.05</length>
            <aftouterdiameter>0.02</aftouterdiameter>
            <shape>conical</shape>
          </nosecone>
          <bodytube>
            <name>Body</name>
            <length>0.20</length>
            <outerdiameter>0.02</outerdiameter>
          </bodytube>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
        ork_file = tmp_path / "plain.ork"
        ork_file.write_text(xml_content)

        airframe = OpenRocketParser.parse(str(ork_file))
        assert airframe.name == "Plain XML Rocket"


class TestOpenRocketParserComponents:
    """Tests for parsing different OpenRocket components."""

    def test_parse_nosecone_shapes(self, tmp_path):
        """Test parsing different nosecone shapes."""
        from airframe.openrocket_parser import OpenRocketParser
        from airframe.components import NoseConeShape

        shapes = ["ogive", "conical", "ellipsoid", "parabolic", "haack"]

        for shape in shapes:
            xml_content = f"""<?xml version="1.0"?>
<openrocket>
  <rocket>
    <name>Shape Test</name>
    <subcomponents>
      <stage>
        <subcomponents>
          <nosecone>
            <name>NC</name>
            <length>0.07</length>
            <aftouterdiameter>0.024</aftouterdiameter>
            <shape>{shape}</shape>
          </nosecone>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
            ork_file = tmp_path / f"shape_{shape}.ork"
            with zipfile.ZipFile(ork_file, "w") as zf:
                zf.writestr("rocket.ork", xml_content)

            airframe = OpenRocketParser.parse(str(ork_file))
            # Should parse without error
            assert airframe is not None

    def test_parse_finset_types(self, tmp_path):
        """Test parsing different fin set types."""
        from airframe.openrocket_parser import OpenRocketParser

        xml_content = """<?xml version="1.0"?>
<openrocket>
  <rocket>
    <name>Fin Test</name>
    <subcomponents>
      <stage>
        <subcomponents>
          <bodytube>
            <name>Body</name>
            <length>0.30</length>
            <outerdiameter>0.024</outerdiameter>
            <subcomponents>
              <trapezoidfinset>
                <name>Trapezoidal Fins</name>
                <fincount>4</fincount>
                <rootchord>0.05</rootchord>
                <tipchord>0.025</tipchord>
                <height>0.04</height>
                <sweeplength>0.01</sweeplength>
                <thickness>0.003</thickness>
              </trapezoidfinset>
            </subcomponents>
          </bodytube>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
        ork_file = tmp_path / "fins.ork"
        with zipfile.ZipFile(ork_file, "w") as zf:
            zf.writestr("rocket.ork", xml_content)

        airframe = OpenRocketParser.parse(str(ork_file))

        # Find the fin component
        fins = [c for c in airframe.components if "Fins" in c.name or "Fin" in c.name]
        assert len(fins) >= 1

    def test_parse_motor_mount(self, tmp_path):
        """Test parsing inner tube as motor mount."""
        from airframe.openrocket_parser import OpenRocketParser

        xml_content = """<?xml version="1.0"?>
<openrocket>
  <rocket>
    <name>Motor Mount Test</name>
    <subcomponents>
      <stage>
        <subcomponents>
          <bodytube>
            <name>Body</name>
            <length>0.30</length>
            <outerdiameter>0.024</outerdiameter>
            <subcomponents>
              <innertube>
                <name>Motor Mount</name>
                <motormount>true</motormount>
                <length>0.07</length>
                <outerdiameter>0.018</outerdiameter>
                <innerdiameter>0.016</innerdiameter>
              </innertube>
            </subcomponents>
          </bodytube>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
        ork_file = tmp_path / "mount.ork"
        with zipfile.ZipFile(ork_file, "w") as zf:
            zf.writestr("rocket.ork", xml_content)

        airframe = OpenRocketParser.parse(str(ork_file))

        # Should find motor mount
        mounts = [
            c for c in airframe.components if "Motor" in c.name or "Mount" in c.name
        ]
        assert len(mounts) >= 1

    def test_parse_mass_component(self, tmp_path):
        """Test parsing mass components."""
        from airframe.openrocket_parser import OpenRocketParser

        xml_content = """<?xml version="1.0"?>
<openrocket>
  <rocket>
    <name>Mass Test</name>
    <subcomponents>
      <stage>
        <subcomponents>
          <bodytube>
            <name>Body</name>
            <length>0.30</length>
            <outerdiameter>0.024</outerdiameter>
            <subcomponents>
              <masscomponent>
                <name>Payload</name>
                <mass>0.050</mass>
                <length>0.02</length>
                <axialoffset>0.10</axialoffset>
              </masscomponent>
            </subcomponents>
          </bodytube>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
        ork_file = tmp_path / "mass.ork"
        with zipfile.ZipFile(ork_file, "w") as zf:
            zf.writestr("rocket.ork", xml_content)

        airframe = OpenRocketParser.parse(str(ork_file))

        # Should find mass component
        masses = [c for c in airframe.components if "Payload" in c.name]
        assert len(masses) >= 1


class TestOpenRocketParserHelpers:
    """Tests for helper methods in OpenRocketParser."""

    def test_get_tag_with_namespace(self):
        """Test _get_tag method handles namespaces."""
        from airframe.openrocket_parser import OpenRocketParser

        # With namespace
        elem = ET.Element("{http://example.com}nosecone")
        assert OpenRocketParser._get_tag(elem) == "nosecone"

        # Without namespace
        elem2 = ET.Element("bodytube")
        assert OpenRocketParser._get_tag(elem2) == "bodytube"

    def test_get_float_default(self):
        """Test _get_float with default value."""
        from airframe.openrocket_parser import OpenRocketParser

        parent = ET.Element("parent")

        # Missing element should return default
        result = OpenRocketParser._get_float(parent, "nonexistent", 42.0)
        assert result == 42.0

    def test_get_float_with_value(self):
        """Test _get_float with actual value."""
        from airframe.openrocket_parser import OpenRocketParser

        parent = ET.Element("parent")
        child = ET.SubElement(parent, "length")
        child.text = "0.15"

        result = OpenRocketParser._get_float(parent, "length", 0.0)
        assert result == pytest.approx(0.15, abs=0.001)

    def test_get_float_invalid_value(self):
        """Test _get_float with invalid value returns default."""
        from airframe.openrocket_parser import OpenRocketParser

        parent = ET.Element("parent")
        child = ET.SubElement(parent, "length")
        child.text = "not_a_number"

        result = OpenRocketParser._get_float(parent, "length", 99.0)
        assert result == 99.0

    def test_get_int(self):
        """Test _get_int method."""
        from airframe.openrocket_parser import OpenRocketParser

        parent = ET.Element("parent")
        child = ET.SubElement(parent, "fincount")
        child.text = "4"

        result = OpenRocketParser._get_int(parent, "fincount", 3)
        assert result == 4

    def test_get_text(self):
        """Test _get_text method."""
        from airframe.openrocket_parser import OpenRocketParser

        parent = ET.Element("parent")
        child = ET.SubElement(parent, "name")
        child.text = "Test Name"

        result = OpenRocketParser._get_text(parent, "name", "Default")
        assert result == "Test Name"

    def test_get_text_missing(self):
        """Test _get_text with missing element."""
        from airframe.openrocket_parser import OpenRocketParser

        parent = ET.Element("parent")
        result = OpenRocketParser._get_text(parent, "missing", "Default Value")
        assert result == "Default Value"

    def test_get_bool(self):
        """Test _get_bool method."""
        from airframe.openrocket_parser import OpenRocketParser

        parent = ET.Element("parent")

        # true value
        child = ET.SubElement(parent, "motormount")
        child.text = "true"
        assert OpenRocketParser._get_bool(parent, "motormount", False) is True

        # false value
        child2 = ET.SubElement(parent, "sparky")
        child2.text = "false"
        assert OpenRocketParser._get_bool(parent, "sparky", True) is False

        # missing element
        assert OpenRocketParser._get_bool(parent, "missing", True) is True

    def test_find_element_multiple_paths(self):
        """Test _find_element with multiple path options."""
        from airframe.openrocket_parser import OpenRocketParser

        parent = ET.Element("parent")
        child = ET.SubElement(parent, "name")
        child.text = "Found"

        # Should find with first path
        result = OpenRocketParser._find_element(parent, "name", "other_name")
        assert result is not None
        assert result.text == "Found"

        # Should return None if not found
        result2 = OpenRocketParser._find_element(parent, "nothere", "alsonot")
        assert result2 is None


class TestOpenRocketParserMaterials:
    """Tests for material parsing."""

    def test_parse_material_element(self, tmp_path):
        """Test parsing material from element."""
        from airframe.openrocket_parser import OpenRocketParser

        xml_content = """<?xml version="1.0"?>
<openrocket>
  <rocket>
    <name>Material Test</name>
    <subcomponents>
      <stage>
        <subcomponents>
          <bodytube>
            <name>Body</name>
            <length>0.30</length>
            <outerdiameter>0.024</outerdiameter>
            <material>Fiberglass</material>
          </bodytube>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
        ork_file = tmp_path / "material.ork"
        with zipfile.ZipFile(ork_file, "w") as zf:
            zf.writestr("rocket.ork", xml_content)

        airframe = OpenRocketParser.parse(str(ork_file))
        # Should parse without error
        assert airframe is not None

    def test_parse_material_with_density(self, tmp_path):
        """Test parsing material with density attribute."""
        from airframe.openrocket_parser import OpenRocketParser

        xml_content = """<?xml version="1.0"?>
<openrocket>
  <rocket>
    <name>Density Test</name>
    <subcomponents>
      <stage>
        <subcomponents>
          <bodytube>
            <name>Body</name>
            <length>0.30</length>
            <outerdiameter>0.024</outerdiameter>
            <material density="1200">Custom Material</material>
          </bodytube>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
        ork_file = tmp_path / "density.ork"
        with zipfile.ZipFile(ork_file, "w") as zf:
            zf.writestr("rocket.ork", xml_content)

        airframe = OpenRocketParser.parse(str(ork_file))
        assert airframe is not None


class TestOpenRocketParserMassOverride:
    """Tests for mass override parsing."""

    def test_parse_mass_override(self, tmp_path):
        """Test parsing mass override on components."""
        from airframe.openrocket_parser import OpenRocketParser

        xml_content = """<?xml version="1.0"?>
<openrocket>
  <rocket>
    <name>Override Test</name>
    <subcomponents>
      <stage>
        <subcomponents>
          <nosecone>
            <name>NC</name>
            <length>0.07</length>
            <aftouterdiameter>0.024</aftouterdiameter>
            <overridemass>0.025</overridemass>
          </nosecone>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
        ork_file = tmp_path / "override.ork"
        with zipfile.ZipFile(ork_file, "w") as zf:
            zf.writestr("rocket.ork", xml_content)

        airframe = OpenRocketParser.parse(str(ork_file))
        assert airframe is not None


class TestOpenRocketParserErrors:
    """Tests for error handling in parser."""

    def test_missing_rocket_element(self, tmp_path):
        """Test error when rocket element is missing."""
        from airframe.openrocket_parser import OpenRocketParser

        xml_content = """<?xml version="1.0"?>
<openrocket>
  <simulation>
    <name>No Rocket Here</name>
  </simulation>
</openrocket>
"""
        ork_file = tmp_path / "no_rocket.ork"
        with zipfile.ZipFile(ork_file, "w") as zf:
            zf.writestr("rocket.ork", xml_content)

        with pytest.raises(ValueError, match="Could not find.*rocket.*element"):
            OpenRocketParser.parse(str(ork_file))

    def test_empty_zip_file(self, tmp_path):
        """Test error when ZIP file has no rocket file."""
        from airframe.openrocket_parser import OpenRocketParser

        ork_file = tmp_path / "empty.ork"
        with zipfile.ZipFile(ork_file, "w") as zf:
            zf.writestr("readme.txt", "This is not a rocket file")

        with pytest.raises(ValueError, match="Could not find rocket definition"):
            OpenRocketParser.parse(str(ork_file))


class TestOpenRocketParserTransition:
    """Tests for transition component parsing."""

    def test_parse_transition(self, tmp_path):
        """Test parsing transition elements."""
        from airframe.openrocket_parser import OpenRocketParser

        xml_content = """<?xml version="1.0"?>
<openrocket>
  <rocket>
    <name>Transition Test</name>
    <subcomponents>
      <stage>
        <subcomponents>
          <bodytube>
            <name>Upper Body</name>
            <length>0.15</length>
            <outerdiameter>0.024</outerdiameter>
          </bodytube>
          <transition>
            <name>Transition</name>
            <length>0.05</length>
          </transition>
          <bodytube>
            <name>Lower Body</name>
            <length>0.20</length>
            <outerdiameter>0.040</outerdiameter>
          </bodytube>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
        ork_file = tmp_path / "transition.ork"
        with zipfile.ZipFile(ork_file, "w") as zf:
            zf.writestr("rocket.ork", xml_content)

        airframe = OpenRocketParser.parse(str(ork_file))
        assert airframe is not None
        # Transition is handled for position tracking, not as a separate component


class TestOpenRocketParserNestedStages:
    """Tests for nested stage parsing."""

    def test_parse_nested_stages(self, tmp_path):
        """Test parsing rockets with nested stages."""
        from airframe.openrocket_parser import OpenRocketParser

        xml_content = """<?xml version="1.0"?>
<openrocket>
  <rocket>
    <name>Multi Stage</name>
    <subcomponents>
      <stage>
        <name>Booster</name>
        <subcomponents>
          <bodytube>
            <name>Booster Tube</name>
            <length>0.20</length>
            <outerdiameter>0.024</outerdiameter>
          </bodytube>
          <stage>
            <name>Sustainer</name>
            <subcomponents>
              <nosecone>
                <name>NC</name>
                <length>0.07</length>
                <aftouterdiameter>0.024</aftouterdiameter>
              </nosecone>
            </subcomponents>
          </stage>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
        ork_file = tmp_path / "nested.ork"
        with zipfile.ZipFile(ork_file, "w") as zf:
            zf.writestr("rocket.ork", xml_content)

        airframe = OpenRocketParser.parse(str(ork_file))
        assert airframe is not None


class TestOpenRocketParserAlternateFormats:
    """Tests for alternate file formats and structures."""

    def test_parse_xml_file_in_zip(self, tmp_path):
        """Test parsing when rocket is in .xml file instead of .ork."""
        from airframe.openrocket_parser import OpenRocketParser

        xml_content = """<?xml version="1.0"?>
<openrocket>
  <rocket>
    <name>XML File Test</name>
    <subcomponents>
      <stage>
        <subcomponents>
          <nosecone>
            <name>NC</name>
            <length>0.07</length>
            <aftouterdiameter>0.024</aftouterdiameter>
          </nosecone>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
        ork_file = tmp_path / "test.ork"
        with zipfile.ZipFile(ork_file, "w") as zf:
            zf.writestr("design.xml", xml_content)

        airframe = OpenRocketParser.parse(str(ork_file))
        assert airframe.name == "XML File Test"

    def test_parse_radius_instead_of_diameter(self, tmp_path):
        """Test parsing when radius is specified instead of diameter."""
        from airframe.openrocket_parser import OpenRocketParser

        xml_content = """<?xml version="1.0"?>
<openrocket>
  <rocket>
    <name>Radius Test</name>
    <subcomponents>
      <stage>
        <subcomponents>
          <nosecone>
            <name>NC</name>
            <length>0.07</length>
            <aftradius>0.012</aftradius>
          </nosecone>
          <bodytube>
            <name>Body</name>
            <length>0.25</length>
            <radius>0.012</radius>
          </bodytube>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""
        ork_file = tmp_path / "radius.ork"
        with zipfile.ZipFile(ork_file, "w") as zf:
            zf.writestr("rocket.ork", xml_content)

        airframe = OpenRocketParser.parse(str(ork_file))
        assert airframe is not None
        # Body tube should have diameter ~0.024 (2 * 0.012)
