"""
OpenRocket .ork file parser.

.ork files are ZIP archives containing rocket.ork (XML).
This parser extracts geometry and mass data without requiring Java/OpenRocket.

Note: OpenRocket files have no formal XML schema, so this parser is based on
examining actual .ork files and may need updates for edge cases.
"""
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List
import logging

from .components import (
    Component, NoseCone, BodyTube, TrapezoidFinSet, MotorMount, MassObject,
    Material, NoseConeShape
)

logger = logging.getLogger(__name__)


class OpenRocketParser:
    """
    Parser for OpenRocket .ork files.

    OpenRocket files are ZIP archives containing:
    - rocket.ork: XML document with rocket design
    - (optional) simulation data, decal images, etc.

    XML structure (simplified):
    <openrocket>
      <rocket>
        <name>Rocket Name</name>
        <subcomponents>
          <stage>
            <subcomponents>
              <nosecone>...</nosecone>
              <bodytube>
                <subcomponents>
                  <trapezoidfinset>...</trapezoidfinset>
                  <innertube>...</innertube>
                </subcomponents>
              </bodytube>
            </subcomponents>
          </stage>
        </subcomponents>
      </rocket>
    </openrocket>
    """

    @classmethod
    def parse(cls, ork_path: str) -> 'RocketAirframe':
        """
        Parse .ork file and create RocketAirframe.

        Args:
            ork_path: Path to .ork file

        Returns:
            RocketAirframe with components from file
        """
        from .airframe import RocketAirframe

        ork_path = Path(ork_path)

        # Extract XML from ZIP
        xml_content = cls._extract_xml(ork_path)

        # Parse XML
        root = ET.fromstring(xml_content)

        # Find rocket element
        rocket = cls._find_element(root, './/rocket', './/{*}rocket')
        if rocket is None:
            raise ValueError(f"Could not find <rocket> element in {ork_path}")

        # Extract rocket name
        name_elem = cls._find_element(rocket, 'name', '{*}name')
        rocket_name = name_elem.text if name_elem is not None else "Imported Rocket"

        # Parse components
        components = cls._parse_rocket_components(rocket)

        logger.info(f"Parsed {len(components)} components from {ork_path.name}")

        return RocketAirframe(
            name=rocket_name,
            description=f"Imported from {ork_path.name}",
            components=components,
            source_file=str(ork_path)
        )

    @classmethod
    def _extract_xml(cls, ork_path: Path) -> str:
        """Extract rocket XML from ZIP archive"""
        try:
            with zipfile.ZipFile(ork_path, 'r') as zf:
                names = zf.namelist()

                # Find the main rocket file
                rocket_file = None
                for name in names:
                    # Look for rocket.ork or any .ork file
                    if name == 'rocket.ork' or (name.endswith('.ork') and '/' not in name):
                        rocket_file = name
                        break

                if rocket_file is None:
                    # Try any XML file
                    for name in names:
                        if name.endswith('.xml'):
                            rocket_file = name
                            break

                if rocket_file is None:
                    raise ValueError(f"Could not find rocket definition in {ork_path}")

                return zf.read(rocket_file).decode('utf-8')

        except zipfile.BadZipFile:
            # Maybe it's already plain XML (uncompressed .ork)
            with open(ork_path, 'r', encoding='utf-8') as f:
                return f.read()

    @classmethod
    def _parse_rocket_components(cls, rocket_elem: ET.Element) -> List[Component]:
        """Parse all components from rocket XML"""
        components = []
        current_position = 0.0

        # Find stage (sustainer for single-stage rockets)
        stage = cls._find_element(rocket_elem, './/stage', './/{*}stage')
        if stage is None:
            # Try looking for subcomponents directly
            stage = rocket_elem

        # Process subcomponents
        subcomps = cls._find_element(stage, 'subcomponents', '{*}subcomponents')
        if subcomps is None:
            logger.warning("No subcomponents found in rocket")
            return components

        cls._parse_subcomponents(subcomps, components, current_position)

        return components

    @classmethod
    def _parse_subcomponents(
        cls,
        parent: ET.Element,
        components: List[Component],
        position: float
    ):
        """Recursively parse subcomponents"""
        current_position = position

        for elem in parent:
            tag = cls._get_tag(elem)

            if tag == 'nosecone':
                comp, length = cls._parse_nosecone(elem, current_position)
                if comp:
                    components.append(comp)
                    current_position += length

            elif tag == 'bodytube':
                comp, length = cls._parse_bodytube(elem, current_position)
                if comp:
                    components.append(comp)

                # Parse components inside body tube (fins, motor mount)
                inner_subcomps = cls._find_element(elem, 'subcomponents', '{*}subcomponents')
                if inner_subcomps is not None:
                    cls._parse_bodytube_contents(inner_subcomps, components, current_position)

                current_position += length

            elif tag == 'transition':
                # Transition piece (tapered section between tubes)
                length = cls._get_float(elem, 'length', 0.05)
                current_position += length

            elif tag == 'stage':
                # Nested stage - recurse
                stage_subcomps = cls._find_element(elem, 'subcomponents', '{*}subcomponents')
                if stage_subcomps is not None:
                    cls._parse_subcomponents(stage_subcomps, components, current_position)

    @classmethod
    def _parse_bodytube_contents(
        cls,
        parent: ET.Element,
        components: List[Component],
        bt_position: float
    ):
        """Parse components inside a body tube (fins, motor mount, etc.)"""
        for elem in parent:
            tag = cls._get_tag(elem)

            if tag in ('trapezoidfinset', 'ellipticalfinset', 'freeformfinset'):
                fin = cls._parse_finset(elem, bt_position)
                if fin:
                    components.append(fin)

            elif tag == 'innertube':
                # Check if it's a motor mount
                is_motor_mount = cls._get_bool(elem, 'motormount', False)
                if is_motor_mount:
                    mount = cls._parse_motor_mount(elem, bt_position)
                    if mount:
                        components.append(mount)

            elif tag == 'masscomponent':
                mass_obj = cls._parse_mass_component(elem, bt_position)
                if mass_obj:
                    components.append(mass_obj)

    @classmethod
    def _parse_nosecone(cls, elem: ET.Element, position: float):
        """Parse nosecone element"""
        length = cls._get_float(elem, 'length', 0.07)
        diameter = cls._get_float(elem, 'aftouterdiameter', 0.0)
        if diameter == 0:
            diameter = cls._get_float(elem, 'aftradius', 0.012) * 2

        thickness = cls._get_float(elem, 'thickness', 0.002)

        # Get shape
        shape_str = cls._get_text(elem, 'shape', 'ogive').lower()
        shape_map = {
            'ogive': NoseConeShape.OGIVE,
            'conical': NoseConeShape.CONICAL,
            'cone': NoseConeShape.CONICAL,
            'ellipsoid': NoseConeShape.ELLIPTICAL,
            'elliptical': NoseConeShape.ELLIPTICAL,
            'parabolic': NoseConeShape.PARABOLIC,
            'power': NoseConeShape.POWER_SERIES,
            'haack': NoseConeShape.HAACK,
            'lvhaack': NoseConeShape.HAACK,
        }
        shape = shape_map.get(shape_str, NoseConeShape.OGIVE)

        material = cls._parse_material(elem)
        mass_override = cls._get_mass_override(elem)

        name = cls._get_text(elem, 'name', 'Nose Cone')

        return NoseCone(
            name=name,
            position=position,
            length=length,
            base_diameter=diameter,
            shape=shape,
            thickness=thickness,
            material=material,
            mass_override=mass_override
        ), length

    @classmethod
    def _parse_bodytube(cls, elem: ET.Element, position: float):
        """Parse bodytube element"""
        length = cls._get_float(elem, 'length', 0.30)

        # Try different ways to get diameter
        outer_d = cls._get_float(elem, 'outerdiameter', 0.0)
        if outer_d == 0:
            outer_d = cls._get_float(elem, 'radius', 0.012) * 2

        inner_d = cls._get_float(elem, 'innerdiameter', 0.0)
        if inner_d == 0:
            # Estimate from wall thickness
            thickness = cls._get_float(elem, 'thickness', 0.001)
            inner_d = outer_d - 2 * thickness

        material = cls._parse_material(elem)
        mass_override = cls._get_mass_override(elem)
        name = cls._get_text(elem, 'name', 'Body Tube')

        return BodyTube(
            name=name,
            position=position,
            length=length,
            outer_diameter=outer_d,
            inner_diameter=inner_d,
            material=material,
            mass_override=mass_override
        ), length

    @classmethod
    def _parse_finset(cls, elem: ET.Element, bt_position: float):
        """Parse fin set element"""
        # Position relative to body tube
        position_type = cls._get_text(elem, 'axialmethod', 'absolute')
        axial_offset = cls._get_float(elem, 'axialoffset', 0.0)

        # For now, assume offset is from body tube start
        position = bt_position + axial_offset

        num_fins = cls._get_int(elem, 'fincount', 4)
        root_chord = cls._get_float(elem, 'rootchord', 0.05)
        tip_chord = cls._get_float(elem, 'tipchord', 0.025)
        span = cls._get_float(elem, 'height', 0.04)  # 'height' = semi-span in OR
        if span == 0:
            span = cls._get_float(elem, 'span', 0.04)

        thickness = cls._get_float(elem, 'thickness', 0.003)
        sweep = cls._get_float(elem, 'sweeplength', 0.0)

        material = cls._parse_material(elem)
        mass_override = cls._get_mass_override(elem)
        name = cls._get_text(elem, 'name', 'Fins')

        return TrapezoidFinSet(
            name=name,
            position=position,
            num_fins=num_fins,
            root_chord=root_chord,
            tip_chord=tip_chord,
            span=span,
            sweep_length=sweep,
            thickness=thickness,
            material=material,
            mass_override=mass_override
        )

    @classmethod
    def _parse_motor_mount(cls, elem: ET.Element, bt_position: float):
        """Parse motor mount (inner tube marked as motor mount)"""
        axial_offset = cls._get_float(elem, 'axialoffset', 0.0)
        position = bt_position + axial_offset

        length = cls._get_float(elem, 'length', 0.07)

        outer_d = cls._get_float(elem, 'outerdiameter', 0.0)
        if outer_d == 0:
            outer_d = cls._get_float(elem, 'radius', 0.010) * 2

        inner_d = cls._get_float(elem, 'innerdiameter', 0.0)
        if inner_d == 0:
            thickness = cls._get_float(elem, 'thickness', 0.001)
            inner_d = outer_d - 2 * thickness

        material = cls._parse_material(elem)
        name = cls._get_text(elem, 'name', 'Motor Mount')

        return MotorMount(
            name=name,
            position=position,
            length=length,
            outer_diameter=outer_d,
            inner_diameter=inner_d,
            material=material
        )

    @classmethod
    def _parse_mass_component(cls, elem: ET.Element, bt_position: float):
        """Parse generic mass component"""
        axial_offset = cls._get_float(elem, 'axialoffset', 0.0)
        position = bt_position + axial_offset

        mass = cls._get_float(elem, 'mass', 0.01)
        length = cls._get_float(elem, 'length', 0.02)
        name = cls._get_text(elem, 'name', 'Mass')

        # Estimate radius of gyration from position (assume near centerline)
        rog = cls._get_float(elem, 'radialdirection', 0.005)

        return MassObject(
            name=name,
            position=position,
            mass=mass,
            length=length,
            radius_of_gyration=rog
        )

    @classmethod
    def _parse_material(cls, elem: ET.Element) -> Material:
        """Parse material from element"""
        mat_elem = cls._find_element(elem, 'material', '{*}material')
        if mat_elem is None:
            return Material.cardboard()

        mat_name = mat_elem.text
        if mat_name is None:
            mat_name = mat_elem.get('name', '')

        # Get density if specified
        density = None
        density_attr = mat_elem.get('density')
        if density_attr:
            try:
                density = float(density_attr)
            except ValueError:
                pass

        if density:
            return Material(name=mat_name or 'Custom', density=density)
        else:
            return Material.from_name(mat_name or 'Cardboard')

    @classmethod
    def _get_mass_override(cls, elem: ET.Element) -> Optional[float]:
        """Check for mass override"""
        # Check overridemass element
        override_elem = cls._find_element(elem, 'overridemass', '{*}overridemass')
        if override_elem is not None and override_elem.text:
            try:
                return float(override_elem.text)
            except ValueError:
                pass

        # Check override attribute
        override_mass = cls._get_bool(elem, 'massoverride', False)
        if override_mass:
            mass_val = cls._get_float(elem, 'mass', 0.0)
            if mass_val > 0:
                return mass_val

        return None

    # XML helper methods

    @classmethod
    def _find_element(cls, parent: ET.Element, *paths) -> Optional[ET.Element]:
        """Find element using multiple path options"""
        for path in paths:
            elem = parent.find(path)
            if elem is not None:
                return elem
        return None

    @classmethod
    def _get_tag(cls, elem: ET.Element) -> str:
        """Get tag name without namespace"""
        tag = elem.tag
        if '}' in tag:
            tag = tag.split('}')[1]
        return tag.lower()

    @classmethod
    def _get_float(cls, elem: ET.Element, name: str, default: float = 0.0) -> float:
        """Get float value from child element"""
        child = cls._find_element(elem, name, f'{{*}}{name}')
        if child is not None and child.text:
            try:
                return float(child.text)
            except ValueError:
                pass
        return default

    @classmethod
    def _get_int(cls, elem: ET.Element, name: str, default: int = 0) -> int:
        """Get int value from child element"""
        return int(cls._get_float(elem, name, float(default)))

    @classmethod
    def _get_text(cls, elem: ET.Element, name: str, default: str = '') -> str:
        """Get text value from child element"""
        child = cls._find_element(elem, name, f'{{*}}{name}')
        if child is not None and child.text:
            return child.text
        return default

    @classmethod
    def _get_bool(cls, elem: ET.Element, name: str, default: bool = False) -> bool:
        """Get boolean value from child element"""
        child = cls._find_element(elem, name, f'{{*}}{name}')
        if child is not None and child.text:
            return child.text.lower() in ('true', '1', 'yes')
        return default
