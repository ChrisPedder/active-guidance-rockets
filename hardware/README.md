# Rocket Fin Servo Mount System

A 3D-printable servo mount system for controlling rocket fin ailerons using 3.7g micro servos.

## Overview

This project provides OpenSCAD designs for mounting micro servos to carbon fibre rocket fins to enable active flight control via movable ailerons. The system is designed for:

- **Fin thickness:** 4mm carbon fibre
- **Servo:** 3.7g digital micro servo (20.0 × 8.75 × 22.0 mm)
- **Aileron:** 50mm × 25mm rectangular profile
- **Deflection range:** ±30° from vertical

## Repository Structure

```
rocket-fin-servo-mount/
├── README.md                    # This file
├── docs/
│   ├── DESIGN_BRIEF.md         # Requirements and constraints
│   └── DESIGN_DOCUMENT.md      # Detailed technical design
├── scad/
│   ├── config.scad             # Shared parameters and configuration
│   ├── servo_mount.scad        # Main servo housing that clamps to fin
│   ├── aileron.scad            # Aileron with servo horn interface
│   ├── servo_arm_adapter.scad  # Adapter between servo horn and aileron
│   └── assembly.scad           # Complete assembly for visualization
└── images/                     # Rendered previews (if generated)
```

## Quick Start

1. Install [OpenSCAD](https://openscad.org/)
2. Open `scad/assembly.scad` to view the complete assembly
3. Export individual STL files from each component file for printing

## Specifications

| Parameter | Value |
|-----------|-------|
| Servo weight | 3.7g |
| Servo dimensions | 20.0 × 8.75 × 22.0 mm |
| Servo torque | 0.5 kg/cm @ 3.6V, 0.7 kg/cm @ 4.8V |
| Servo speed | 0.13 sec/60° @ 4.8V |
| Working voltage | 3.6V - 4.8V |
| Fin thickness | 4mm |
| Aileron size | 50mm × 25mm |
| Deflection range | ±30° |

## Documentation

- [Design Brief](docs/DESIGN_BRIEF.md) - Requirements, constraints, and acceptance criteria
- [Design Document](docs/DESIGN_DOCUMENT.md) - Detailed technical specifications and rationale

## Printing Recommendations

- **Material:** PETG or ASA recommended for heat resistance and strength
- **Layer height:** 0.2mm or finer
- **Infill:** 40-60% for structural integrity
- **Orientation:** See individual component files for optimal print orientation

## Assembly

1. Print all components
2. Press-fit or glue the servo into the servo mount housing
3. Attach the servo mount to the fin trailing edge
4. Connect the servo arm adapter to the servo horn
5. Attach the aileron to the servo arm adapter

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or pull request.
