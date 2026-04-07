# Design Brief: Rocket Fin Servo Mount System

## Project Overview

Design a 3D-printable mounting system to attach 3.7g micro servos to 4mm carbon fibre rocket fins for active flight control via movable ailerons.

## Background

Model rockets and high-power rockets can benefit from active stabilization and control surfaces. This system enables the attachment of lightweight micro servos to existing fin designs, allowing real-time adjustment of fin surfaces during flight.

## Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | Mount shall securely clamp onto 4mm thick carbon fibre fin trailing edge | Must |
| FR-02 | Mount shall hold servo in aerodynamically favorable orientation | Must |
| FR-03 | Aileron shall deflect ±30° from vertical (neutral) position | Must |
| FR-04 | Aileron shall have rectangular profile of 50mm × 25mm | Must |
| FR-05 | System shall accommodate specified 3.7g micro servo (20×8.75×22mm) | Must |
| FR-06 | Servo output shaft shall align with aileron hinge axis | Must |
| FR-07 | Mount shall provide cable routing path for servo wires | Should |
| FR-08 | System shall be field-serviceable (replaceable servo) | Should |

### Physical Constraints

| Parameter | Value | Notes |
|-----------|-------|-------|
| Fin material | Carbon fibre | 4mm nominal thickness |
| Fin trailing edge | Flat/horizontal | Mount clamps over trailing edge |
| Servo dimensions | 20.0 × 8.75 × 22.0 mm | Body only, excluding tabs |
| Servo mounting tabs | Standard micro servo pattern | ~2mm thick tabs with screw holes |
| Servo shaft position | Offset from body center | Typical micro servo configuration |
| Cable length | 150mm | Must route without kinking |

### Performance Requirements

| Parameter | Requirement | Rationale |
|-----------|-------------|-----------|
| Deflection range | ±30° minimum | Standard control surface range |
| Structural integrity | Withstand 10g acceleration | Typical rocket acceleration |
| Vibration resistance | Secure at frequencies up to 100Hz | Motor-induced vibration |
| Temperature range | -10°C to +60°C | Ground and flight conditions |

### Aerodynamic Requirements

| Requirement | Description |
|-------------|-------------|
| Minimal drag | Streamlined profile when viewed from airflow direction |
| Smooth transitions | No sharp edges perpendicular to airflow |
| Low profile | Minimize protrusion from fin surface |
| Symmetric neutral | Aileron neutral position aligned with fin chord |

## Servo Specifications

Based on the provided product description:

```
Weight:          3.7g
Servo type:      Digital servo
Motor type:      Coreless (hollow cup)
Dimensions:      20.0 × 8.75 × 22.0 mm
Torque:          0.5 kg/cm @ 3.6V
                 0.7 kg/cm @ 4.8V
Speed:           0.13 sec/60° @ 4.8V
                 0.10 sec/60° @ 6.0V
Working voltage: 3.6V - 4.8V
Working angle:   90° (±45° from center)
Cable length:    150mm
Gear material:   Nylon
```

### Servo Dimensional Analysis

Standard 3.7g micro servo geometry (approximate):
- Body: 20.0mm (L) × 8.75mm (W) × 22.0mm (H including gear case)
- Mounting tabs: Extend ~2mm beyond body width
- Tab holes: ~2mm diameter, spaced for M2 screws
- Output shaft: ~4mm diameter, centered on short axis
- Shaft offset: ~2mm from top of body

## Design Constraints

### Manufacturing Constraints

| Constraint | Value | Notes |
|------------|-------|-------|
| Manufacturing method | FDM 3D printing | Consumer-grade printers |
| Minimum wall thickness | 1.2mm | 3 perimeters at 0.4mm nozzle |
| Minimum feature size | 0.8mm | Reliable printing |
| Maximum overhang | 45° | Without support structures |
| Layer adhesion orientation | Critical loads parallel to layers | Maximize strength |

### Material Considerations

| Material | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| PLA | Easy to print, stiff | Heat sensitive, brittle | Development only |
| PETG | Good strength, heat resistant | Slight flexibility | Recommended |
| ASA | UV stable, heat resistant | Requires enclosure | Outdoor use |
| Nylon | Very strong, flexible | Hygroscopic, difficult | High-stress applications |

### Assembly Constraints

- No specialized tools required beyond hex keys
- Servo must be replaceable without destroying mount
- Aileron must be removable for transport
- All fasteners should be metric (M2, M3)

## Component Breakdown

### 1. Servo Mount Housing

**Purpose:** Secure servo to fin trailing edge in aerodynamic orientation.

**Key features:**
- Clamshell or clip design to clamp 4mm fin
- Integrated servo pocket with retention
- Aerodynamic fairing
- Wire routing channel
- Mounting provisions for M2 screws

### 2. Aileron

**Purpose:** Movable control surface attached to servo.

**Key features:**
- 50mm span × 25mm chord
- Hinge line at leading edge
- Servo horn interface (direct or via adapter)
- Aerodynamic profile
- Lightweight construction

### 3. Servo Arm Adapter (Optional)

**Purpose:** Connect standard servo horn to aileron hinge mechanism.

**Key features:**
- Interfaces with standard micro servo horn
- Provides moment arm for aileron actuation
- Allows fine adjustment of neutral position

## Acceptance Criteria

### Fit Verification

- [ ] Servo fits snugly in mount pocket
- [ ] Mount clamps securely on 4mm material (test with 4mm stock)
- [ ] Aileron achieves ±30° deflection without binding
- [ ] All components assemble without forced fitting

### Function Verification

- [ ] Servo operates through full range when mounted
- [ ] No binding or interference at deflection limits
- [ ] Cable routes cleanly without pinching
- [ ] Aileron returns to neutral when servo centered

### Aerodynamic Verification

- [ ] Smooth profile with no abrupt transitions
- [ ] Minimal frontal area when viewed from flight direction
- [ ] Symmetric appearance at neutral deflection

## Deliverables

1. **OpenSCAD source files** - Parametric designs for all components
2. **Configuration file** - Central parameter definitions
3. **Assembly visualization** - Combined view of all components
4. **Documentation** - This brief and detailed design document

## Revision History

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | 2026-01-12 | Initial design brief |
