# Design Document: Rocket Fin Servo Mount System

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Component Designs](#component-designs)
3. [Mechanical Analysis](#mechanical-analysis)
4. [Assembly Procedure](#assembly-procedure)
5. [Manufacturing Notes](#manufacturing-notes)

## System Architecture

### Overview

The system consists of three primary components that work together to provide aileron control:

```
                    AIRFLOW DIRECTION
                          ↓
    ┌─────────────────────────────────────────┐
    │                  FIN                     │
    │              (4mm carbon)                │
    │                                          │
    └─────────────────┬───────────────────────┘
                      │
              ┌───────┴───────┐
              │  SERVO MOUNT  │ ← Clamps to trailing edge
              │   HOUSING     │
              │   ┌─────┐     │
              │   │SERVO│     │
              │   └──┬──┘     │
              └──────┼────────┘
                     │
              ┌──────┴──────┐
              │   ADAPTER   │ ← Connects horn to aileron
              └──────┬──────┘
                     │
              ┌──────┴──────┐
              │   AILERON   │ ← 50mm × 25mm control surface
              │  ±30° range │
              └─────────────┘
```

### Coordinate System

Throughout this document, the following coordinate system is used:
- **X-axis:** Parallel to fin chord (positive = trailing edge direction)
- **Y-axis:** Spanwise along fin (positive = outboard)
- **Z-axis:** Normal to fin surface (positive = upper surface)

The origin is located at the intersection of the fin trailing edge and the servo mount centerline.

### Design Philosophy

1. **Aerodynamic priority:** All external surfaces are streamlined. The servo is enclosed in a fairing that minimizes drag.

2. **Structural simplicity:** The fin clamp uses a single-piece design with compliant features rather than multiple fastened parts.

3. **Parametric flexibility:** All dimensions are defined in a central configuration file for easy adaptation to different servos or fin thicknesses.

4. **Print-friendly geometry:** Parts are designed to print without supports where possible, with critical surfaces oriented for optimal layer adhesion.

## Component Designs

### 1. Servo Mount Housing

#### Concept

The servo mount is a streamlined housing that clamps onto the fin trailing edge. It contains a pocket for the servo, positions the servo output shaft at the correct location for aileron actuation, and provides an aerodynamic fairing.

#### Geometry

```
                    TOP VIEW (looking down Z-axis)
                    
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │    ┌─────────────────────────────────────┐     │
    │    │          FIN (4mm thick)            │     │
    │    └─────────────────────────────────────┘     │
    │                      │                          │
    │         ┌────────────┼────────────┐            │
    │         │            │            │            │
    │         │   SERVO    │  POCKET    │            │
    │         │            │            │            │
    │         └────────────┴────────────┘            │
    │                      ○ ← Servo shaft           │
    │                                                 │
    └─────────────────────────────────────────────────┘
    
                    SIDE VIEW (looking along Y-axis)
                    
                         FIN
                          │
                     ┌────┴────┐
                    ╱          ╲
                   ╱   SERVO    ╲
                  │    POCKET    │
                  │   ┌─────┐    │
                  │   │█████│    │
                  │   │SERVO│    │
                  │   │█████│    │
                  │   └──┬──┘    │
                  ╲      ○      ╱  ← Shaft exit
                   ╲          ╱
                    ╲        ╱
                     ────────
```

#### Key Dimensions

| Feature | Dimension | Tolerance | Notes |
|---------|-----------|-----------|-------|
| Overall length | 35mm | ±0.5mm | Along X-axis |
| Overall width | 16mm | ±0.5mm | Along Y-axis |
| Overall height | 32mm | ±0.5mm | Along Z-axis (below fin) |
| Fin slot width | 4.2mm | +0.2/-0mm | Allows for fin thickness variation |
| Fin slot depth | 10mm | ±0.5mm | Clamping engagement |
| Servo pocket | 20.3 × 9.0 × 22.5mm | +0.3mm | Press fit with slight clearance |
| Wall thickness | 2.0mm | min | Structural minimum |
| Fairing radius | 8mm | - | Leading/trailing edge rounding |

#### Fin Clamping Mechanism

The mount uses a compliant clip design with the following features:

1. **Entry chamfer:** 45° chamfer guides fin into slot
2. **Retention tabs:** Small barbs prevent pullout
3. **Compliance slot:** A thin slot in the wall allows flex during installation
4. **Optional screw:** M2 set screw can lock the mount if vibration is a concern

```
    CLAMP DETAIL (Section through fin slot)
    
         ┌─────┬─────┐
         │     │     │
         │  ▼  │  ▼  │ ← Entry chamfers
         │    ╲│╱    │
         │     │     │ ← Fin (4mm)
         │    ╱│╲    │
         │  ▲  │  ▲  │ ← Retention barbs
         │     │     │
         └─────┴─────┘
               │
               ├── Compliance slot
               │
```

#### Servo Retention

The servo is held by:
1. **Friction fit:** Pocket sized for press fit
2. **Tab capture:** Mounting tabs rest on internal ledge
3. **Optional screws:** M2 screw holes align with servo tabs

#### Wire Routing

A 4mm diameter channel runs from the servo pocket to exit at the fin clamp area, allowing the servo cable to route along the fin.

### 2. Aileron

#### Concept

The aileron is a flat rectangular control surface that pivots about its leading edge. It interfaces with the servo via a horn or adapter mechanism.

#### Geometry

```
    AILERON PLAN VIEW
    
    ←───────── 50mm ──────────→
    ┌─────────────────────────┐ ↑
    │                         │ │
    │    Hinge holes (×2)     │ 25mm
    │    ○             ○      │ │
    │                         │ ↓
    └─────────────────────────┘
    
    AILERON SIDE VIEW
    
    ┌───────────────────────────┐
    │                           │ ← 3mm thick
    ○─────────────────────────○─┘
    ↑
    Hinge axis (at leading edge)
```

#### Key Dimensions

| Feature | Dimension | Tolerance | Notes |
|---------|-----------|-----------|-------|
| Span | 50mm | ±0.3mm | Along Y-axis |
| Chord | 25mm | ±0.3mm | Along X-axis |
| Thickness | 3mm | ±0.2mm | Streamlined profile |
| Hinge hole diameter | 2.0mm | ±0.1mm | For M2 pivot pins |
| Hinge hole spacing | 40mm | ±0.2mm | Between centers |
| Leading edge radius | 1.5mm | - | Full round |
| Trailing edge | 0.5mm | - | Sharp for aerodynamics |

#### Profile

The aileron has a symmetric airfoil-like profile:
- Leading edge: Semicircular (R = 1.5mm)
- Maximum thickness at 30% chord
- Trailing edge: Tapered to 0.5mm

#### Servo Interface

The aileron includes a slot for the servo arm adapter at its hinge line. This slot is:
- Width: 3mm (to accept adapter)
- Depth: 5mm
- Location: Center of span

### 3. Servo Arm Adapter

#### Concept

The adapter connects the servo's output horn to the aileron, translating the servo's rotary motion into aileron deflection. It must accommodate the ±30° range requirement.

#### Geometry

```
    ADAPTER TOP VIEW
    
         ┌─────────┐
         │  ○  ○   │ ← Servo horn mount holes
         │    ○    │
         └────┬────┘
              │
              │ ← Vertical post
              │
         ┌────┴────┐
         │         │ ← Aileron interface tab
         └─────────┘
    
    ADAPTER SIDE VIEW
    
              ┌───┐
              │   │ ← Horn interface plate
              │   │
              └─┬─┘
                │
                │ ← 8mm post
                │
              ┌─┴─┐
              │   │ ← Aileron tab (inserts into slot)
              └───┘
```

#### Key Dimensions

| Feature | Dimension | Tolerance | Notes |
|---------|-----------|-----------|-------|
| Horn plate diameter | 7mm | ±0.2mm | Matches micro servo horn |
| Horn hole pattern | Standard micro | - | Typically 4-hole cross |
| Post height | 8mm | ±0.3mm | Provides moment arm |
| Post diameter | 3mm | ±0.1mm | Structural member |
| Aileron tab width | 2.8mm | -0.2mm | Fits in 3mm aileron slot |
| Aileron tab depth | 4mm | ±0.2mm | Engagement into aileron |

#### Motion Analysis

With an 8mm moment arm and ±30° aileron deflection, the servo experiences:

- **Angular travel:** 60° total (well within 90° servo range)
- **Torque multiplication:** ~3:1 (aileron chord / moment arm ratio)
- **Required servo torque at max deflection:** Minimal (aerodynamic loads are small at these scales)

### 4. Hinge Pins

Simple M2×12mm pins or smooth rods serve as hinge pivots. They pass through:
1. Servo mount housing (fixed)
2. Aileron hinge holes (rotating)

## Mechanical Analysis

### Aerodynamic Loads

At typical model rocket speeds (50-100 m/s) and aileron dimensions, the expected loads are:

| Parameter | Value | Calculation |
|-----------|-------|-------------|
| Dynamic pressure (q) | 3000-12000 Pa | ½ρv² at sea level |
| Aileron area | 0.00125 m² | 50mm × 25mm |
| Max lift coefficient | ~1.2 | At 30° deflection |
| Max aerodynamic force | 4.5-18 N | q × A × Cl |
| Moment arm to servo | ~8mm | Adapter post length |
| Max torque at servo | 0.036-0.144 N·m | 3.6-14.4 kg·mm |

The servo's rated torque of 0.5-0.7 kg/cm (5-7 kg·mm) appears marginal at high speeds. Consider:
- Limiting deflection at high speeds (software)
- Using a higher-torque servo if sustained high-speed deflection is needed
- The actual loads are likely lower due to the small size and brief exposure times

### Structural Analysis

Critical load cases:
1. **Acceleration loads:** 10g in any axis
2. **Vibration:** 100Hz motor-induced
3. **Handling:** Assembly and transport loads

| Component | Critical Stress | Safety Factor |
|-----------|-----------------|---------------|
| Fin clamp | Clamping shear | 3× (PETG) |
| Servo pocket | Wall bending | 4× (PETG) |
| Adapter post | Bending at root | 2.5× (PETG) |
| Hinge pins | Shear | 5× (steel) |

### Thermal Analysis

Operating temperature range: -10°C to +60°C

| Material | Tg/Tm | Suitability |
|----------|-------|-------------|
| PLA | ~60°C | Marginal |
| PETG | ~80°C | Good |
| ASA | ~100°C | Excellent |

PETG is recommended as the minimum acceptable material.

## Assembly Procedure

### Required Components

| Item | Quantity | Notes |
|------|----------|-------|
| Servo mount housing | 1 | 3D printed |
| Aileron | 1 | 3D printed |
| Servo arm adapter | 1 | 3D printed |
| 3.7g micro servo | 1 | With standard horn |
| M2×12mm pins | 2 | Hinge pivots |
| M2×6mm screws | 2 | Servo mounting (optional) |
| M2×4mm screws | 2 | Horn to adapter |

### Assembly Steps

1. **Servo Installation**
   - Insert servo into mount pocket, cable exiting through channel
   - Press until mounting tabs seat on internal ledge
   - (Optional) Secure with M2 screws through access holes

2. **Adapter to Servo Horn**
   - Place adapter on servo horn, aligning holes
   - Secure with M2×4mm screws (use thread-locker)
   - Attach horn+adapter assembly to servo spline

3. **Mount to Fin**
   - Slide fin trailing edge into mount slot
   - Push until retention barbs click over fin edge
   - (Optional) Tighten M2 set screw for high-vibration applications

4. **Aileron Installation**
   - Insert M2 pins through mount housing hinge holes
   - Slide aileron onto pins
   - Insert adapter tab into aileron slot
   - Secure with a small dab of CA glue if needed

5. **Functional Check**
   - Power servo and verify full deflection range
   - Check for binding at ±30° positions
   - Verify neutral alignment with fin chord

## Manufacturing Notes

### Print Orientation

| Component | Orientation | Rationale |
|-----------|-------------|-----------|
| Servo mount | Flat side down | Best surface finish on fairing |
| Aileron | Flat (spanwise) | Layer lines parallel to hinge axis |
| Adapter | Post vertical | Strongest axis for bending loads |

### Print Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Layer height | 0.2mm | Balance of speed and quality |
| Perimeters | 3 | Minimum for wall strength |
| Infill | 50% | Good strength-to-weight |
| Infill pattern | Gyroid | Best for all-axis loads |
| Top/bottom layers | 4 | Seals infill |

### Post-Processing

1. **Support removal:** Clean all support material from servo pocket and hinge holes
2. **Hole sizing:** Ream hinge holes with 2mm drill bit
3. **Surface finishing:** Light sanding of fairing surfaces (optional)
4. **Test fit:** Verify servo fits before gluing anything

### Quality Checks

- [ ] Servo slides into pocket with light friction
- [ ] Fin slot accepts 4mm test material
- [ ] Hinge pins rotate freely in housing
- [ ] Adapter tab slides into aileron slot
- [ ] No warping or layer separation visible

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-12 | Claude | Initial design document |
