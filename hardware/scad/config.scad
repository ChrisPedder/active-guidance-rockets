// config.scad - Central configuration for Rocket Fin Servo Mount System
// All dimensions in millimeters

// ============================================================================
// SERVO SPECIFICATIONS (3.7g Micro Servo)
// ============================================================================

servo_body_length = 20.0;      // X dimension
servo_body_width = 8.75;       // Y dimension
servo_body_height = 22.0;      // Z dimension (including gear case)
servo_tab_thickness = 2.0;     // Mounting tab thickness
servo_tab_extension = 2.0;     // How far tabs extend beyond body
servo_tab_height = 15.5;       // Height from bottom to tab underside
servo_shaft_diameter = 4.0;    // Output shaft diameter
servo_shaft_offset_x = 5.0;    // Shaft offset from body center (toward one end)
servo_shaft_offset_z = 2.0;    // Shaft height above body top
servo_cable_diameter = 3.0;    // Cable bundle diameter

// Clearances for servo pocket
servo_clearance = 0.3;         // Extra space around servo body

// ============================================================================
// FIN SPECIFICATIONS
// ============================================================================

fin_thickness = 4.0;           // Carbon fibre fin thickness
fin_slot_clearance = 0.2;      // Extra space in slot for easy insertion
fin_engagement_depth = 10.0;   // How deep fin inserts into mount

// ============================================================================
// AILERON SPECIFICATIONS
// ============================================================================

aileron_span = 50.0;           // Y dimension (along fin span)
aileron_chord = 25.0;          // X dimension (depth in flow direction)
aileron_thickness = 3.0;       // Z dimension (max thickness)
aileron_le_radius = 1.5;       // Leading edge radius (full round)
aileron_te_thickness = 0.5;    // Trailing edge minimum thickness
aileron_max_deflection = 30;   // Maximum deflection in degrees

// Hinge geometry
aileron_hinge_hole_dia = 2.0;  // For M2 pins
aileron_hinge_spacing = 40.0;  // Distance between hinge holes
aileron_hinge_offset = 2.0;    // Distance from leading edge to hinge center

// Servo interface
aileron_slot_width = 3.0;      // Width of adapter slot
aileron_slot_depth = 5.0;      // Depth of adapter slot

// ============================================================================
// SERVO MOUNT HOUSING
// ============================================================================

mount_wall_thickness = 2.0;    // Minimum wall thickness
mount_fairing_radius = 8.0;    // Rounding on leading/trailing edges
mount_total_length = 35.0;     // Overall X dimension
mount_total_width = 16.0;      // Overall Y dimension

// Calculated pocket dimensions (with clearances)
mount_pocket_length = servo_body_length + servo_clearance;
mount_pocket_width = servo_body_width + servo_clearance;
mount_pocket_height = servo_body_height + servo_clearance;

// Wire channel
wire_channel_diameter = 4.0;   // Diameter of cable routing channel

// Retention features
retention_barb_height = 0.8;   // Height of retention barbs
retention_barb_depth = 1.0;    // Depth of barb engagement

// ============================================================================
// SERVO ARM ADAPTER
// ============================================================================

adapter_horn_plate_dia = 7.0;  // Diameter of horn interface plate
adapter_horn_hole_dia = 1.5;   // Holes for horn screws
adapter_horn_hole_spacing = 4.0; // Spacing between horn holes (cross pattern)
adapter_post_height = 8.0;     // Height of connecting post
adapter_post_diameter = 3.0;   // Diameter of post
adapter_tab_width = 2.8;       // Width of aileron interface tab
adapter_tab_depth = 4.0;       // Depth of tab insertion
adapter_tab_height = 5.0;      // Height of tab

// ============================================================================
// HARDWARE
// ============================================================================

m2_hole_diameter = 2.2;        // Clearance hole for M2 screw
m2_tap_diameter = 1.8;         // Tap hole for M2 thread
hinge_pin_diameter = 2.0;      // M2 rod for hinge
hinge_pin_length = 12.0;       // Length of hinge pins

// ============================================================================
// RENDERING SETTINGS
// ============================================================================

$fn = 64;                      // Circle resolution for final renders
$fa = 2;                       // Minimum angle for arc segments
$fs = 0.5;                     // Minimum segment size

// Preview quality (use these for faster rendering during development)
preview_fn = 32;

// ============================================================================
// COLORS (for assembly visualization)
// ============================================================================

color_mount = [0.3, 0.3, 0.35];      // Dark grey
color_aileron = [0.9, 0.9, 0.95];    // Light grey/white
color_adapter = [0.2, 0.5, 0.8];     // Blue
color_servo = [0.1, 0.1, 0.15];      // Near black
color_fin = [0.15, 0.15, 0.15];      // Carbon black

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Convert degrees to radians
function deg2rad(d) = d * PI / 180;

// Calculate hypotenuse
function hyp(a, b) = sqrt(a*a + b*b);
