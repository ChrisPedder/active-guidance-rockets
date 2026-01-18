// assembly.scad - Complete assembly visualization
// Rocket Fin Servo Mount System

include <config.scad>

// ============================================================================
// ASSEMBLY CONFIGURATION
// ============================================================================

// Show/hide components
show_fin = true;
show_servo_mount = true;
show_servo = true;
show_adapter = true;
show_aileron = true;
show_hinge_pins = true;

// Aileron deflection angle (degrees, positive = trailing edge moves in +Z)
aileron_deflection = 0;  // Try: -30, 0, 30

// Exploded view offset
exploded = 0;  // Set to 1 for exploded view, 0 for assembled

// ============================================================================
// COMPONENT MODULES (simplified for assembly view)
// ============================================================================

// Simplified servo representation
module servo_model() {
    color(color_servo) {
        // Main body
        cube([servo_body_length, servo_body_width, servo_body_height], center = true);
        
        // Mounting tabs
        translate([0, 0, servo_body_height/2 - servo_tab_height])
            cube([servo_body_length, 
                  servo_body_width + servo_tab_extension * 2, 
                  servo_tab_thickness], center = true);
        
        // Output shaft
        translate([servo_body_length/2 - servo_shaft_offset_x, 
                   0, 
                   servo_body_height/2 + servo_shaft_offset_z])
            cylinder(h = 4, d = servo_shaft_diameter, center = false);
        
        // Horn (cross pattern)
        translate([servo_body_length/2 - servo_shaft_offset_x, 
                   0, 
                   servo_body_height/2 + servo_shaft_offset_z + 3])
            horn();
    }
}

module horn() {
    linear_extrude(height = 1.5)
        union() {
            circle(d = 5);
            for (angle = [0, 90, 180, 270])
                rotate([0, 0, angle])
                    hull() {
                        circle(d = 3);
                        translate([6, 0])
                            circle(d = 2);
                    }
        }
}

// Simplified fin section
module fin_section() {
    color(color_fin)
        translate([0, 0, fin_engagement_depth + 20])
            cube([60, aileron_span + 20, fin_thickness], center = true);
}

// Hinge pins
module hinge_pins() {
    color([0.7, 0.7, 0.7])  // Silver
        for (y = [-aileron_hinge_spacing/2, aileron_hinge_spacing/2])
            translate([0, y, 0])
                rotate([0, 90, 0])
                    cylinder(h = hinge_pin_length, d = hinge_pin_diameter, center = true);
}

// ============================================================================
// IMPORT COMPONENT FILES
// ============================================================================

// Import from separate files (or use inline definitions)
use <servo_mount.scad>
use <aileron.scad>
use <servo_arm_adapter.scad>

// ============================================================================
// ASSEMBLY
// ============================================================================

module assembly() {
    // Calculate positions
    pocket_z = -(mount_wall_thickness + mount_pocket_height/2) - fin_engagement_depth;
    shaft_z = pocket_z + mount_pocket_height/2 + servo_shaft_offset_z;
    shaft_x = mount_total_length/2;
    
    // Fin (reference)
    if (show_fin)
        translate([0, 0, exploded * 30])
            fin_section();
    
    // Servo mount housing
    if (show_servo_mount)
        translate([0, 0, exploded * -10])
            servo_mount();
    
    // Servo inside mount
    if (show_servo)
        translate([servo_shaft_offset_x, 0, pocket_z + exploded * -20])
            servo_model();
    
    // Adapter on servo horn
    if (show_adapter)
        translate([shaft_x + 2, 0, shaft_z + 4 + exploded * -30])
            rotate([0, 0, aileron_deflection])  // Rotates with servo
                servo_arm_adapter();
    
    // Aileron
    if (show_aileron)
        translate([shaft_x + 5 + aileron_hinge_offset, 0, shaft_z - 5 + exploded * -40])
            rotate([0, -aileron_deflection, 0])  // Deflection rotation
                rotate([90, 0, 0])
                    aileron();
    
    // Hinge pins
    if (show_hinge_pins)
        translate([shaft_x + 5, 0, shaft_z - 5 + exploded * -35])
            hinge_pins();
}

// ============================================================================
// RENDER ASSEMBLY
// ============================================================================

assembly();

// ============================================================================
// ANNOTATION (for documentation renders)
// ============================================================================

// Uncomment to show coordinate axes
// color([1, 0, 0]) translate([50, 0, 0]) cylinder(h = 2, d = 3);  // X - red
// color([0, 1, 0]) translate([0, 50, 0]) cylinder(h = 2, d = 3);  // Y - green
// color([0, 0, 1]) translate([0, 0, 50]) cylinder(h = 2, d = 3);  // Z - blue

// ============================================================================
// DEMO: Deflection Animation
// ============================================================================

// To create an animation, use OpenSCAD's animation feature:
// Set FPS and Steps in View > Animate
// Then uncomment this line and comment out the static assembly() call above:
//
// aileron_deflection = 30 * sin($t * 360);
// assembly();
