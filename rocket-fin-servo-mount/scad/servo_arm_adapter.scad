// servo_arm_adapter.scad - Connects servo horn to aileron
// Rocket Fin Servo Mount System

include <config.scad>

// ============================================================================
// MAIN MODULE
// ============================================================================

module servo_arm_adapter() {
    union() {
        // Horn interface plate (top)
        horn_plate();
        
        // Connecting post
        adapter_post();
        
        // Aileron interface tab (bottom)
        aileron_tab();
    }
}

// ============================================================================
// SUB-MODULES
// ============================================================================

// Plate that attaches to servo horn
module horn_plate() {
    difference() {
        // Circular plate
        cylinder(h = 2, d = adapter_horn_plate_dia, center = false);
        
        // Center hole for servo spline
        translate([0, 0, -0.5])
            cylinder(h = 3, d = 2.5, center = false);
        
        // Screw holes in cross pattern (standard micro servo horn)
        for (angle = [0, 90, 180, 270]) {
            rotate([0, 0, angle])
                translate([adapter_horn_hole_spacing/2, 0, -0.5])
                    cylinder(h = 3, d = adapter_horn_hole_dia, center = false);
        }
    }
}

// Vertical connecting post
module adapter_post() {
    translate([0, 0, -adapter_post_height])
        cylinder(h = adapter_post_height, d = adapter_post_diameter, center = false);
}

// Tab that inserts into aileron slot
module aileron_tab() {
    translate([0, 0, -adapter_post_height - adapter_tab_height])
        difference() {
            // Main tab body
            hull() {
                // Top (connected to post)
                cylinder(h = 1, d = adapter_post_diameter, center = false);
                
                // Bottom (wider for strength)
                translate([0, 0, adapter_tab_height - 1])
                    cube([adapter_tab_depth, adapter_tab_width, 1], center = true);
            }
            
            // Chamfer bottom edges for easy insertion
            translate([0, 0, adapter_tab_height - 0.5])
                for (x_sign = [-1, 1]) {
                    translate([x_sign * adapter_tab_depth/2, 0, 0])
                        rotate([0, x_sign * 45, 0])
                            cube([2, adapter_tab_width + 1, 2], center = true);
                }
        }
}

// ============================================================================
// RENDER
// ============================================================================

// Render the adapter
// Oriented for printing: post vertical (standing up)
color(color_adapter)
    servo_arm_adapter();

// ============================================================================
// VARIATIONS
// ============================================================================

// Longer post for increased moment arm (reduced servo load)
module servo_arm_adapter_long() {
    long_post_height = 12;
    
    union() {
        horn_plate();
        
        translate([0, 0, -long_post_height])
            cylinder(h = long_post_height, d = adapter_post_diameter, center = false);
        
        translate([0, 0, -long_post_height - adapter_tab_height])
            hull() {
                cylinder(h = 1, d = adapter_post_diameter, center = false);
                translate([0, 0, adapter_tab_height - 1])
                    cube([adapter_tab_depth, adapter_tab_width, 1], center = true);
            }
    }
}

// Double-ended adapter (for testing both deflection directions)
module servo_arm_adapter_double() {
    union() {
        horn_plate();
        adapter_post();
        aileron_tab();
        
        // Mirror on opposite side
        rotate([0, 0, 180]) {
            adapter_post();
            aileron_tab();
        }
    }
}

// Adjustable adapter with multiple horn positions
module servo_arm_adapter_adjustable() {
    difference() {
        union() {
            // Elongated plate
            hull() {
                cylinder(h = 2, d = adapter_horn_plate_dia, center = false);
                translate([4, 0, 0])
                    cylinder(h = 2, d = adapter_horn_plate_dia, center = false);
            }
            
            adapter_post();
            aileron_tab();
        }
        
        // Multiple mounting hole positions
        for (x_offset = [0, 2, 4]) {
            translate([x_offset, 0, 0]) {
                // Center hole
                translate([0, 0, -0.5])
                    cylinder(h = 3, d = 2.5, center = false);
                
                // Screw holes
                for (angle = [0, 90, 180, 270]) {
                    rotate([0, 0, angle])
                        translate([adapter_horn_hole_spacing/2, 0, -0.5])
                            cylinder(h = 3, d = adapter_horn_hole_dia, center = false);
                }
            }
        }
    }
}
