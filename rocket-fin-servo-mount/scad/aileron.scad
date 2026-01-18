// aileron.scad - Movable control surface for rocket fin
// Rocket Fin Servo Mount System

include <config.scad>

// ============================================================================
// MAIN MODULE
// ============================================================================

module aileron() {
    difference() {
        // Main aileron body with airfoil profile
        aileron_body();
        
        // Hinge holes
        hinge_holes();
        
        // Servo adapter slot
        adapter_slot();
    }
}

// ============================================================================
// SUB-MODULES
// ============================================================================

// Main aileron body with streamlined profile
module aileron_body() {
    // Create airfoil-like cross section extruded along span
    linear_extrude(height = aileron_span, center = true) {
        aileron_profile();
    }
}

// 2D airfoil-like profile for the aileron
module aileron_profile() {
    // Symmetric profile with:
    // - Rounded leading edge
    // - Maximum thickness at ~30% chord
    // - Tapered trailing edge
    
    // Using hull of circles to create smooth shape
    hull() {
        // Leading edge - full round
        translate([aileron_le_radius, 0])
            circle(r = aileron_le_radius);
        
        // Maximum thickness point (at 30% chord)
        translate([aileron_chord * 0.3, 0])
            scale([1, aileron_thickness / aileron_le_radius / 2])
                circle(r = aileron_le_radius);
        
        // Trailing edge - thin
        translate([aileron_chord - aileron_te_thickness/2, 0])
            scale([aileron_te_thickness/2, aileron_te_thickness/2])
                circle(r = 1);
    }
}

// Hinge holes at leading edge
module hinge_holes() {
    // Two holes symmetrically placed
    for (y = [-aileron_hinge_spacing/2, aileron_hinge_spacing/2]) {
        translate([aileron_hinge_offset, 0, y])
            rotate([0, 0, 90])
                rotate([90, 0, 0])
                    cylinder(h = aileron_thickness + 2, d = aileron_hinge_hole_dia, center = true);
    }
}

// Slot for servo arm adapter
module adapter_slot() {
    // Centered slot at hinge line
    translate([aileron_hinge_offset, 0, 0]) {
        // Main slot
        cube([aileron_slot_depth * 2, aileron_slot_width, adapter_tab_height + 1], center = true);
        
        // Widen at bottom for adapter insertion
        translate([0, 0, -adapter_tab_height/2])
            cube([aileron_slot_depth * 2, aileron_slot_width + 1, 2], center = true);
    }
}

// ============================================================================
// RENDER
// ============================================================================

// Render the aileron
// Oriented for printing: flat on bed
color(color_aileron)
    rotate([90, 0, 0])  // Rotate so span is along Y, chord along X
        aileron();

// ============================================================================
// VARIATIONS
// ============================================================================

// Thicker aileron variant (for higher loads)
module aileron_thick() {
    scale([1, 1.5, 1])  // 50% thicker
        aileron();
}

// Aileron with lightening holes (weight reduction)
module aileron_lightened() {
    difference() {
        aileron();
        
        // Lightening holes along span
        for (y = [-aileron_hinge_spacing/3, 0, aileron_hinge_spacing/3]) {
            translate([aileron_chord * 0.5, 0, y])
                rotate([0, 0, 90])
                    cylinder(h = aileron_thickness + 2, d = 8, center = true);
        }
    }
}
