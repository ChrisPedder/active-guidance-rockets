// servo_mount.scad - Servo mount housing that clamps to fin trailing edge
// Rocket Fin Servo Mount System

include <config.scad>

// ============================================================================
// MAIN MODULE
// ============================================================================

module servo_mount() {
    // Total height of mount below fin
    total_height = mount_pocket_height + mount_wall_thickness * 2 + 5;
    
    difference() {
        // Main body with aerodynamic fairing
        mount_body(total_height);
        
        // Fin slot
        fin_slot();
        
        // Servo pocket
        servo_pocket();
        
        // Wire channel
        wire_channel();
        
        // Servo mounting screw holes (optional)
        servo_screw_holes();
        
        // Shaft exit hole
        shaft_exit();
        
        // Hinge pin holes
        hinge_holes();
    }
    
    // Add retention barbs
    retention_barbs();
}

// ============================================================================
// SUB-MODULES
// ============================================================================

// Main aerodynamic body shape
module mount_body(height) {
    hull() {
        // Front rounded section
        translate([-mount_total_length/2 + mount_fairing_radius, 0, -height/2])
            cylinder(h = height, r = mount_fairing_radius, center = true);
        
        // Rear rounded section
        translate([mount_total_length/2 - mount_fairing_radius, 0, -height/2])
            cylinder(h = height, r = mount_fairing_radius, center = true);
    }
}

// Slot for fin trailing edge
module fin_slot() {
    slot_width = fin_thickness + fin_slot_clearance * 2;
    slot_length = mount_total_length;
    
    // Main slot
    translate([0, 0, fin_engagement_depth/2 + 0.01])
        cube([slot_length + 1, slot_width, fin_engagement_depth], center = true);
    
    // Entry chamfer (45 degrees)
    translate([0, 0, fin_engagement_depth + 1])
        hull() {
            cube([slot_length + 1, slot_width, 0.01], center = true);
            translate([0, 0, 2])
                cube([slot_length + 1, slot_width + 4, 0.01], center = true);
        }
    
    // Compliance slot (allows flex for clip action)
    compliance_depth = 15;
    translate([mount_total_length/4, mount_total_width/2 - 2, -compliance_depth/2])
        cube([1, 5, compliance_depth], center = true);
    translate([mount_total_length/4, -mount_total_width/2 + 2, -compliance_depth/2])
        cube([1, 5, compliance_depth], center = true);
}

// Pocket for servo body
module servo_pocket() {
    // Position: servo shaft should be at rear of mount, below fin
    pocket_z = -(mount_wall_thickness + mount_pocket_height/2) - fin_engagement_depth;
    
    // Main pocket
    translate([servo_shaft_offset_x, 0, pocket_z])
        cube([mount_pocket_length, mount_pocket_width, mount_pocket_height + 1], center = true);
    
    // Tab recesses (wider area at tab height)
    tab_recess_width = servo_body_width + servo_tab_extension * 2 + servo_clearance;
    tab_z = pocket_z + mount_pocket_height/2 - servo_tab_height;
    
    translate([servo_shaft_offset_x, 0, tab_z])
        cube([mount_pocket_length, tab_recess_width, servo_tab_thickness + 1], center = true);
}

// Channel for servo wire
module wire_channel() {
    // Runs from servo pocket up alongside fin slot
    pocket_z = -(mount_wall_thickness + mount_pocket_height/2) - fin_engagement_depth;
    
    // Vertical section
    translate([servo_shaft_offset_x - servo_body_length/2 - wire_channel_diameter/2, 
               0, 
               pocket_z/2])
        cylinder(h = abs(pocket_z) + 5, d = wire_channel_diameter, center = true);
    
    // Horizontal exit (alongside fin)
    translate([servo_shaft_offset_x - servo_body_length/2 - wire_channel_diameter/2,
               0,
               1])
        rotate([0, 90, 0])
            cylinder(h = mount_total_length, d = wire_channel_diameter, center = true);
}

// Optional servo mounting screw holes
module servo_screw_holes() {
    pocket_z = -(mount_wall_thickness + mount_pocket_height/2) - fin_engagement_depth;
    screw_z = pocket_z + mount_pocket_height/2 - servo_tab_height - servo_tab_thickness/2;
    
    // Screw positions (at ends of tabs)
    screw_x_offset = servo_body_length/2 - 3;  // Typical screw position
    
    for (x_sign = [-1, 1]) {
        translate([servo_shaft_offset_x + x_sign * screw_x_offset, 0, screw_z]) {
            // Through hole
            cylinder(h = 20, d = m2_hole_diameter, center = true);
            
            // Access from outside (countersink)
            translate([0, 0, -8])
                cylinder(h = 10, d = 4, center = true);
        }
    }
}

// Exit hole for servo output shaft
module shaft_exit() {
    // Shaft position
    shaft_x = servo_shaft_offset_x + servo_body_length/2 - servo_shaft_offset_x;
    pocket_z = -(mount_wall_thickness + mount_pocket_height/2) - fin_engagement_depth;
    shaft_z = pocket_z + mount_pocket_height/2 + servo_shaft_offset_z;
    
    // Elongated hole to allow shaft rotation clearance
    translate([mount_total_length/2 - mount_fairing_radius/2, 0, shaft_z])
        hull() {
            cylinder(h = 20, d = servo_shaft_diameter + 2, center = true);
            translate([5, 0, 0])
                cylinder(h = 20, d = servo_shaft_diameter + 2, center = true);
        }
}

// Hinge pin holes for aileron attachment
module hinge_holes() {
    pocket_z = -(mount_wall_thickness + mount_pocket_height/2) - fin_engagement_depth;
    shaft_z = pocket_z + mount_pocket_height/2 + servo_shaft_offset_z;
    
    // Holes at each end, aligned with aileron hinge
    for (y = [-aileron_hinge_spacing/2, aileron_hinge_spacing/2]) {
        translate([mount_total_length/2 + 2, y, shaft_z - 5])
            rotate([0, 90, 0])
                cylinder(h = 15, d = hinge_pin_diameter + 0.2, center = true);
    }
}

// Retention barbs for fin clamping
module retention_barbs() {
    barb_z = fin_engagement_depth - 2;
    slot_width = fin_thickness + fin_slot_clearance * 2;
    
    for (y_sign = [-1, 1]) {
        translate([0, y_sign * (slot_width/2 + retention_barb_depth/2), barb_z]) {
            // Small triangular barb
            rotate([0, 90, 0])
                linear_extrude(height = mount_total_length - 10, center = true)
                    polygon([
                        [0, 0],
                        [-retention_barb_height, 0],
                        [-retention_barb_height/2, -y_sign * retention_barb_depth]
                    ]);
        }
    }
}

// ============================================================================
// RENDER
// ============================================================================

// Render the mount
color(color_mount)
    servo_mount();

// Optional: Show servo for reference (comment out for STL export)
// %translate([servo_shaft_offset_x, 0, -(mount_wall_thickness + mount_pocket_height/2) - fin_engagement_depth])
//     cube([servo_body_length, servo_body_width, servo_body_height], center = true);
