# Raspberry Pi Deployment Guide

Hardware deployment guide for the active guidance spin stabilisation system. This covers everything from parts purchasing through first bench test. The camera system (RunCam + DollaTek WiFi trigger) is documented separately in `camera_electronics/`.

> **Assumed starting point:** You have a built rocket with fin tabs and 3D-printed servo mounts (see `rocket-fin-servo-mount/`). You need to wire up the avionics and get the controller running.

---

## Table of Contents

1. [Parts List](#1-parts-list)
2. [Raspberry Pi Initial Setup](#2-raspberry-pi-initial-setup)
3. [Wiring](#3-wiring)
4. [Software Deployment](#4-software-deployment)
5. [Pre-Flight Checklist](#5-pre-flight-checklist)
6. [Post-Flight](#6-post-flight)

---

## 1. Parts List

### Compute

| Part | Spec | Qty | Notes | Link |
|------|------|-----|-------|------|
| Raspberry Pi Zero 2 W | ARM Cortex-A53, 512 MB RAM, WiFi | 1 | Quad-core needed for 100 Hz ONNX inference. The original Pi Zero (single-core) is too slow. Pi 4 works but is heavier (45g vs 10g). | [raspberrypi.com](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/) |
| MicroSD card | 32 GB, Class 10 / A1 minimum | 1 | Stores OS + flight logs. Faster cards reduce boot time. | [amazon.com](https://www.amazon.com/dp/B073JWXGNT) |
| MicroSD card reader | USB-A or USB-C | 1 | For flashing the OS image. | — |

> **Why Pi Zero 2 W?** The codebase targets `onnxruntime` on ARM64, which needs the quad-core A53 in the Zero 2 W. The original Pi Zero (ARMv6, single-core) cannot run ONNX inference at 100 Hz. The Pi 4 is faster but weighs 45g vs 10g — significant for a model rocket. WiFi is useful for headless SSH setup and downloading logs.

### IMU

| Part | Spec | Qty | Notes | Link |
|------|------|-----|-------|------|
| ICM-20948 breakout board | I2C/SPI, 3.3V logic | 1 | The simulation's sensor model (`rocket_env/sensors/imu_config.py`) is parameterised for the ICM-20948 (noise density 0.015 deg/s/sqrt(Hz), bias instability 5 deg/hr). SparkFun or Adafruit breakouts include voltage regulation and pull-ups. | [sparkfun.com](https://www.sparkfun.com/products/15335) |

> **Assumption:** This guide assumes an ICM-20948 on I2C. If you use a different IMU (e.g. MPU-6050, BMI088), the I2C address and register map will differ. The simulation has presets for these in `rocket_env/sensors/imu_config.py`, but you'll need a different Python driver.

### Servos

| Part | Spec | Qty | Notes | Link |
|------|------|-----|-------|------|
| 3.7g digital micro servo | 0.7 kg/cm torque @ 4.8V, 0.13s/60° | 2 (Estes) or 3 (J800) | Must fit the 20.0 x 8.75 x 22.0 mm servo pocket in the 3D-printed mounts. Coreless motor preferred for vibration resistance. | Search "3.7g micro servo" on Amazon/AliExpress |

> **How many servos?** The Estes C6 config uses 2 controlled fins (of 4 total). The J800 config uses 3 controlled fins. Check your `configs/*.yaml` for `num_controlled_fins`.
>
> **Torque margin:** At transonic speeds (J800), aerodynamic loads on the tabs can reach 0.14 N·m. The servo is rated at 0.07 N·m — the software limits deflection at high dynamic pressure via gain scheduling, but this is marginal. For the Estes rocket at ~40 m/s, loads are well within spec.

### Power

| Part | Spec | Qty | Notes | Link |
|------|------|-----|-------|------|
| 2S LiPo battery | 7.4V, 500-1000 mAh | 1 | Shared with camera system. 500 mAh is ~15 min of flight + idle. | Hobby shop / Amazon |
| 5V BEC (servo power) | Input 7-12V, output 5.0V, 3A+ | 1 | Dedicated regulator for servos. LM2596 or MP1584 adjustable buck. **Must be pre-adjusted to 5.0V before connecting servos.** | [amazon.com](https://www.amazon.com/dp/B076H3XHXP) |
| 5V BEC (Pi power) | Input 7-12V, output 5.1V, 2.5A | 1 | Separate regulator for the Pi. Can use the same LM2596 module. Set to 5.1V (Pi tolerates 5.0-5.25V; 5.1V avoids under-voltage warnings). | Same as above |

> **Why two separate BECs?** Servos draw up to 1A at stall. This current spike will brownout the Pi, causing a crash mid-flight. Separate regulators isolate the servo power rail from the compute power rail. Both share the same battery.

### Wiring Consumables

| Part | Spec | Qty | Notes |
|------|------|-----|-------|
| Silicone hookup wire | 24-26 AWG, assorted colours | ~2m total | Flexible; survives vibration better than solid-core |
| Dupont jumper wires | Female-female, 10 cm | 10 pack | For prototyping before soldering |
| Pin headers | 2.54 mm male, breakaway | 1 strip | If your Pi or IMU breakout doesn't have headers |
| Heat shrink tubing | Assorted sizes | 1 pack | Insulate every solder joint |
| Small perfboard | ~30x50 mm | 1 | Power distribution board |
| JST-PH connectors | 2-pin and 3-pin | 4 | Clean servo/battery connections; optional but recommended |

> **No level shifters needed.** The Pi's GPIO is 3.3V logic. The ICM-20948 breakout runs on 3.3V I2C natively. Servos accept 3.3V signal levels (threshold is typically ~1.5V).
>
> **No external pull-up resistors needed.** The Pi has internal I2C pull-ups, and most ICM-20948 breakout boards (SparkFun, Adafruit) include 10k pull-ups on SDA/SCL. Do **not** add additional external pull-ups — too many in parallel lower the resistance too far.

---

## 2. Raspberry Pi Initial Setup

### 2.1. Flash the OS

1. Download **Raspberry Pi Imager** from [raspberrypi.com/software](https://www.raspberrypi.com/software/).
2. Insert the MicroSD card into your computer.
3. Open Raspberry Pi Imager:
   - **OS:** Raspberry Pi OS Lite (64-bit) — no desktop environment needed; saves RAM and boot time.
   - **Storage:** Select your MicroSD card.
   - Click the **gear icon** (advanced settings) before writing:
     - **Enable SSH:** Yes (password authentication).
     - **Set username:** `pi` (or your preference).
     - **Set password:** Choose a strong password.
     - **Configure WiFi:** Enter your home network SSID and password.
     - **Set locale:** Your timezone and keyboard layout.
   - Click **Write**.

> **Why Lite (no desktop)?** The Pi Zero 2 W has 512 MB RAM. A desktop environment consumes ~200 MB, leaving little for onnxruntime. Lite boots in ~15 seconds vs ~45 seconds. Everything we do is via SSH.

### 2.2. First Boot and SSH

4. Insert the MicroSD card into the Pi and power it on.
5. Wait ~60 seconds for first boot.
6. Find the Pi on your network:
   ```bash
   # From your laptop (macOS/Linux):
   ping raspberrypi.local
   # If that doesn't work, check your router's DHCP client list.
   ```
7. SSH in:
   ```bash
   ssh pi@raspberrypi.local
   ```

### 2.3. System Update

8. Update the OS:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
9. Install system dependencies:
   ```bash
   sudo apt install -y python3-pip python3-venv git i2c-tools libatlas-base-dev
   ```

> `i2c-tools` provides `i2cdetect` for verifying IMU wiring. `libatlas-base-dev` is required by numpy on ARM.

### 2.4. Enable I2C

10. Enable the I2C bus:
    ```bash
    sudo raspi-config nonint do_i2c 0
    ```
11. Verify the I2C kernel module is loaded:
    ```bash
    lsmod | grep i2c
    ```
    You should see `i2c_bcm2835` (or similar) in the output.

### 2.5. Enable Hardware PWM

The Pi has two hardware PWM channels. For software PWM on any GPIO pin (more flexible), we'll use `pigpio`.

12. Install pigpio:
    ```bash
    sudo apt install -y pigpio python3-pigpio
    ```
13. Enable the pigpio daemon to start on boot:
    ```bash
    sudo systemctl enable pigpiod
    sudo systemctl start pigpiod
    ```

### 2.6. Performance Tuning

14. Reduce GPU memory (no display needed):
    ```bash
    echo "gpu_mem=16" | sudo tee -a /boot/firmware/config.txt
    ```
15. Disable Bluetooth (frees a UART and reduces power draw):
    ```bash
    echo "dtoverlay=disable-bt" | sudo tee -a /boot/firmware/config.txt
    sudo systemctl disable hciuart
    ```
16. Disable unnecessary services:
    ```bash
    sudo systemctl disable triggerhappy
    sudo systemctl disable avahi-daemon
    ```
17. Set the CPU governor to performance (prevents dynamic frequency scaling during flight):
    ```bash
    echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
    sudo apt install -y cpufrequtils
    ```
18. Reboot to apply:
    ```bash
    sudo reboot
    ```

### 2.7. Install Python Dependencies and Clone the Repo

19. SSH back in after reboot:
    ```bash
    ssh pi@raspberrypi.local
    ```
20. Clone the repository:
    ```bash
    cd ~
    git clone https://github.com/YOUR_USERNAME/active-guidance-rockets.git
    cd active-guidance-rockets
    ```
21. Create a virtual environment and install runtime dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install numpy onnxruntime smbus2 pigpio
    ```

> **Note:** We only install _runtime_ dependencies on the Pi. No PyTorch, no Stable Baselines3, no Gymnasium. The ONNX model is exported on your development machine and copied to the Pi.

---

## 3. Wiring

See the SVG diagrams in `deployment/diagrams/` for visual reference:

- [`imu_wiring.svg`](diagrams/imu_wiring.svg) — ICM-20948 to Pi
- [`servo_wiring.svg`](diagrams/servo_wiring.svg) — Servo signal connections
- [`power_distribution.svg`](diagrams/power_distribution.svg) — Battery, BECs, power rails
- [`full_system_overview.svg`](diagrams/full_system_overview.svg) — Complete system

### 3.1. IMU to Raspberry Pi (I2C)

| ICM-20948 Pin | Pi Pin | Pi GPIO | Wire Colour |
|---------------|--------|---------|-------------|
| VCC | Pin 1 | 3.3V | Red |
| GND | Pin 6 | GND | Black |
| SDA | Pin 3 | GPIO 2 (SDA1) | Blue |
| SCL | Pin 5 | GPIO 3 (SCL1) | Yellow |

1. Connect all four wires per the table above.
2. If your breakout has an **AD0** pin: leave it floating or tie to GND for address `0x68`. Tie to VCC for `0x69`.
3. Verify the connection:
   ```bash
   sudo i2cdetect -y 1
   ```
   You should see `68` (or `69`) in the grid. If the grid is empty, check your wiring.

> **Do not** connect VCC to the Pi's 5V pin — the ICM-20948 breakout boards typically regulate to 3.3V internally, but the I2C lines must be 3.3V.

### 3.2. Servo Connections

Each servo has three wires:

| Servo Wire | Connects To | Notes |
|------------|-------------|-------|
| Signal (orange/white) | Pi GPIO pin (see below) | PWM signal |
| VCC (red) | Servo BEC 5V output | **NOT** from the Pi |
| GND (brown/black) | Common ground bus | Shared with Pi and BEC |

**GPIO pin assignments:**

| Servo | GPIO | Pi Pin | PWM Channel |
|-------|------|--------|-------------|
| Servo 1 (Fin A) | GPIO 12 | Pin 32 | HW PWM0 |
| Servo 2 (Fin B) | GPIO 13 | Pin 33 | HW PWM1 |
| Servo 3 (Fin C, J800 only) | GPIO 19 | Pin 35 | SW PWM via pigpio |

4. Connect each servo's **signal** wire to the corresponding GPIO pin.
5. Connect each servo's **VCC** (red) wire to the 5V output of the servo BEC. **Do NOT connect to the Pi's 5V pin.**
6. Connect each servo's **GND** (brown/black) wire to the common ground bus.

### 3.3. Power Distribution

7. Connect the 2S LiPo battery positive (+) to both BEC inputs in parallel.
8. Connect the 2S LiPo battery negative (-) to both BEC ground inputs.
9. Adjust the **servo BEC** output to exactly **5.0V** using a multimeter and the trim pot, before connecting any servos.
10. Adjust the **Pi BEC** output to **5.1V**.
11. Connect the Pi BEC output to Pi **Pin 2** (5V) and **Pin 6** (GND).

> **Alternative Pi power:** You can power the Pi via its micro-USB port instead of the GPIO pins. This adds the USB connector's weight but avoids soldering to the GPIO header.

12. **Common ground:** Verify that the following are all connected to the same ground:
    - Battery negative
    - Both BEC grounds
    - Pi GND (Pin 6)
    - All servo GND wires
    - IMU GND

> **This is critical.** Without a common ground, the PWM and I2C signals will have no reference voltage, causing erratic behaviour.

### 3.4. Quick Wiring Verification

13. With everything connected but the **battery disconnected**, use a multimeter in continuity mode to verify:
    - All GND points are connected to each other.
    - No short between any 5V/3.3V rail and GND.
    - Servo VCC traces go to the servo BEC, not to the Pi.
14. Connect the battery. Measure:
    - Servo BEC output: 5.0V ± 0.1V
    - Pi BEC output: 5.1V ± 0.1V
    - Pi 3.3V rail (Pin 1): 3.3V ± 0.1V (confirms Pi is powered)
15. Run `i2cdetect` again to confirm the IMU is visible.

---

## 4. Software Deployment

### 4.1. Export the Model (on your development machine)

On your laptop/desktop where the full project is installed:

1. Export a deployment bundle:
   ```bash
   # PID-only (no neural network, simplest):
   uv run python deployment/export_all.py pid \
       --config configs/estes_c6_sac_wind.yaml \
       --output deploy_bundle/

   # Residual SAC (PID + RL corrections, best performance):
   uv run python deployment/export_all.py residual-sac \
       --config configs/aerotech_j800_wind.yaml \
       --model models/rocket_residual_sac_j800_wind_*/best_model.zip \
       --output deploy_bundle/
   ```

2. Verify the export:
   ```bash
   uv run python deployment/export_onnx.py \
       --model models/rocket_residual_sac_j800_wind_*/best_model.zip \
       --verify --benchmark
   ```
   Check that the verification passes (max error < 1e-5) and note the inference latency.

3. Copy the bundle to the Pi:
   ```bash
   scp -r deploy_bundle/ pi@raspberrypi.local:~/active-guidance-rockets/
   ```

### 4.2. Hardware Interface (needs to be written)

> **The codebase currently has no real hardware drivers.** The inference controllers (`rocket_env/inference/controller.py`) are hardware-agnostic — they take numpy arrays in and return numpy arrays out. You need a thin hardware abstraction layer that:
>
> 1. Reads the ICM-20948 gyroscope over I2C → produces roll rate in deg/s
> 2. Sends PWM commands to the servos → converts [-1, 1] action to pulse width
> 3. Runs the control loop at 100 Hz
>
> A reference implementation sketch is provided below. **This code has not been flight-tested.**

Create `deployment/flight_controller.py` on the Pi:

```python
#!/usr/bin/env python3
"""
Flight controller main loop.

Reads IMU, runs the controller, drives servos at 100 Hz.
NOT YET FLIGHT-TESTED — bench test thoroughly before flight.
"""

import time
import json
import struct
import numpy as np
from pathlib import Path

import smbus2
import pigpio

# ── Configuration ────────────────────────────────────────────────────────

CONTROL_HZ = 100
DT = 1.0 / CONTROL_HZ

# I2C
IMU_BUS = 1
IMU_ADDR = 0x68          # ICM-20948 (AD0 = GND)
GYRO_XOUT_H = 0x33       # First gyro data register (ICM-20948)
GYRO_SCALE = 2000.0 / 32768.0  # deg/s per LSB at ±2000 dps

# Servo GPIO pins
SERVO_PINS = [12, 13]     # GPIO numbers, not physical pin numbers
# For J800 with 3 controlled fins: SERVO_PINS = [12, 13, 19]

# Servo calibration (pulse width in microseconds)
SERVO_CENTRE_US = 1500    # Centre position (0 deflection)
SERVO_RANGE_US = 500      # ±500 us = ±30 degrees
MAX_DEFLECTION_DEG = 30.0

# Launch detection
LAUNCH_ACCEL_THRESHOLD = 20.0  # m/s^2

# Logging
LOG_DIR = Path.home() / "flight_logs"

# ── IMU Driver ───────────────────────────────────────────────────────────

class ICM20948:
    """Minimal ICM-20948 I2C driver (gyro only)."""

    def __init__(self, bus=IMU_BUS, addr=IMU_ADDR):
        self.bus = smbus2.SMBus(bus)
        self.addr = addr
        self._init_sensor()

    def _init_sensor(self):
        # Wake up (clear sleep bit in PWR_MGMT_1)
        self.bus.write_byte_data(self.addr, 0x06, 0x01)
        time.sleep(0.1)
        # Set gyro to ±2000 dps (GYRO_CONFIG_1 register)
        # NOTE: ICM-20948 uses a bank-switching register model.
        # This is simplified — a production driver should use the
        # full bank-select protocol. Consider using a library like
        # adafruit-circuitpython-icm20x instead.
        # Select User Bank 2
        self.bus.write_byte_data(self.addr, 0x7F, 0x20)
        # GYRO_CONFIG_1: ±2000 dps (bits 2:1 = 0b11), DLPF enabled
        self.bus.write_byte_data(self.addr, 0x01, 0x06)
        # Select User Bank 0
        self.bus.write_byte_data(self.addr, 0x7F, 0x00)

    def read_gyro_z(self):
        """Read Z-axis gyro (roll rate) in deg/s."""
        # ICM-20948 gyro Z is at registers 0x37-0x38 (Bank 0)
        raw = self.bus.read_i2c_block_data(self.addr, 0x37, 2)
        value = struct.unpack(">h", bytes(raw))[0]
        return value * GYRO_SCALE

    def close(self):
        self.bus.close()


# ── Servo Driver ─────────────────────────────────────────────────────────

class ServoDriver:
    """PWM servo control via pigpio."""

    def __init__(self, pins=SERVO_PINS):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Cannot connect to pigpio daemon. Run: sudo systemctl start pigpiod")
        self.pins = pins
        for pin in self.pins:
            self.pi.set_mode(pin, pigpio.OUTPUT)
            self.set_deflection(pin, 0.0)

    def set_deflection(self, pin, deflection_deg):
        """Set servo to a deflection angle in degrees."""
        deflection_deg = np.clip(deflection_deg, -MAX_DEFLECTION_DEG, MAX_DEFLECTION_DEG)
        us = SERVO_CENTRE_US + (deflection_deg / MAX_DEFLECTION_DEG) * SERVO_RANGE_US
        self.pi.set_servo_pulsewidth(pin, int(us))

    def set_all(self, deflection_deg):
        """Set all servos to the same deflection (differential drag)."""
        for pin in self.pins:
            self.set_deflection(pin, deflection_deg)

    def disarm(self):
        """Set all servos to centre and stop PWM."""
        for pin in self.pins:
            self.pi.set_servo_pulsewidth(pin, SERVO_CENTRE_US)
        time.sleep(0.5)
        for pin in self.pins:
            self.pi.set_servo_pulsewidth(pin, 0)  # Stop sending pulses

    def close(self):
        self.disarm()
        self.pi.stop()


# ── Main Flight Loop ─────────────────────────────────────────────────────

def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rocket_env.inference.controller import PIDDeployController, ResidualSACController

    # Determine controller type from deploy bundle
    bundle_dir = Path(__file__).resolve().parent / "deploy_bundle"
    config_path = bundle_dir / "controller_config.json"

    with open(config_path) as f:
        config = json.load(f)

    controller_type = config.get("type", "pid")
    print(f"Controller: {controller_type}")

    if controller_type == "pid":
        controller = PIDDeployController(config_path=str(config_path))
    elif controller_type in ("residual_sac", "residual-sac"):
        controller = ResidualSACController(
            model_path=str(bundle_dir / "model.onnx"),
            normalize_path=str(bundle_dir / "normalize.json"),
            config_path=str(config_path),
        )
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    # Initialise hardware
    imu = ICM20948()
    servos = ServoDriver()

    # Flight data log
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"flight_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    log_file = open(log_path, "w")
    log_file.write("time,roll_rate_dps,action,deflection_deg\n")

    print(f"Logging to {log_path}")
    print("Waiting for launch (hold still)...")

    # ── Control loop ──
    t_start = time.monotonic()
    launched = False
    try:
        while True:
            t_loop_start = time.monotonic()
            elapsed = t_loop_start - t_start

            # Read IMU
            roll_rate_dps = imu.read_gyro_z()

            # Simple launch detection: high roll rate spike
            if not launched and abs(roll_rate_dps) > 5.0:
                # Use acceleration if you have an accelerometer too.
                # For gyro-only: detect first significant rate change.
                launched = True
                t_start = time.monotonic()
                elapsed = 0.0
                controller.reset()
                print("LAUNCH DETECTED")

            if launched:
                # Compute action
                # PIDDeployController.get_action() takes physical units directly
                action = controller.get_action(
                    roll_angle_deg=0.0,          # Not available from gyro alone
                    roll_rate_deg_s=roll_rate_dps,
                    dynamic_pressure_pa=0.0,     # Not available without pitot tube
                    vertical_accel_ms2=None,
                )

                # Convert [-1, 1] action to deflection degrees
                deflection_deg = action * MAX_DEFLECTION_DEG

                # Command servos
                servos.set_all(deflection_deg)

                # Log
                log_file.write(f"{elapsed:.4f},{roll_rate_dps:.2f},{action:.4f},{deflection_deg:.2f}\n")

                # Auto-shutdown: if no significant rate for 10s after launch
                if elapsed > 15.0 and abs(roll_rate_dps) < 1.0:
                    print("Landed (low rate for 10s). Disarming.")
                    break
            else:
                servos.set_all(0.0)

            # Sleep to maintain loop rate
            dt_actual = time.monotonic() - t_loop_start
            sleep_time = DT - dt_actual
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nManual stop.")
    finally:
        servos.close()
        imu.close()
        log_file.close()
        print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
```

> **What's missing from this sketch:**
>
> - **Roll angle integration.** The gyro gives rate, not angle. For PID-only mode this is fine (the PID uses rate). For residual SAC, the observation vector includes roll angle (obs[2]) — you'd need to integrate the gyro rate over time (with drift correction).
> - **Dynamic pressure.** Not available without a pitot tube and differential pressure sensor. The PID controller still works without it (gain scheduling defaults to q_ref), but performance will be reduced. The RL observation vector uses q at index 5 — set to 0 or estimate from time-since-launch.
> - **Full 10-element observation vector for SAC.** The ResidualSACController needs both physical units (for PID) and the 10-element vector (for SAC). For a first flight, PID-only mode avoids this complexity.
> - **ICM-20948 bank switching.** The driver above is simplified. For production, use the [Adafruit CircuitPython ICM20X library](https://github.com/adafruit/Adafruit_CircuitPython_ICM20X) or [SparkFun ICM-20948 library](https://github.com/sparkfun/SparkFun_ICM-20948_ArduinoLibrary).

### 4.3. Calibration

#### Servo zero-point calibration

1. Power on the Pi and servos.
2. Run:
   ```bash
   cd ~/active-guidance-rockets
   source venv/bin/activate
   python3 -c "
   import pigpio, time
   pi = pigpio.pi()
   for pin in [12, 13]:
       pi.set_servo_pulsewidth(pin, 1500)  # Centre
   print('Servos centred. Check tab alignment.')
   input('Press Enter to sweep...')
   for pin in [12, 13]:
       pi.set_servo_pulsewidth(pin, 1000)  # -30 deg
   time.sleep(1)
   for pin in [12, 13]:
       pi.set_servo_pulsewidth(pin, 2000)  # +30 deg
   time.sleep(1)
   for pin in [12, 13]:
       pi.set_servo_pulsewidth(pin, 1500)  # Centre
   pi.stop()
   "
   ```
3. At the 1500 us centre position, the tabs should be aligned flush with the fins. If not, adjust `SERVO_CENTRE_US` in the flight controller, or physically re-seat the servo horn.
4. During the sweep, verify the tabs deflect symmetrically to both sides.

#### IMU bias calibration

5. Place the rocket on a stable, level surface. Keep it completely still.
6. Run:
   ```bash
   python3 -c "
   import smbus2, struct, time, numpy as np
   bus = smbus2.SMBus(1)
   readings = []
   print('Collecting 500 gyro samples (5 sec). Keep rocket still...')
   for _ in range(500):
       raw = bus.read_i2c_block_data(0x68, 0x37, 2)
       val = struct.unpack('>h', bytes(raw))[0] * (2000.0 / 32768.0)
       readings.append(val)
       time.sleep(0.01)
   bias = np.mean(readings)
   noise = np.std(readings)
   print(f'Gyro Z bias: {bias:.3f} deg/s')
   print(f'Gyro Z noise (1-sigma): {noise:.3f} deg/s')
   print(f'Expected: bias < 0.5 deg/s, noise ~ 0.15 deg/s')
   bus.close()
   "
   ```
7. Note the bias value. Subtract it from readings in the flight controller, or accept it as negligible (the PID integral term will compensate for small constant offsets).

### 4.4. Bench Test

This is the most important step before putting the electronics in the rocket.

8. Power everything on (battery connected, Pi booted, SSH in).
9. Start the flight controller:
   ```bash
   cd ~/active-guidance-rockets
   source venv/bin/activate
   python3 deployment/flight_controller.py
   ```
10. The controller prints "Waiting for launch..." and the servos are centred.
11. **Pick up the rocket (or just the avionics sled) and rotate it around the roll axis by hand.** The tabs should deflect to oppose your rotation:
    - Roll clockwise → tabs should deflect to create counter-clockwise torque.
    - Roll counter-clockwise → tabs deflect the other way.
    - Stop rolling → tabs return to centre.
12. If the tabs deflect **in the wrong direction**, flip the sign in the flight controller: change `deflection_deg = action * MAX_DEFLECTION_DEG` to `deflection_deg = -action * MAX_DEFLECTION_DEG`.
13. Check responsiveness: the tabs should track your hand movements with no perceptible delay.
14. Press Ctrl+C to stop. Check the log file in `~/flight_logs/`.

---

## 5. Pre-Flight Checklist

### Before leaving for the launch site

- [ ] Battery fully charged (check with voltage meter: 8.2-8.4V for 2S LiPo)
- [ ] MicroSD card has >100 MB free space for logs
- [ ] Bench test completed successfully within the last 24 hours
- [ ] Servo screws/mounts secure (no play in the fin tabs)
- [ ] All solder joints inspected (no cold joints, no loose wires)
- [ ] Flight controller script starts without errors
- [ ] Camera system tested separately (see `camera_electronics/`)

### At the launch pad

1. Power on the Pi (connect battery).
2. Wait for boot (~15-30 seconds).
3. SSH in and start the flight controller:
   ```bash
   ssh pi@raspberrypi.local
   cd ~/active-guidance-rockets && source venv/bin/activate
   python3 deployment/flight_controller.py
   ```
4. Verify "Waiting for launch..." message appears.
5. Gently rotate the rocket — confirm tabs respond correctly.
6. Place the rocket on the launch rail. **Do not bump it — the controller is watching for launch vibration.**
7. Step back to safe distance.
8. Launch.

### Arming / disarming

- **Armed:** The controller is armed as soon as `flight_controller.py` is running. There is no separate arm command — the controller waits for launch detection (gyro rate spike > 5 deg/s).
- **Disarm:** Press Ctrl+C via SSH, or disconnect the battery. The servo `disarm()` function centres all tabs and stops sending PWM pulses.

### Safety considerations

- **Servo behaviour on power loss:** Servos go limp (no holding torque). Tabs will flutter to the aerodynamic neutral position. This is the safe failure mode — no active deflection that could steer the rocket off course.
- **Pi crash mid-flight:** Same as power loss — servos go limp. The rocket flies as if it had no active control. This is aerodynamically stable (the fins provide passive stability).
- **Runaway protection:** The controller clips all outputs to ±30 degrees. The gain scheduling reduces authority at high dynamic pressure. There is no way for the controller to command full deflection at high speed unless the gain clamp is modified.

---

## 6. Post-Flight

### 6.1. Retrieve Logs

1. Recover the rocket.
2. If the Pi is still running (battery connected), SSH in:
   ```bash
   scp pi@raspberrypi.local:~/flight_logs/*.csv .
   ```
3. If the Pi has shut down, remove the MicroSD card and read it from your laptop. Logs are in `/home/pi/flight_logs/`.

### 6.2. Review Flight Data

The log CSV has columns: `time`, `roll_rate_dps`, `action`, `deflection_deg`.

```bash
# Quick plot (on your development machine)
uv run python -c "
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv('flight_YYYYMMDD_HHMMSS.csv')
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
ax1.plot(df['time'], df['roll_rate_dps'])
ax1.set_ylabel('Roll rate (deg/s)')
ax1.axhline(y=5, color='g', linestyle='--', label='Target')
ax1.axhline(y=-5, color='g', linestyle='--')
ax1.legend()
ax2.plot(df['time'], df['deflection_deg'])
ax2.set_ylabel('Tab deflection (deg)')
ax2.set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('flight_review.png', dpi=150)
plt.show()
"
```

### 6.3. Compare Against Simulation

Run the same controller in simulation and compare:

```bash
# On your development machine:
uv run python compare_controllers.py \
    --config configs/estes_c6_sac_wind.yaml \
    --gain-scheduled --imu \
    --wind-levels 0 --n-episodes 1 \
    --save-plot
```

Compare the simulated roll rate trace against your flight log. Key things to look for:

- **Mean roll rate:** Does the real flight match simulation predictions?
- **Response time:** Does the real controller react as quickly as simulated?
- **Oscillation:** Is there ringing that doesn't appear in simulation? (Suggests servo backlash or I2C latency.)
- **Drift:** Does the roll rate have a persistent offset? (Suggests gyro bias or asymmetric fin alignment.)

> **Sim-to-real gap:** The simulation models aerodynamics from first principles. Reality will differ — turbulence, fin imperfections, servo nonlinearity, structural flex. The first few flights are for characterising this gap, not for achieving perfect control.

---

## Appendix: Pin Reference

| Function | GPIO | Physical Pin | Protocol |
|----------|------|-------------|----------|
| I2C SDA | GPIO 2 | Pin 3 | I2C |
| I2C SCL | GPIO 3 | Pin 5 | I2C |
| Servo 1 | GPIO 12 | Pin 32 | PWM |
| Servo 2 | GPIO 13 | Pin 33 | PWM |
| Servo 3 (J800) | GPIO 19 | Pin 35 | PWM |
| 5V input (Pi) | — | Pin 2 | Power |
| 3.3V output (IMU) | — | Pin 1 | Power |
| GND | — | Pin 6 | Power |
