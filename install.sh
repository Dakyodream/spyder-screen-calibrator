#!/usr/bin/env bash
# install.sh — Install system dependencies for spyder-screen-calibrator
# Tested on Ubuntu 22.04 / Debian 12.  Adapt for other distros as needed.

set -e

echo "=== SpyderCheckr Screen Calibrator — Dependency Installer ==="
echo ""

# Detect package manager
if command -v apt-get &>/dev/null; then
    PKG="apt-get"
elif command -v apt &>/dev/null; then
    PKG="apt"
else
    echo "ERROR: apt not found. Please install dependencies manually."
    exit 1
fi

echo "[1/4] Updating package lists…"
sudo $PKG update -qq

echo "[2/4] Installing system packages…"
sudo $PKG install -y \
    python3 python3-pip python3-tk \
    gphoto2 \
    dcraw \
    libgphoto2-dev \
    libgphotopath/to/venv/bin/python2-6 \
    colord \
    xcalib \
    libjpeg-dev libtiff-dev

echo "[3/4] Installing Python dependencies…"
pip install --upgrade pip
pip install -r requirements.txt

echo "[4/4] Setting up udev rules for Canon cameras (avoids permission errors)…"
# Check if rule already exists
RULE_FILE="/etc/udev/rules.d/99-gphoto2.rules"
if [ ! -f "$RULE_FILE" ]; then
    # Get Canon USB rules from gphoto2
    GPHOTO_RULES=$(gphoto2 --list-cameras 2>/dev/null | grep -i "canon" | head -1 || echo "")
    # Use a broad rule for Canon (vendor ID 04a9)
    echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="04a9", MODE="0666", GROUP="plugdev"' | \
        sudo tee "$RULE_FILE" > /dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo "udev rule added: $RULE_FILE"
    echo "Add yourself to plugdev group if not already: sudo usermod -aG plugdev \$USER"
else
    echo "udev rule already exists: $RULE_FILE"
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "Usage:"
echo "  python run_calibration.py             # Full GUI mode"
echo "  python run_calibration.py --passes 5  # More passes"
echo "  python run_calibration.py --no-gui --image photo.jpg  # Headless"
echo ""
echo "Make sure to:"
echo "  1. Connect your Canon M50 MK2 via USB"
echo "  2. Set the camera to Manual mode"
echo "  3. Close any app that might grab the USB device (e.g. Darktable)"
