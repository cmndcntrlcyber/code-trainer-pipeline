#!/bin/bash
set -e

# Start Xvfb virtual display
echo "Starting Xvfb on display :99..."
Xvfb :99 -screen 0 2560x1440x24 -ac &
sleep 2

# Verify display is running
if ! xdpyinfo -display :99 > /dev/null 2>&1; then
    echo "ERROR: Xvfb failed to start"
    exit 1
fi

echo "Xvfb running on :99"

# Run the data collection pipeline
exec python3 -m phase1_data_collection.scripts.run_collection \
    --config config/config.yaml \
    "$@"
