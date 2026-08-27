#!/usr/bin/env bash
# Full level-B replay of MARS-LVIG AMtown03 through FAST-LIVO2, then offline
# true-ortho (orthoproto) and the T19 metrics run. Reproduces the pipeline that
# produced the geoloc_capture_01 evaluation.
#
# Usage (inside the geoloc_fastlivo container):
#   ./tools/metrics/t19/replay_amtown03.sh
#
# Stages:
#   1. FAST-LIVO2 on the raw AMtown03 bag  ->  AMtown03_processed/  (odometry +
#      registered clouds + rgb + RTK passthrough)
#   2. orthoproto run                       ->  data/outputs/AMtown03_ortho/
#   3. run_t19 + metrics                    ->  data/outputs/AMtown03_t19/
#
# Notes:
#   - AMtown03 uses start-offset 106 s (the "start hover" table in
#     MARS_LVIG_AMtown.yaml); the first ~106 s of hover is skipped.
#   - The bag's /left_camera/image is compressed -> use_republish:=True.
#   - Replay rate 2.0 is a safe ceiling; lower it if FAST-LIVO2 drops frames.

set -euo pipefail

FAST_WS=/opt/fast_ws
REPO=/home/sverk/geoloc
BAG="$REPO/data/captures/AMtown03"
OUT="$REPO/data/outputs/AMtown03_processed"

source "$FAST_WS/install/setup.bash"

echo "==> stage 1: FAST-LIVO2 replay of $BAG"
rm -rf "$OUT"
mkdir -p "$OUT"

# Launch FAST-LIVO2 (sim-time) + republish compressed->raw + record output.
# The mapper publishes odometry/clouds stamped from measurement time; everything
# must run on the bag clock.
ros2 launch fast_livo mapping_bag.launch.py \
    bag_path:="$BAG" \
    avia_params_file:="$FAST_WS/src/fast_livo/config/MARS_LVIG_AMtown.yaml" \
    camera_params_file:="$FAST_WS/src/fast_livo/config/camera_MARS_LVIG_AM.yaml" \
    use_republish:=True \
    rate:=2.0 \
    start_offset:=106.0 \
    use_rviz:=False &
LAUNCH_PID=$!

# Let the mapper subscribe before playback starts (mapping_bag plays paused -p;
# unpause by pressing space in its terminal, or play here explicitly).
sleep 8

ros2 bag play --clock "$BAG" --rate 2.0 --start-offset 106.0 &

ros2 bag record -o "$OUT" \
    /aft_mapped_to_init /cloud_registered /rgb_img /path \
    /dji_osdk_ros/rtk_position /dji_osdk_ros/attitude /clock &
RECORD_PID=$!

wait "$RECORD_PID"
kill "$LAUNCH_PID" 2>/dev/null || true

echo "==> stage 2: orthoproto on the processed bag"
cd "$REPO"
PYTHONPATH=tools/orthoproto python3 -u -m orthoproto run \
    --config configs/orthoproto/AMtown03.yaml

echo "==> stage 3: T19 evaluation + metrics"
python3 -u tools/metrics/t19/run_t19.py \
    --config configs/metrics/t19_eval_amtown03.yaml

echo "==> done: report at $REPO/data/outputs/AMtown03_t19/report.html"
