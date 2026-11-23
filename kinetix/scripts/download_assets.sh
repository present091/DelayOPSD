#!/usr/bin/env bash
# The Real-Time Chunking release's Kinetix assets (gs://rtc-assets over HTTPS) into assets/: the BC policies of
# epochs 24 and 31 (12 MB per level) and the 12 expert datasets (2.8 GB each, 33.5 GB in all). Resumes partial files.
set -uo pipefail
cd "$(dirname "$0")/../assets"
L="car_launch cartpole_thrust catapult catcher_v3 chain_lander grasp_easy h17_unicycle hard_lunar_lander mjc_half_cheetah mjc_swimmer mjc_walker trampoline"
mkdir -p bc/24/policies bc/31/policies expert/data
get() { curl -sfL -C - -o "$1" "https://storage.googleapis.com/rtc-assets/$1" || echo "failed $1"; }
for e in 24 31; do for l in $L; do get bc/$e/policies/worlds_l_$l.pkl; done; done
for l in $L; do get expert/data/worlds_l_$l.npz; echo "expert $l done"; done
