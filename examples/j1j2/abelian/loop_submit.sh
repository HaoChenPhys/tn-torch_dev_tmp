#!/bin/bash
# run_grid_tmux.sh
# Launch job_run.sh in separate tmux windows for each (cuda_index, j2) pair.

set -euo pipefail

JOB="./submit_job.sh"          # path to your script
SESSION="sweep_j2_$(date +%y%m%d_%H%M%S)"

# Pairs: (index, j2)
indices=(0 1 2 3 4 5 6 7)
j2s=(0.40 0.42 0.44 0.45 0.46 0.48 0.50 10.0)

# sanity checks
command -v tmux >/dev/null || { echo "tmux not found in PATH"; exit 1; }
[[ -x "$JOB" ]] || { echo "JOB not executable: $JOB"; exit 1; }

# create a detached session with the first job
tmux new-session -d -s "$SESSION" -n "gpu${indices[0]}_j2_${j2s[0]}"
tmux send-keys -t "${SESSION}:0" "$JOB ${indices[0]} ${j2s[0]}" C-m

# create windows for the rest
for ((i=1; i<${#indices[@]}; i++)); do
  win="gpu${indices[$i]}_j2_${j2s[$i]}"
  tmux new-window -t "$SESSION" -n "$win"
  tmux send-keys -t "$SESSION:$i" "$JOB ${indices[$i]} ${j2s[$i]}" C-m
done

# attach so you can watch them
tmux attach -t "$SESSION"
