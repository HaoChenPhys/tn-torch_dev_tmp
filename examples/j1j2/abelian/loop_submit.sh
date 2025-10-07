#!/usr/bin/env bash
#
# driver.sh — submit slurm_job.sh 6×, sleeping 4h10m between each

# starting cont index (matches your existing *_cont5_state.json)
cont=0

# how many iterations
n_iters=3

for ((i=1; i<=n_iters; i++)); do
  echo "[$(date +'%F\ %T')] Submitting cont${cont} → cont$((cont+1))"
  sbatch submit_job.sh "${cont}"
  cont=$((cont + 1))

  # sleep 4h10m, but skip after the last iteration
  if (( i < n_iters )); then
    sleep 4h 5m
  fi
done
