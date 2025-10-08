#!/bin/bash
# submit_job.sh
# Usage: ./submit_job.sh <cuda_index> <j2>
# Example: ./submit_job.sh 3 0.45

set -euo pipefail

# ---- args ----
cuda_index="${1:-0}"   # which GPU to use
j2_val="${2:-0.5}"     # J2 value

# ---- config ----
SERVICE="${SERVICE:-yastn}"                    # docker-compose service name
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

tot_D=8
grad_tol=1e-9
ctm_conv_tol=5e-11
policy=QR
thresh=0.1
omp_cores=8
seed=123
chi=192
in_tot_D=8
in_omp_cores=16
in_chi=192
in_ctm_conv_tol=5e-11

instate="/workspace/tn-torch_dev_tmp/examples/j1j2/abelian/seed_${seed}/Neel/j1j2_0.5_fp_c4v_gpu_cores_32_Neel_D_8_chi_192_ARP_0.1_LS_strong_wolfe_gradtol_1e-9_ctmtol_5e-11_state.json"
out_prefix="/workspace/tn-torch_dev_tmp/examples/j1j2/abelian/seed_${seed}/Neel/j1j2_${j2_val}_fp_c4v_gpu_cores_${omp_cores}_Neel_D_${tot_D}_chi_${chi}_qr_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}"

# ---- run inside container via docker compose ----
# -T: no TTY (good for scripts); -e: pass envs into container
docker compose exec -T \
  -e CUDA_VISIBLE_DEVICES="${cuda_index}" \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  "${SERVICE}" bash -lc "
    set -euo pipefail
    python -u /workspace/tn-torch_dev_tmp/examples/j1j2/abelian/optim_j1j2_c4v_u1_lc_yastn.py --grad_type='c4v_fp' \
      --j1 1.0 --j2 '${j2_val}' \
      --bond_dim '${tot_D}' --u1_charges 1 -1 0 2 -2 0 2 -2 0 2 --u1_total_charge 1 \
      --OPTARGS_tolerance_grad ${grad_tol}  --CTMARGS_ctm_conv_tol ${ctm_conv_tol} \
      --CTMARGS_ctm_max_iter 3000 --chi='${chi}' --seed '${seed}' --omp_cores '${omp_cores}' \
      --CTMARGS_projector_svd_method '${policy}' \
      --OPTARGS_no_opt_ctm_reinit \
      --CTMARGS_fwd_svds_thresh '${thresh}' \
      --CTMARGS_fwd_checkpoint_move nonreentrant \
      --CTMARGS_fpcm_init_iter 100 --OPTARGS_line_search strong_wolfe \
      --energy_checkpoint nonreentrant \
      --out_prefix='${out_prefix}' --GLOBALARGS_device cuda \
      --instate '${instate}' --instate_noise 0.2 --opt_max_iter 400 \
      > '${out_prefix}.out' 2>&1
  "
