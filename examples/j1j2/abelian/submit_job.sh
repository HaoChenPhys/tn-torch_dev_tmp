#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

tot_D=6
grad_tol=1e-9
ctm_conv_tol=5e-11
policy=GESDD
thresh=0.1
omp_cores=8
seed=123
chi=192
in_tot_D=8
in_omp_cores=16
in_chi=192
in_ctm_conv_tol=5e-11

# instate="./seed_${seed}/j1j2_0.5_fp_c4v_gpu_cores_${in_omp_cores}_D_${in_tot_D}_chi_${in_chi}_qr_LS_backtracking_gradtol_${grad_tol}_ctmtol_${in_ctm_conv_tol}_cont5_state.json"
# instate="./seed_${seed}/j1j2_0.5_fp_c4v_gpu_cores_${omp_cores}_D_${tot_D}_chi_${chi}_qr_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}_cont${cont}_state.json"
# instate="./seed_${seed}/cRVB/j1j2_0.5_fp_c4v_gpu_cores_${omp_cores}_cRVB_D_${in_tot_D}_chi_${in_chi}_ARP_${thresh}_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}_cont3_state.json"
# instate="../../../test-input/abelian/c4v/BFGS100LS_U1B_D7-chi147-j20.0-run0-c1U1guess0AD7j20chi147n0_state.json"
# instate="j1j2_ipeps_states/single-site_pg-C4v-A1_internal-U1/j20.5/state_1s_A1_U1B_j20.5_D8_chi_opt192.json"
# instate="./seed_${seed}/Neel/j1j2_0.5_fp_c4v_gpu_cores_${omp_cores}_Neel_D_${in_tot_D}_chi_${in_chi}_ARP_${thresh}_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}_state.json"
instate="None"

# out_prefix="./seed_${seed}/j1j2_0.5_fp_c4v_gpu_cores_${omp_cores}_D_${tot_D}_chi_${chi}_qr_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}_cont${next_cont}"
# out_prefix="./seed_${seed}/Neel/runtime_benchmark/j1j2_0.5_fp_c4v_gpu_cores_${omp_cores}_Neel_D_${tot_D}_chi_${chi}_ARP_${thresh}_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}"
# out_prefix="./seed_${seed}/Neel/runtime_benchmark/j1j2_0.5_fp_c4v_gpu_cores_${omp_cores}_Neel_D_${tot_D}_chi_${chi}_ARP_${thresh}_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}"
# out_prefix="./seed_${seed}/Neel/runtime_benchmark/j1j2_0.5_c4v_cpu1_cores_${omp_cores}_Neel_D_${tot_D}_chi_${chi}_fullrank_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}"
# out_prefix="./seed_${seed}/Neel/runtime_benchmark/j1j2_0.5_fp_c4v_gpu_cores_${omp_cores}_Neel_D_${tot_D}_chi_${chi}_qr_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}"
# out_prefix="./seed_${seed}/Neel/j1j2_0.5_fp_c4v_gpu_cores_${omp_cores}_Neel_D_${tot_D}_chi_${chi}_qr_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}"
# out_prefix="./seed_${seed}/cRVB/j1j2_0.5_fp_c4v_gpu_cores_${omp_cores}_cRVB_D_${in_tot_D}_chi_${in_chi}_ARP_${thresh}_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}_cont4"
# out_prefix="./seed_${seed}/cRVB/j1j2_0.5_fp_c4v_gpu_cores_${omp_cores}_cRVB_D_${tot_D}_chi_${chi}_qr_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}_cont0"
# out_prefix="./seed_${seed}/j1j2_fp_c4v_cores_${omp_cores}_D_${tot_D}_chi_${chi}_qr_LS_strong_wolfe"ls
out_prefix="tmp"

python -u optim_j1j2_c4v_u1_lc_yastn.py --grad_type='c4v' \
    --j1 1.0 --j2 0.5 \
    --bond_dim "$tot_D" --u1_charges 1 -1 0 2 -2 0 2 -2 --u1_total_charge 1 \
    --OPTARGS_tolerance_grad $grad_tol  --CTMARGS_ctm_conv_tol $ctm_conv_tol \
    --CTMARGS_ctm_max_iter 5000 --chi=$chi --seed "$seed" --omp_cores "$omp_cores" --CTMARGS_projector_svd_method "$policy"\
    --OPTARGS_no_opt_ctm_reinit \
    --CTMARGS_fwd_svds_thresh "$thresh" \
    --CTMARGS_fwd_checkpoint_move nonreentrant \
    --CTMARGS_fpcm_init_iter 0 --OPTARGS_line_search strong_wolfe \
    --energy_checkpoint nonreentrant \
    --out_prefix="$out_prefix" --GLOBALARGS_device cpu \
    --instate_noise 0.01  --opt_max_iter 10 \
    >> "${out_prefix}.out" 2>&1
# --instate="$instate"

# --energy_checkpoint nonreentrant\

# D=10: {-2:3, 0:4, 2:3} tot=1
# D=11: {-2:3, 0:4, 2:4} tot=1
# --bond_dim "$tot_D" --u1_charges 1 -1 0 2 -2 0 2 -2 0 2 -2 0 2  --u1_total_charge 1 \

# cRVB:
# D=7: {-1:3, 0:1, 1:3} tot=0
# D=7: {-1:4, 0:1, 1:4} tot=0