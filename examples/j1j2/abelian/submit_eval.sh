#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=72G
#SBATCH --partition=shared

#SBATCH --output=jobout/output_%j.txt      # Sortie standard (%j se remplace par l’ID du job)
#SBATCH --error=jobout/error_%j.txt        # Error log

module load anaconda3

# nvcc --version
source activate torch_peps
echo $SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

tot_D=11
policy=ARP
grad_tol=1e-9
ctm_conv_tol=5e-11
omp_cores=$SLURM_CPUS_PER_TASK
seed=123
chi="$1"

in_D=11
in_chi=363
thresh=0.1


# instate="./seed_${seed}/j1j2_0.5_fp_c4v_gpu_cores_${omp_cores}_D_${in_D}_chi_${in_chi}_arnoldi_threshold_${thresh}_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}_state.json"
# out_prefix="./seed_${seed}/j1j2_0.5_fp_c4v_obs_D_${tot_D}_optchi_${in_chi}_chi_${chi}_arnoldi_threshold_${thresh}_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}"

instate="./seed_${seed}/j1j2_0.5_fp_c4v_gpu_cores_${omp_cores}_D_${in_D}_chi_${in_chi}_qr_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}_cont26_state.json"
out_prefix="./seed_${seed}/obs/j1j2_0.5_fp_c4v_obs_D_${tot_D}_optchi_${in_chi}_chi_${chi}_qr_LS_strong_wolfe_gradtol_${grad_tol}_ctmtol_${ctm_conv_tol}"

echo "JOBID: $SLURM_JOB_ID" > "${out_prefix}.out"
apptainer exec --nv /work/conteneurs/calmip/custom_users/cuquantum_arrayfire.sif \
    python -u /users/m25112/m25112chnh/tn-torch_dev/examples/j1j2/abelian/eval_energy.py --grad_type='c4v_fp' \
    --j1 1.0 --j2 0.5 \
    --instate="$instate"  --instate_noise 0.0 --opt_max_iter 2000 --out_prefix "$out_prefix" --energy_checkpoint nonreentrant\
    --OPTARGS_tolerance_grad $grad_tol  --CTMARGS_ctm_conv_tol $ctm_conv_tol \
    --CTMARGS_ctm_max_iter 1000 --chi=$chi --seed "$seed" --omp_cores "$omp_cores" --CTMARGS_projector_svd_method "$policy"\
    --CTMARGS_fwd_svds_thresh 0.1 \
    --OPTARGS_line_search strong_wolfe \
    --GLOBALARGS_device cuda\
    >> "${out_prefix}.out" 2>&1
