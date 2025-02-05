#!/bin/bash
#SBATCH --job-name=resnet_sage_hyperparams  
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-cpu=40G
#SBATCH --time=24:00:00
#SBATCH --array=0-43

declare -a dataset_pairs=(
    
    'rel-stack user-engagement'
    'rel-stack post-votes'
    'rel-stack user-badge'
    'rel-stack user-post-comment'
    'rel-stack post-post-related'
    'rel-trial study-outcome'
    'rel-trial study-adverse'
    'rel-trial site-success'
    'rel-trial condition-sponsor-run'
    'rel-trial site-sponsor-run'
    'rel-amazon user-churn'
    'rel-amazon user-ltv'
    'rel-amazon item-churn'
    'rel-amazon item-ltv'
    'rel-amazon user-item-purchase'
    'rel-amazon user-item-rate'
    'rel-amazon user-item-review'
    'rel-avito ad-ctr'
    'rel-avito user-visits'
    'rel-avito user-clicks'

    'rel-avito user-ad-visit'
    'ctu-accidents accidents-original'
    'ctu-accidents accidents-temporal'
    'ctu-employee employee-original'
    'ctu-employee employee-temporal'
    'ctu-expenditures expenditures-original'
    'ctu-grants grants-original'
    'ctu-grants grants-temporal'
    'ctu-imdb imdb-original'
    'ctu-legalacts legalacts-original'
    'ctu-legalacts legalacts-temporal'
    'ctu-sales sales-original'
    'ctu-sap sap-original'
    'ctu-sap sap-sales'
    'ctu-sap sap-sales-temporal'
    'ctu-seznam seznam-original'
    'ctu-seznam seznam-temporal'
    'ctu-tpcd tpcd-original'
    'ctu-tpcds tpcds-original'
    'ctu-tpcds tpcds-temporal'
    'ctu-tpch tpch-original'
    'ctu-visualgenome visualgenome-original'
    'ctu-walmart walmart-original'
    'ctu-walmart walmart-temporal'
)

conda_env="relational-py"

source "$(conda info --base)""/etc/profile.d/conda.sh"
conda activate "$conda_env"

EXPERIMENT_NAME="resnet_sage_hyperparams"

echo $SLURM_JOB_ID

# START_TIME=$(sacct -j ${SLURM_JOB_ID} --format=Start -n | head -n 1)

EXPERIMENT_ID="${EXPERIMENT_NAME}_${SLURM_ARRAY_JOB_ID}"

NUM_SAMPLES=3

MLFLOW_TRACKING_URI="http://potato.felk.cvut.cz:2222"

# Create log directory
experiment_dir=logs/${EXPERIMENT_ID}
mkdir -p $experiment_dir

ray_address="local"
# ******************************************

# Run experiment on different datasets

pair=${dataset_pairs[$SLURM_ARRAY_TASK_ID]}
read -a strarr <<< "$pair"  # uses default whitespace IFS
dataset=${strarr[0]}
task=${strarr[1]}

log_dir=${experiment_dir}/${dataset}_${task}
mkdir -p $log_dir

python -u experiments/dbgnn_hyperparams.py --ray_address=${ray_address} \
  --ray_storage=${log_dir} --run_name=${EXPERIMENT_ID}_${dataset}_${task} --dataset=${dataset} \
  --mlflow_uri=${MLFLOW_TRACKING_URI} --mlflow_experiment=pelesjak_${EXPERIMENT_NAME} \
  --task=${task} --num_samples=${NUM_SAMPLES} --num_gpus=1 --num_cpus=${SLURM_CPUS_PER_GPU} \
  --model="sage" --row_encoder="resnet" &> "${log_dir}/run.log"



