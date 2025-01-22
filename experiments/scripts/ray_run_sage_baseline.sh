#!/bin/bash

conda_env="relational-py"

source "$(conda info --base)""/etc/profile.d/conda.sh"
conda activate "$conda_env"

declare -a dataset_pairs=(
    'ctu-accidents accidents-original'
    'ctu-accidents accidents-temporal'
    'ctu-adventureworks adventureworks-original'
    'ctu-adventureworks adventureworks-temporal'
    'ctu-airline airline-original'
    'ctu-airline airline-temporal'
    'ctu-atherosclerosis atherosclerosis-original'
    'ctu-basketballmen basketballmen-original'
    'ctu-basketballwomen basketballwomen-original'
    'ctu-biodegradability biodegradability-original'
    'ctu-bupa bupa-original'
    'ctu-carcinogenesis carcinogenesis-original'
    'ctu-cde cde-original'
    'ctu-chess chess-original'
    'ctu-classicmodels classicmodels-original'
    'ctu-classicmodels classicmodels-temporal'
    'ctu-cora cora-original'
    'ctu-countries countries-original'
    'ctu-craftbeer craftbeer-original'
    'ctu-credit credit-original'
    'ctu-dallas dallas-original'
    'ctu-dallas dallas-temporal'
    'ctu-dcg dcg-original'
    'ctu-diabetes diabetes-original'
    'ctu-dunur dunur-original'
    'ctu-elti elti-original'
    'ctu-employee employee-original'
    'ctu-employee employee-temporal'
    'ctu-ergastf1 ergastf1-original'
    'ctu-expenditures expenditures-original'
    'ctu-financial financial-original'
    'ctu-financial financial-temporal'
    'ctu-fnhk fnhk-original'
    'ctu-fnhk fnhk-temporal'
    'ctu-ftp ftp-original'
    'ctu-ftp ftp-temporal'
    'ctu-geneea geneea-original'
    'ctu-geneea geneea-temporal'
    'ctu-genes genes-original'
    'ctu-gosales gosales-original'
    'ctu-gosales gosales-temporal'
    'ctu-grants grants-original'
    'ctu-grants grants-temporal'
    'ctu-hepatitis hepatitis-original'
    'ctu-hockey hockey-original'
    'ctu-imdb imdb-original'
    'ctu-lahman lahman-original'
    'ctu-lahman lahman-temporal'
    'ctu-legalacts legalacts-original'
    'ctu-legalacts legalacts-temporal'
    'ctu-mesh mesh-original'
    'ctu-mondial mondial-original'
    'ctu-mooney mooney-original'
    'ctu-movielens movielens-original'
    'ctu-musklarge musklarge-original'
    'ctu-musksmall musksmall-original'
    'ctu-mutagenesis mutagenesis-original'
    'ctu-ncaa ncaa-original'
    'ctu-northwind northwind-original'
    'ctu-northwind northwind-temporal'
    'ctu-pima pima-original'
    'ctu-premiereleague premiereleague-original'
    'ctu-premiereleague premiereleague-temporal'
    'ctu-restbase restbase-original'
    'ctu-sakila sakila-original'
    'ctu-sakila sakila-temporal'
    'ctu-sales sales-original'
    'ctu-samegen samegen-original'
    'ctu-sap sap-original'
    'ctu-sap sap-sales'
    'ctu-sap sap-sales-temporal'
    'ctu-satellite satellite-original'
    'ctu-seznam seznam-original'
    'ctu-seznam seznam-temporal'
    'ctu-sfscores sfscores-original'
    'ctu-sfscores sfscores-temporal'
    'ctu-shakespeare shakespeare-original'
    'ctu-stats stats-original'
    'ctu-stats stats-temporal'
    'ctu-studentloan studentloan-original'
    'ctu-thrombosis thrombosis-original'
    'ctu-toxicology toxicology-original'
    'ctu-tpcc tpcc-original'
    'ctu-tpcd tpcd-original'
    'ctu-tpcds tpcds-original'
    'ctu-tpcds tpcds-temporal'
    'ctu-tpch tpch-original'
    'ctu-triazine triazine-original'
    'ctu-uwcse uwcse-original'
    'ctu-visualgenome visualgenome-original'
    'ctu-voc voc-original'
    'ctu-voc voc-temporal'
    'ctu-walmart walmart-original'
    'ctu-walmart walmart-temporal'
    'ctu-webkp webkp-original'
    'ctu-world world-original'
)


run_timeout=$(expr 60 \* 60 \* 24) # 24 hours

EXPERIMENT_ID="sage_baseline_$(date '+%d-%m-%Y_%H:%M:%S')"

NUM_GPUS=2
NUM_CPUS=40

NUM_SAMPLES=5

MLFLOW_TRACKING_URI="http://potato.felk.cvut.cz:2222"

# Create log directory
experiment_dir=logs/${EXPERIMENT_ID}
mkdir -p $experiment_dir

# ******************************************
# Init Ray cluster
ip=$(hostname --ip-address)

port_head=7890
port_dashboard=7891

ray_address=$ip:$port_head
ray_dashboard_address=$ip:$port_dashboard

ray start --head --block --node-ip-address=$ip --port=$port_head \
 --dashboard-host=$ip --dashboard-port=$port_dashboard \
 --num-cpus=${NUM_CPUS} --num-gpus=${NUM_GPUS} --memory=250000000000 --object-store-memory=8000000000 \
 --log-style=record &> "${experiment_dir}/ray_head.log" &
ray_head=$!

echo "ray head address is ${ray_address}"
sleep 10
echo "ray dashboard available at http://${ray_dashboard_address}"

ray status --address=${ray_address}
# ******************************************

# Run experiment on different datasets
dataset_runs=()

for pair in "${dataset_pairs[@]}"; do
    read -a strarr <<< "$pair"  # uses default whitespace IFS
    dataset=${strarr[0]}
    task=${strarr[1]}

    log_dir=${experiment_dir}/${dataset}_${task}
    mkdir -p $log_dir

    python -u experiments/sage_baseline.py --ray_address=${ray_address} \
    --ray_storage=${log_dir} --run_name=${EXPERIMENT_ID}_${dataset}_${task} --dataset=${dataset} \
    --mlflow_uri=${MLFLOW_TRACKING_URI} --aim_repo=logs/.aim \
    --task=${task} --num_samples=${NUM_SAMPLES} &> "${log_dir}/run.log" &
    dataset_runs+=($!)
    sleep 5
done


# Stop after given timeout
function timeout_monitor() {
    echo "Run with stop in ${run_timeout}s"
    sleep "$run_timeout"
    ray stop
}

timeout_monitor &
timeout_monitor_pid=$!

# Wait for all experiments to finish
for run_pid in ${dataset_runs[@]}; do
    wait $run_pid
done

echo "All runs finished!"

# Stop ray cluster if was not stopped by timeout
if ps -p $timeout_monitor_pid > /dev/null
then
    kill $timeout_monitor_pid
    echo "Ray will stop in 30s"
    sleep 30
    ray stop
fi

kill $ray_head &

wait $ray_head

