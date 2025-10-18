#!/bin/bash
#SBATCH -p deadline                        # Partition
#SBATCH --qos=deadline                     # QoS
#SBATCH --output=output_bash/KGQPP_qq_%j.txt
#SBATCH --error=output_bash/err_gnn_%j.txt # Error log
#SBATCH --gres=gpu:rtx6000ada:1            # Request 1 GPU
#SBATCH --ntasks=1                         # Single task
#SBATCH --cpus-per-task=16                 # Number of CPU cores
#SBATCH --mem=60G                          # Memory allocation

echo "Starting job $SLURM_JOB_ID at $(date)"

# -------------------------------
# Environment Setup
# -------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gnn

# -------------------------------
# Model and Directory Configuration
# -------------------------------
export MODEL_NAME=distilbert
export MODEL=sentence-transformers/msmarco-distilbert-base-v4
export MAIN_DIR_MODEL=/path/to/project/dataset/KGQPP/V1_BM25_OnlyQ/${MODEL_NAME}

mkdir -p $MAIN_DIR_MODEL

# -------------------------------
# Nearest Neighbor Graph Construction (Q–Q)
# -------------------------------
python sim_search.py \
    --query_file_main dataset/v1/queries.train.small.tsv \
    --query_file_search dataset/v1/queries.train.small.tsv \
    --output_file dataset/NNQ/NNQ_${MODEL_NAME}_train_V1.tsv \
    --output_file_json dataset/NNQ/NNQ_${MODEL_NAME}_train_V1.json \
    --model_name $MODEL \
    --top_k 100

for year in dev 2019 2020 hard; do
    python sim_search.py \
        --query_file_main dataset/v1/queries.train.small.tsv \
        --query_file_search dataset/trec_data/${year}/${year}_queries.tsv \
        --output_file dataset/NNQ/NNQ_${MODEL_NAME}_${year}.tsv \
        --output_file_json dataset/NNQ/NNQ_${MODEL_NAME}_${year}.json \
        --model_name $MODEL \
        --top_k 100
done

# -------------------------------
# Dataset Construction (JSON)
# -------------------------------
python make_dataset_qq.py \
    --query_path dataset/v1/queries.train.small.tsv \
    --nnq_path dataset/NNQ/NNQ_${MODEL_NAME}_train_V1.json \
    --tsv_eval_path dataset/Bm25/eval/v1/msmarco_v1_train_eval_file.tsv_1000 \
    --graph_dataset_path ${MAIN_DIR_MODEL}/MotherDataset_train_v1_${MODEL_NAME}.json

for year in dev 2019 2020 hard; do
    python make_dataset_qq.py \
        --query_path dataset/trec_data/${year}/${year}_queries.tsv \
        --nnq_path dataset/NNQ/NNQ_${MODEL_NAME}_${year}.json \
        --tsv_eval_path dataset/Bm25/eval/v1/msmarco_v1_${year}_cut1000_eval_file.tsv_1000 \
        --graph_dataset_path ${MAIN_DIR_MODEL}/MotherDataset_${year}_${MODEL_NAME}.json
done

# -------------------------------
# Graph Construction (.pt)
# -------------------------------
export TOPK_QQ=10
export MEASUREMENT=ndcg  # or map

python dataset_builder_qq.py \
    --graph_dataset_path ${MAIN_DIR_MODEL}/MotherDataset_train_v1_${MODEL_NAME}.json \
    --main_path $MAIN_DIR_MODEL/ \
    --out_path "$MAIN_DIR_MODEL/graph_train_${TOPK_QQ}_${MEASUREMENT}.pt" \
    --model_name $MODEL \
    --eval_measurement $MEASUREMENT \
    --topk_qq $TOPK_QQ

for year in dev 2019 2020 hard; do
    python dataset_builder_qq.py \
        --graph_dataset_path ${MAIN_DIR_MODEL}/MotherDataset_${year}_${MODEL_NAME}.json \
        --main_path $MAIN_DIR_MODEL/ \
        --out_path "$MAIN_DIR_MODEL/graph_${year}_${TOPK_QQ}_${MEASUREMENT}.pt" \
        --model_name $MODEL \
        --eval_measurement $MEASUREMENT \
        --topk_qq $TOPK_QQ
done

# -------------------------------
# Model Training and Evaluation
# -------------------------------
export EPOCHS=20
export HID_DIM=256
export HEAD_TYPE=linear   # or mlp
export GNN_TYPE=GCN

export OUT_DIR="$MAIN_DIR_MODEL/cache_${GNN_TYPE}_${MEASUREMENT}_${HEAD_TYPE}_${HID_DIM}"
mkdir -p $OUT_DIR

python train_model_GCN_qq.py \
  --graph "$MAIN_DIR_MODEL/graph_train_${TOPK_QQ}_${MEASUREMENT}.pt" \
  --eval_measurement $MEASUREMENT \
  --model_name $MODEL \
  --test_root $MAIN_DIR_MODEL \
  --epochs $EPOCHS \
  --train_bs 2048 \
  --test_bs 4096 \
  --hid $HID_DIM \
  --lr 0.001 \
  --wd 0.0001 \
  --topk_qq $TOPK_QQ \
  --out_dir $OUT_DIR \
  --head $HEAD_TYPE \
  --model_tag $MODEL_NAME \
  --train

python correlation.py \
  --input $OUT_DIR \
  --collection V1 \
  --save "correlation/${MODEL_NAME}_${GNN_TYPE}_${MEASUREMENT}_${HEAD_TYPE}_hid_${HID_DIM}_qq_${TOPK_QQ}_e_${EPOCHS}"
