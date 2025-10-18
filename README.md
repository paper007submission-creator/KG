# 🧠 Graph-Based Query Performance Prediction (QPP)

This project implements a **Graph Neural Network (GNN)** framework for **Query Performance Prediction (QPP)** using **query–query (Q–Q)** similarity graphs under the **BM25** baseline.  
It enables building Q–Q datasets, constructing graphs, training GCN model, and evaluating performance using correlation metrics.

---

## ⚙️ Overview

- Constructs **query–query graphs** from nearest-neighbor relationships  
- Uses **Sentence-Transformers** for embeddings  
- Trains **GCN** to predict query effectiveness (MAP/NDCG)  
- Supports both **linear** and **MLP** prediction heads  
- Fully automated through **SLURM job scripts**

---

## 🧩 Setup

```bash
conda create -n gnn python=3.10 -y
conda activate gnn
pip install torch torch-geometric faiss-cpu sentence-transformers tqdm
```

(Optional for faster sampling):
```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster
```

---

## 🚀 Run the Full Pipeline (Recommended)

Submit the main batch script:

```bash
sbatch KGQPP_qq.sh
```

### Key Parameters
```bash
MODEL=sentence-transformers/msmarco-distilbert-base-v4
measurement=ndcg        # or map
topk_qq=10
hid_d=256
head=linear             # or mlp
gnn=GCN                 # or GAT
```

This script handles:
1. Nearest-neighbor graph construction  
2. Dataset and graph building  
3. Model training  
4. Correlation evaluation  

---

## 🧠 Manual Example

### 1️⃣ Build Query–Query Neighbors
For both training and test query set these similarities should be included. For test set, change the query_file_search to the target test set.

```bash
python sim_search.py   --query_file_main dataset/v1/queries.train.small.tsv   --query_file_search dataset/v1/queries.train.small.tsv   --output_file_json dataset/NNQ/NNQ_distilbert_train_V1.json   --model_name sentence-transformers/msmarco-distilbert-base-v4   --top_k 100
```

### 2️⃣ Build Graph
Here we make the graph torch dataset. before using this, you should prepare the nearest queries from previous part also pass the run file in trec format also the evaluation file that has the MAP or Ndcg of the queries . for test and training set it should be doone. 

```bash
python make_dataset.py   --query_path dataset/v1/queries.train.small.tsv   --nnq_path dataset/NNQ/NNQ_distilbert_train_V1.json   --tsv_eval_path dataset/Bm25/eval/v1/msmarco_v1_train_eval_file.tsv_1000   --graph_dataset_path distilbert/MotherDataset_train_v1_distilbert.json
```

```bash
python dataset_builder.py   --graph_dataset_path dataset/KGQPP/.../MotherDataset_train_v1_distilbert.json   --out_path dataset/KGQPP/.../graph_train_10_ndcg.pt   --eval_measurement ndcg   --topk_qq 10
```

### 3️⃣ Train the Model
```bash
python train_model_GCN.py   --graph dataset/KGQPP/.../graph_train_10_ndcg.pt   --hid 256 --head linear --epochs 20 --train
```

### 4️⃣ Evaluate Correlations
```bash
python correlation.py   --input dataset/KGQPP/.../cache_GCN_ndcg_linear_256   --collection V1
```

---

## 📊 Outputs

| File | Description |
|------|--------------|
| `gnn_best.pt`, `head_best.pt` | Trained model weights |
| `preds_trained_{year}.tsv` | Predictions for each test set |
| `correlation/*.txt` | Pearson/Spearman/Kendall correlations |
| `output_bash/*.txt` | SLURM log files |

---


