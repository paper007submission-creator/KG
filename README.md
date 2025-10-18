# Graph-Based Query Performance Prediction (QPP)

This repository provides a PyTorch Geometric (PyG)-based implementation of a **Graph Neural Network (GNN)** framework for **Query Performance Prediction (QPP)**.  
using similarity graphs constructed from dense embeddings of queries.  It predicts query difficulty or effectiveness metrics such as **MAP**.

---

The code uses:
- **PyTorch Geometric** for message passing
- **Sentence-Transformers** for dense query embeddings
- **FAISS** for nearest neighbor search between queries
- **Huber loss** for robust regression training

---

## 🧩 Graph Structure

Each node represents a **query**, and edges encode **semantic similarity** between queries based on their dense embeddings.

- **Nodes:** Queries  
- **Edges:** Query–Query similarities (`('query', 'similar', 'query')`)  
- **Edge weights:** Cosine similarity scores normalized by row-wise softmax

The model passes messages between similar queries to predict per-query performance.

---

## ⚙️ Model Components

### 1. **BaseGraphConv**
A simple 2-layer GCN (GraphConv) architecture operating only on Q–Q edges:

```python
class BaseGraphConv(nn.Module):
    def __init__(self, hid: int):
        super().__init__()
        self.conv1 = GraphConv(-1, hid, aggr='add')
        self.conv2 = GraphConv(hid, hid, aggr='add')
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.2)



# 🧠 Graph-Based Query Performance Prediction (QPP) — Query-Only Pipeline

This repository provides the full experimental pipeline for **graph-based Query Performance Prediction (QPP)** using **query–query (Q–Q)** relations under the **BM25 baseline**.  
The pipeline is designed to run efficiently on an **HPC cluster** via the provided **SLURM batch script** (`KGQPP_qq.sh`).

---

## 🚀 Overview

This system builds and trains a **Graph Neural Network (GNN)** to estimate query effectiveness (e.g., MAP or NDCG) using **semantic relationships between queries**.  
Each query node connects to its top-K similar neighbors based on dense embeddings.

### Pipeline Steps
1. Encode queries using Sentence-Transformers  
2. Build Q–Q nearest neighbor graphs using FAISS  
3. Generate JSON datasets for training and evaluation  
4. Convert them into PyTorch Geometric `.pt` graphs  
5. Train a GCN (or GAT) for performance prediction  
6. Evaluate using Pearson, Spearman, and Kendall correlations  

---

## 📁 Project Structure
