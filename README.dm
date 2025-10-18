# Query-Only Graph-Based Query Performance Prediction (QPP)

This repository provides a PyTorch Geometric (PyG)-based implementation of a **Graph Neural Network (GNN)** framework for **Query Performance Prediction (QPP)**.  
The model operates **only on query–query (Q–Q)** relationships, using similarity graphs constructed from dense embeddings of queries.  
It predicts query difficulty or effectiveness metrics such as **MAP**, **NDCG**, or **MRR**.

---

## 🔍 Overview

Unlike standard heterogeneous graph models that consider both query–document (Q–D) and document–document (D–D) relations,  
this version focuses exclusively on **query–query connectivity** for lightweight reasoning and interpretability.

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

