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
