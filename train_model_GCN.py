#!/usr/bin/env python3
"""
Train and evaluate a query-only Graph Convolutional Network (GCN) model
for Query Performance Prediction (QPP).

The model learns from a graph of queries connected by semantic similarity edges
(`('query','similar','query')`), using query embeddings as node features and
evaluation metrics (e.g., MAP, NDCG) as labels.

Input:
  --graph              Path to the base .pt graph (built by dataset_builder_qq.py)
  --eval_measurement   Metric to predict (e.g., map, ndcg)
  --model_name         SentenceTransformer model used in graph building
  --test_root          Directory containing MotherDataset_* test sets
  --epochs, --hid, --lr, --wd, --train_bs, --test_bs, etc. for training control
  --head               Prediction head type ("linear" or "mlp")

Outputs:
  - Trained model weights: gnn_best.pt, head_best.pt
  - Predictions TSV for each test year
  - Correlation metrics via correlation.py
"""

import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch_geometric.nn import GraphConv, to_hetero
from torch_geometric.loader import NeighborLoader
from sentence_transformers import SentenceTransformer
import faiss


# ------------------------------
# Utility Functions
# ------------------------------
def load_graph(path):
    """Safely load PyTorch Geometric HeteroData."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _ensure_long_edges(edge_index_dict):
    """Ensure all edge indices are torch.long dtype."""
    for k, e in list(edge_index_dict.items()):
        if e is None:
            edge_index_dict.pop(k)
        elif not torch.is_tensor(e) or e.dtype != torch.long:
            edge_index_dict[k] = e.long()
    return edge_index_dict


def set_seed(seed: int = 42):
    """Ensure deterministic training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_queries(queries, model_name, device):
    """Encode text queries into dense embeddings."""
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 256
    embs = model.encode(
        queries,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embs.cpu()


def normalize_row(values):
    total = sum(values) + 1e-12
    return [v / total for v in values]


# ------------------------------
# Model Components
# ------------------------------
class BaseGraphConv(nn.Module):
    """Two-layer GCN with ReLU + dropout."""
    def __init__(self, hid):
        super().__init__()
        self.conv1 = GraphConv(-1, hid, aggr='add')
        self.conv2 = GraphConv(hid, hid, aggr='add')
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.drop(self.act(self.conv1(x, edge_index, edge_weight=edge_weight)))
        x = self.drop(self.act(self.conv2(x, edge_index, edge_weight=edge_weight)))
        return x


class QPPHead(nn.Module):
    """Linear regression head for QPP."""
    def __init__(self, hid):
        super().__init__()
        self.out = nn.Linear(hid, 1)

    def forward(self, x):
        return self.out(x).squeeze(-1)


class QPPHeadMLP(nn.Module):
    """Two-layer MLP regression head (optional)."""
    def __init__(self, hid, width=256, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(hid, width)
        self.norm = nn.LayerNorm(width)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width, 1)

    def forward(self, x):
        h = self.fc1(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return self.fc2(h).squeeze(-1)


# ------------------------------
# Graph Augmentation
# ------------------------------
def augment_with_test_qonly(data, test_json_path, model_name, device, test_root, topk_qq, metric):
    """Augment training graph with test queries (q-q edges only)."""
    with open(test_json_path, "r") as f:
        Test = json.load(f)
    with open(os.path.join(test_root, "query_ids.json"), "r") as f:
        train_qids = json.load(f)

    Q0 = data["query"].x.size(0)
    train_qids = train_qids[:Q0]
    test_qids = list(Test.keys())
    test_qtexts = [Test[q]["query"] for q in test_qids]
    test_emb = encode_queries(test_qtexts, model_name, device)

    # Append test nodes
    data["query"].x = torch.cat([data["query"].x, test_emb], dim=0)
    y_add = torch.full((len(test_qids),), float("nan"))
    for i, qid in enumerate(test_qids):
        ev = Test[qid].get("eval", {})
        if metric in ev:
            y_add[i] = float(ev[metric])
    data["query"].y = torch.cat([data["query"].y, y_add], dim=0)

    # q-q edges between test and train
    train_emb = data["query"].x[:Q0].cpu().numpy().astype("float32")
    test_emb_np = test_emb.cpu().numpy().astype("float32")
    index = faiss.IndexFlatIP(train_emb.shape[1])
    index.add(train_emb)
    sims, nbrs = index.search(test_emb_np, topk_qq)

    src, dst, wts = [], [], []
    for i in range(len(test_qids)):
        qi = Q0 + i
        weights = normalize_row(sims[i].tolist())
        for j, w in zip(nbrs[i].tolist(), weights):
            if w > 0:
                src.append(qi)
                dst.append(int(j))
                wts.append(float(w))

    if src:
        new_edges = torch.tensor([src, dst], dtype=torch.long)
        new_weights = torch.tensor(wts, dtype=torch.float32).unsqueeze(-1)
        old_edges = data[("query", "similar", "query")].edge_index
        old_weights = data[("query", "similar", "query")].edge_attr
        data[("query", "similar", "query")].edge_index = torch.cat([old_edges, new_edges], dim=1)
        data[("query", "similar", "query")].edge_attr = torch.cat([old_weights, new_weights], dim=0)

    mask = torch.zeros(data["query"].x.size(0), dtype=torch.bool)
    mask[Q0:Q0 + len(test_qids)] = True
    data["query"].test_mask = mask
    return Test, Q0


# ------------------------------
# Training & Prediction
# ------------------------------
def train_all_qonly(data, device, epochs, train_bs, lr, wd, hid=128, head_type="linear"):
    """Train on query-only graph."""
    y_all = data["query"].y
    train_mask = torch.isfinite(y_all)
    data["query"].train_mask = train_mask

    fanouts = {etype: [0, 0] for etype in data.edge_types}
    fanouts[("query", "similar", "query")] = [-1, -1]

    loader = NeighborLoader(
        data,
        input_nodes=("query", data["query"].train_mask),
        num_neighbors=fanouts,
        batch_size=train_bs,
        shuffle=True,
    )

    gnn = to_hetero(BaseGraphConv(hid), data.metadata(), aggr="sum").to(device)
    head = (QPPHead(hid) if head_type == "linear" else QPPHeadMLP(hid)).to(device)
    opt = torch.optim.Adam(list(gnn.parameters()) + list(head.parameters()), lr=lr, weight_decay=wd)

    def train_epoch():
        gnn.train(); head.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            y = batch["query"].y
            m = batch["query"].train_mask & torch.isfinite(y)
            if m.sum() == 0:
                continue

            edge_index_dict = _ensure_long_edges(batch.edge_index_dict)
            edge_weight_dict = {}
            etype = ("query", "similar", "query")
            if etype in edge_index_dict:
                w = getattr(batch[etype], "edge_attr", None)
                if w is not None:
                    w = w.squeeze(-1).to(batch[etype].edge_index.device, dtype=torch.float32)
                edge_weight_dict[etype] = w

            opt.zero_grad(set_to_none=True)
            x_dict = gnn(batch.x_dict, edge_index_dict, edge_weight=edge_weight_dict)
            pred = head(x_dict["query"]).squeeze(-1)
            loss = F.huber_loss(pred[m], y[m], delta=0.5)
            loss.backward()
            nn.utils.clip_grad_norm_(list(gnn.parameters()) + list(head.parameters()), 2.0)
            opt.step()
            total += loss.item() * m.sum().item()
            count += m.sum().item()
        return total / max(count, 1)

    best = {"loss": float("inf")}
    for ep in range(epochs):
        tr = train_epoch()
        print(f"[train] epoch {ep:02d} | loss={tr:.6f}")
        if tr < best["loss"]:
            best = {"loss": tr, "gnn": gnn.state_dict(), "head": head.state_dict()}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--eval_measurement", default="map")
    ap.add_argument("--test_root", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train_bs", type=int, default=2048)
    ap.add_argument("--test_bs", type=int, default=4096)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--topk_qq", type=int, default=10)
    ap.add_argument("--head", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    data_base = load_graph(args.graph)

    if args.train:
        print("[info] Training on base graph (Q-Q only)...")
        best = train_all_qonly(
            data=load_graph(args.graph),
            device=device,
            epochs=args.epochs,
            train_bs=args.train_bs,
            lr=args.lr,
            wd=args.wd,
            hid=args.hid,
            head_type=args.head,
        )
        torch.save(best["gnn"], os.path.join(args.test_root, "gnn_best.pt"))
        torch.save(best["head"], os.path.join(args.test_root, "head_best.pt"))
        print("[ok] Saved checkpoints")

    print("[info] Inference complete")


if __name__ == "__main__":
    main()
