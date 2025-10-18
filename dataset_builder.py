#!/usr/bin/env python3
"""
Builds a query-only graph dataset (.pt) for Query Performance Prediction (QPP).

Inputs:
  --graph_dataset_path   JSON produced by make_dataset_qq.py
  --main_path            Directory to save intermediate outputs (e.g., query_ids.json)
  --out_path             Output PyTorch Geometric graph file (.pt)
  --model_name           SentenceTransformer model for encoding queries
  --eval_measurement     Metric to extract (e.g., map or ndcg)
  --topk_qq              Number of nearest neighbors per query to include as edges

Outputs:
  A heterogeneous graph with:
    - Node type: "query"
    - Edge type: ('query', 'similar', 'query')
    - Node features: Sentence embeddings
    - Edge attributes: Normalized similarity weights
"""

import os
import json
import torch
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_weights(weights, eps=1e-12):
    total = sum(weights)
    if total < eps:
        return [0.0 for _ in weights]
    return [w / total for w in weights]


def encode_queries(queries, model_name, device):
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 256
    embeddings = model.encode(
        queries,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=device,
    )
    return embeddings.cpu()


def build_graph(data_dict, model_name, device, eval_measurement, topk_qq):
    qids = list(data_dict.keys())
    queries = [data_dict[q]["query"] for q in qids]
    embs = encode_queries(queries, model_name, device)

    labels = []
    for q in qids:
        ev = data_dict[q].get("eval", {})
        labels.append(float(ev.get(eval_measurement, "nan")))

    src, dst, wts = [], [], []
    for i, q in enumerate(qids):
        nn_list = data_dict[q].get("neighbors", [])[:topk_qq]
        weights = [float(x.get("score", 0.0)) for x in nn_list]
        weights = normalize_weights(weights)
        for nb, wt in zip(nn_list, weights):
            if nb["qid"] in data_dict:
                j = qids.index(nb["qid"])
                src.append(i)
                dst.append(j)
                wts.append(wt)

    src = torch.tensor(src, dtype=torch.long)
    dst = torch.tensor(dst, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.tensor(wts, dtype=torch.float32).unsqueeze(-1)
    x = embs
    y = torch.tensor(labels, dtype=torch.float32)

    graph = HeteroData()
    graph["query"].x = x
    graph["query"].y = y
    graph[("query", "similar", "query")].edge_index = edge_index
    graph[("query", "similar", "query")].edge_attr = edge_attr

    return graph, qids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dataset_path", required=True, help="Input JSON dataset path")
    ap.add_argument("--main_path", required=True, help="Output directory for metadata")
    ap.add_argument("--out_path", required=True, help="Output .pt file path")
    ap.add_argument("--model_name", required=True, help="SentenceTransformer model")
    ap.add_argument("--eval_measurement", required=True, help="Evaluation metric (e.g., map, ndcg)")
    ap.add_argument("--topk_qq", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.main_path, exist_ok=True)

    data = load_json(args.graph_dataset_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    graph, qids = build_graph(
        data_dict=data,
        model_name=args.model_name,
        device=device,
        eval_measurement=args.eval_measurement,
        topk_qq=args.topk_qq,
    )

    torch.save(graph, args.out_path)
    with open(os.path.join(args.main_path, "query_ids.json"), "w", encoding="utf-8") as f:
        json.dump(qids, f)

    print(f"[ok] Graph saved -> {args.out_path}")
    print(f"[ok] Query IDs saved -> {os.path.join(args.main_path, 'query_ids.json')}")


if __name__ == "__main__":
    main()
  
