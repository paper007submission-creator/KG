#!/usr/bin/env python3
"""
Build a query-only graph dataset JSON from:
  1) A TSV of queries:              --query_path    (qid \\t query)
  2) A JSON of nearest neighbors:   --nnq_path      ({qid: [{"qid": nb_qid, "score": s}, ...]})
  3) An eval TSV with per-query metric: --tsv_eval_path (qid \\t score)

Output:
  --graph_dataset_path  -> JSON with format:
    {
      "qid": {
        "query": "<text>",
        "neighbors": [{"qid": "...", "score": <float>}, ...],   # from NNQ
        "eval": {"map": x} or {"ndcg": x} etc. (metric key preserved from file header or arg)
      },
      ...
    }
"""

import os
import json
import argparse
from typing import Dict, List, Tuple


def read_queries_tsv(path: str) -> Dict[str, str]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            qid, text = parts
            out[qid] = text
    return out


def read_nnq_json(path: str) -> Dict[str, List[Dict[str, float]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expected structure: { qid: [{"qid": "...", "score": float}, ...], ... }
    return data


def read_eval_tsv(path: str) -> Tuple[str, Dict[str, float]]:
    """
    Reads 'qid<tab>score' (any metric). Returns (metric_name, {qid: score}).
    If header present like 'qid<TAB>map', uses that metric name; otherwise 'metric'.
    """
    scores: Dict[str, float] = {}
    metric_name = "metric"
    with open(path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if first and len(parts) >= 2 and parts[0].lower() == "qid":
                metric_name = parts[1].strip().lower() or "metric"
                first = False
                continue
            first = False
            if len(parts) < 2:
                continue
            qid, val = parts[0], parts[1]
            try:
                scores[qid] = float(val)
            except ValueError:
                continue
    return metric_name, scores


def build_dataset(
    queries: Dict[str, str],
    nnq: Dict[str, List[Dict[str, float]]],
    eval_name: str,
    eval_scores: Dict[str, float],
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for qid, text in queries.items():
        item = {
            "query": text,
            "neighbors": nnq.get(qid, []),
            "eval": {},
        }
        if qid in eval_scores:
            item["eval"][eval_name] = eval_scores[qid]
        out[qid] = item
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_path", required=True, help="TSV: qid\\tquery")
    ap.add_argument("--nnq_path", required=True, help="JSON with {qid: [{qid, score}, ...]}")
    ap.add_argument("--tsv_eval_path", required=True, help="TSV: qid\\tscore (header optional)")
    ap.add_argument("--graph_dataset_path", required=True, help="Output JSON path")
    args = ap.parse_args()

    queries = read_queries_tsv(args.query_path)
    nnq = read_nnq_json(args.nnq_path)
    metric, eval_scores = read_eval_tsv(args.tsv_eval_path)

    dataset = build_dataset(queries, nnq, metric, eval_scores)

    os.makedirs(os.path.dirname(args.graph_dataset_path) or ".", exist_ok=True)
    with open(args.graph_dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)

    print(f"[ok] Dataset written: {args.graph_dataset_path} (queries={len(queries)})")


if __name__ == "__main__":
    main()
