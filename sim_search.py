#!/usr/bin/env python3
"""
Compute nearest-neighbor queries using a sentence embedding model (FAISS, inner-product).

Inputs:
  --query_file_main   TSV with "qid\\tquery" used as the base set to index (search targets)
  --query_file_search TSV with "qid\\tquery" used as the query set to search with
  --model_name        Sentence-Transformers model (default: msmarco-distilbert-base-v4)
  --top_k             Number of nearest neighbors to retrieve
Outputs:
  --output_file       TSV: qid \\t neighbor_qid \\t score (one row per neighbor)
  --output_file_json  JSON: { qid: [{"qid": neighbor_qid, "score": float}, ...], ... }
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def read_tsv_qid_text(path: str) -> Tuple[List[str], List[str]]:
    qids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            qid, text = parts
            qids.append(qid)
            texts.append(text)
    return qids, texts


def encode_texts(texts: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    model.max_seq_length = 256
    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return emb.astype("float32")


def search_nn(
    base_emb: np.ndarray,
    query_emb: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    index = faiss.IndexFlatIP(base_emb.shape[1])
    index.add(base_emb)  # (N, d)
    sims, nbrs = index.search(query_emb, top_k)  # (Q, k)
    return sims, nbrs


def save_outputs(
    out_tsv: str,
    out_json: str,
    query_ids: List[str],
    base_ids: List[str],
    sims: np.ndarray,
    nbrs: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(out_tsv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    with open(out_tsv, "w", encoding="utf-8") as ftsv:
        for qi, qid in enumerate(query_ids):
            for rank in range(nbrs.shape[1]):
                nb_idx = int(nbrs[qi, rank])
                score = float(sims[qi, rank])
                ftsv.write(f"{qid}\t{base_ids[nb_idx]}\t{score:.6f}\n")

    packed: Dict[str, List[Dict[str, float]]] = {}
    for qi, qid in enumerate(query_ids):
        items = []
        for rank in range(nbrs.shape[1]):
            nb_idx = int(nbrs[qi, rank])
            score = float(sims[qi, rank])
            items.append({"qid": base_ids[nb_idx], "score": score})
        packed[qid] = items

    with open(out_json, "w", encoding="utf-8") as fj:
        json.dump(packed, fj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_file_main", required=True, help="TSV: qid\\tquery (indexed/targets)")
    ap.add_argument("--query_file_search", required=True, help="TSV: qid\\tquery (queries)")
    ap.add_argument("--output_file", required=True, help="Output TSV path")
    ap.add_argument("--output_file_json", required=True, help="Output JSON path")
    ap.add_argument("--model_name", default="sentence-transformers/msmarco-distilbert-base-v4")
    ap.add_argument("--top_k", type=int, default=100)
    args = ap.parse_args()

    base_ids, base_texts = read_tsv_qid_text(args.query_file_main)
    query_ids, query_texts = read_tsv_qid_text(args.query_file_search)

    base_emb = encode_texts(base_texts, args.model_name)
    query_emb = encode_texts(query_texts, args.model_name)

    sims, nbrs = search_nn(base_emb, query_emb, args.top_k)
    save_outputs(args.output_file, args.output_file_json, query_ids, base_ids, sims, nbrs)

    print(f"[ok] NN written: {args.output_file} and {args.output_file_json}")


if __name__ == "__main__":
    main()
