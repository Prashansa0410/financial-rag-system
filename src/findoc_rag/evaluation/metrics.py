"""
Information retrieval and ranking evaluation metrics.
Includes Hit Rate@k, Mean Reciprocal Rank (MRR), and Rank Change Rate (RCR).
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np


def compute_hit_rate(
    results_list: List[List[Dict[str, Any]]],
    ground_truth_chunks: List[str],
    k: int = 3
) -> float:
    """
    Computes Hit Rate@k: proportion of queries where target chunk appears in top-k.
    """
    if not results_list or not ground_truth_chunks:
        return 0.0

    hits = 0
    for results, target in zip(results_list, ground_truth_chunks):
        top_k_ids = [r.get("chunk_id", "") for r in results[:k]]
        if target in top_k_ids:
            hits += 1

    return hits / len(ground_truth_chunks)


def compute_mrr(
    results_list: List[List[Dict[str, Any]]],
    ground_truth_chunks: List[str]
) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) across all queries.
    """
    if not results_list or not ground_truth_chunks:
        return 0.0

    reciprocal_ranks = []
    for results, target in zip(results_list, ground_truth_chunks):
        rr = 0.0
        for rank, r in enumerate(results, start=1):
            if r.get("chunk_id", "") == target:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return float(np.mean(reciprocal_ranks))


def compute_rank_change_rate(
    pre_rerank_results: List[List[Dict[str, Any]]],
    post_rerank_results: List[List[Dict[str, Any]]]
) -> float:
    """
    Computes Rank Change Rate (RCR): percentage of queries where Top-1 chunk changed after reranking.
    """
    if not pre_rerank_results or not post_rerank_results:
        return 0.0

    changed = 0
    total = min(len(pre_rerank_results), len(post_rerank_results))
    for pre, post in zip(pre_rerank_results[:total], post_rerank_results[:total]):
        pre_top1 = pre[0].get("chunk_id", "") if pre else ""
        post_top1 = post[0].get("chunk_id", "") if post else ""
        if pre_top1 != post_top1:
            changed += 1

    return changed / total if total > 0 else 0.0
