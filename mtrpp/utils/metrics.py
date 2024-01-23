"""Placeholder for metrics."""
from functools import partial
import evaluate
import numpy as np
import torch
import torchmetrics.retrieval as retrieval_metrics

# RETRIEVAL METRICS
def _prepare_torchmetrics_input(scores, query2target_idx):
    target = [
        [i in target_idxs for i in range(len(scores[0]))]
        for query_idx, target_idxs in query2target_idx.items()
    ]
    indexes = torch.arange(len(scores)).unsqueeze(1).repeat((1, len(target[0])))
    return torch.as_tensor(scores), torch.as_tensor(target), indexes


def _call_torchmetrics(
    metric: retrieval_metrics, scores, query2target_idx, **kwargs
):
    preds, target, indexes = _prepare_torchmetrics_input(scores, query2target_idx)
    return metric(preds, target, indexes=indexes, **kwargs).item()


def recall(predicted_scores, query2target_idx, top_k: int) -> float:
    """Compute retrieval recall score at cutoff k.

    Args:
        predicted_scores: N x M similarity matrix
        query2target_idx: a dictionary with
            key: unique query idx
            values: list of target idx
        k: number of top-k results considered
    Returns:
        average score of recall@k
    """
    recall_metric = retrieval_metrics.RetrievalRecall(top_k=top_k)
    return _call_torchmetrics(recall_metric, predicted_scores, query2target_idx)


def mean_average_precision(predicted_scores, query2target_idx, top_k: int) -> float:
    """Compute retrieval mean average precision (MAP) score at cutoff k.

    Args:
        predicted_scores: N x M similarity matrix
        query2target_idx: a dictionary with
            key: unique query idx
            values: list of target idx
    Returns:
        MAP@k score
    """
    map_metric = retrieval_metrics.RetrievalMAP(top_k=top_k)
    return _call_torchmetrics(map_metric, predicted_scores, query2target_idx)


def mean_reciprocal_rank(predicted_scores, query2target_idx) -> float:
    """Compute retrieval mean reciprocal rank (MRR) score.

    Args:
        predicted_scores: N x M similarity matrix
        query2target_idx: a dictionary with
            key: unique query idx
            values: list of target idx
    Returns:
        MRR score
    """
    mrr_metric = retrieval_metrics.RetrievalMRR()
    return _call_torchmetrics(mrr_metric, predicted_scores, query2target_idx)

def median_rank(query_list, query2target_idx, query2target_score):
    results, query2rank = [], []
    ctr = 0
    for track_id, scores in zip(query_list, query2target_score):
        targets = query2target_idx[track_id]
        desc_indices = np.argsort(scores, axis=-1)[::-1]
        rank = [idx for idx, i in enumerate(desc_indices) if i in targets]
        results.append(min(rank))
        query2rank.append({
            "index": ctr,
            "query": track_id,
            "targets": targets,
            "min_rank": min(rank),
        })
        ctr += 1
    return np.median(results), query2rank
