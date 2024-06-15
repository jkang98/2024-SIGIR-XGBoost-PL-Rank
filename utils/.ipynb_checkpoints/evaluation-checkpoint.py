# Use code from Harrie Oosterhuis's repository
# URL: https://github.com/HarrieO/2022-SIGIR-plackett-luce
# License: MIT License

import numpy as np
import torch
import utils.ranking as rnk


def ideal_metrics(data_split, rank_weights, labels):
    cutoff = rank_weights.size
    result = np.zeros(data_split.num_queries())
    for qid in range(data_split.num_queries()):
        q_labels = data_split.query_values_from_vector(qid, labels)
        ranking = rnk.cutoff_ranking(-q_labels, cutoff)
        result[qid] = np.sum(rank_weights[: ranking.size] * q_labels[ranking])
    return result


def compute_results(data_split, model, rank_weights, labels, ideal_metrics):
    scores = model(torch.from_numpy(data_split.feature_matrix))[:, 0].detach().numpy()

    return compute_results_from_scores(
        data_split, scores, rank_weights, labels, ideal_metrics
    )


def evaluate_max_likelihood(data_split, scores, rank_weights, labels, ideal_metrics):
    cutoff = rank_weights.size
    result = np.zeros(data_split.num_queries())
    query_normalized_result = np.zeros(data_split.num_queries())
    for qid in range(data_split.num_queries()):
        q_scores = data_split.query_values_from_vector(qid, scores)
        q_labels = data_split.query_values_from_vector(qid, labels)
        ranking = rnk.cutoff_ranking(-q_scores, cutoff)
        q_result = np.sum(rank_weights[: ranking.size] * q_labels[ranking])
        if ideal_metrics[qid] == 0:
            query_normalized_result[qid] = 0.0
        else:
            query_normalized_result[qid] = q_result / ideal_metrics[qid]
        result[qid] = q_result / np.mean(ideal_metrics)
    return float(np.mean(query_normalized_result)), float(np.mean(result))


def compute_results_from_scores(
    data_split, scores, rank_weights, labels, ideal_metrics
):
    QN_ML, N_ML = evaluate_max_likelihood(
        data_split, scores, rank_weights, labels, ideal_metrics
    )

    return {
        "query normalized maximum likelihood": QN_ML,
        "dataset normalized maximum likelihood": N_ML,
    }