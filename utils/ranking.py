# Use code from Harrie Oosterhuis's repository
# URL: https://github.com/HarrieO/2022-SIGIR-plackett-luce
# License: MIT License

import numpy as np


def cutoff_ranking(scores, cutoff, invert=False):
    n_docs = scores.shape[0]
    cutoff = min(n_docs, cutoff)
    full_partition = np.argpartition(scores, cutoff - 1)
    partition = full_partition[:cutoff]
    sorted_partition = np.argsort(scores[partition])
    ranked_partition = partition[sorted_partition]
    if not invert:
        return ranked_partition
    else:
        full_partition[:cutoff] = ranked_partition
        inverted = np.empty(n_docs, dtype=ranked_partition.dtype)
        inverted[full_partition] = np.arange(n_docs)
        return ranked_partition, inverted


def multiple_cutoff_rankings(scores, cutoff, return_full_rankings):
    n_samples = scores.shape[0]
    n_docs = scores.shape[1]
    cutoff = min(n_docs, cutoff)

    ind = np.arange(n_samples)
    partition = np.argpartition(scores, cutoff - 1)
    sorted_partition = np.argsort(scores[ind[:, None], partition[:, :cutoff]])
    rankings = partition[ind[:, None], sorted_partition]

    if return_full_rankings:
        partition[:, :cutoff] = rankings
        rankings = partition

    return rankings