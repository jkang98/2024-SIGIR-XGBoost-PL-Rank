# License: MIT License

import sys
import time

import numpy as np
import torch
import torch.optim as optim
import utils.dataset as dataset
import utils.evaluation as evl
import utils.plackettluce as pl
from torch import nn


def init_model():
    nn_model = nn.Sequential(
        nn.Linear(519, 118, dtype=torch.float64),
        nn.Sigmoid(),
        nn.Linear(118, 1, dtype=torch.float64),
    )
    return nn_model


def PL_rank_3(rank_weights, labels, scores, n_samples):
    n_docs = labels.shape[0]
    cutoff = min(rank_weights.shape[0], n_docs)

    sampled_rankings = pl.gumbel_sample_rankings(
        scores, n_samples, cutoff=cutoff, return_full_rankings=True
    )

    cutoff_sampled_rankings = sampled_rankings[:, :cutoff]

    scores = scores.copy() - np.amax(scores) + 10.0

    srange = np.arange(n_samples)
    crange = np.arange(cutoff)

    weighted_labels = labels[cutoff_sampled_rankings] * rank_weights[None, :cutoff]
    cumsum_labels = np.cumsum(weighted_labels[:, ::-1], axis=1)[:, ::-1]

    # first order
    result1 = np.zeros(n_docs, dtype=np.float64)
    np.add.at(result1, cutoff_sampled_rankings[:, :-1], cumsum_labels[:, 1:])
    result1 /= n_samples

    exp_scores = np.exp(scores).astype(np.float64)
    denom_per_rank = np.cumsum(exp_scores[sampled_rankings[:, ::-1]], axis=1)[
        :, : -cutoff - 1 : -1
    ]

    # DR
    cumsum_weight_denom = np.cumsum(rank_weights[:cutoff] / denom_per_rank, axis=1)
    # RI
    cumsum_reward_denom = np.cumsum(cumsum_labels / denom_per_rank, axis=1)

    relevant_docs = np.where(np.not_equal(labels, 0))[0]
    if cutoff < n_docs:
        second_part = -exp_scores[None, :] * cumsum_reward_denom[:, -1, None]
        second_part[:, relevant_docs] += (
            labels[relevant_docs][None, :]
            * exp_scores[None, relevant_docs]
            * cumsum_weight_denom[:, -1, None]
        )
    else:
        second_part = np.empty((n_samples, n_docs), dtype=np.float64)

    sampled_direct_reward = (
        labels[cutoff_sampled_rankings]
        * exp_scores[cutoff_sampled_rankings]
        * cumsum_weight_denom
    )
    sampled_following_reward = exp_scores[cutoff_sampled_rankings] * cumsum_reward_denom
    second_part[srange[:, None], cutoff_sampled_rankings] = (
        sampled_direct_reward - sampled_following_reward
    )

    return result1 + np.mean(second_part, axis=0)


cutoff = int(sys.argv[1])
num_samples = 200

n_epochs = 1000

data = dataset.get_dataset_from_json_info("Webscope_C14_Set1", "local_dataset_info.txt")
fold_id = (1 - 1) % data.num_folds()
data = data.get_data_folds()[fold_id]


data.read_data()

max_ranking_size = np.min((cutoff, data.max_query_size()))

model = init_model()

epoch_results = []

longest_possible_metric_weights = 1.0 / np.log2(np.arange(data.max_query_size()) + 2)
metric_weights = longest_possible_metric_weights[:max_ranking_size]
train_labels = 2 ** data.train.label_vector - 1
vali_labels = 2 ** data.validation.label_vector - 1
test_labels = 2 ** data.test.label_vector - 1
ideal_train_metrics = evl.ideal_metrics(data.train, metric_weights, train_labels)
ideal_vali_metrics = evl.ideal_metrics(data.validation, metric_weights, vali_labels)
ideal_test_metrics = evl.ideal_metrics(data.test, metric_weights, test_labels)

test_result = evl.compute_results(
    data.test, model, metric_weights, test_labels, ideal_test_metrics,
)
epoch_results.append(
    {"epoch": 0, "total time": 0, "test result": test_result,}
)

n_queries = data.train.num_queries()

optimizer = torch.optim.SGD(model.parameters(), lr=0.0011345339558965816)

batch_size = 64
start_time = time.time()
for epoch_i in range(n_epochs):
    query_permutation = np.random.permutation(n_queries)
    for batch_i in range(int(np.ceil(n_queries / batch_size))):
        batch_queries = query_permutation[
            batch_i * batch_size : (batch_i + 1) * batch_size
        ]
        cur_batch_size = batch_queries.shape[0]
        batch_ranges = np.zeros(cur_batch_size + 1, dtype=np.int64)
        batch_features = [data.train.query_feat(batch_queries[0])]
        batch_ranges[1] = batch_features[0].shape[0]
        for i in range(1, cur_batch_size):
            batch_features.append(data.train.query_feat(batch_queries[i]))
            batch_ranges[i + 1] = batch_ranges[i] + batch_features[i].shape[0]
        batch_features = torch.from_numpy(np.concatenate(batch_features, axis=0))

        batch_tf_scores = model(batch_features)
        loss = 0
        batch_doc_weights = np.zeros(batch_features.shape[0], dtype=np.float64)

        for i, qid in enumerate(batch_queries):
            q_labels = data.train.query_values_from_vector(qid, train_labels)
            q_feat = batch_features[batch_ranges[i] : batch_ranges[i + 1], :]
            q_ideal_metric = ideal_train_metrics[qid]

            if q_ideal_metric != 0:
                q_metric_weights = metric_weights
                q_tf_scores = model(q_feat)

                q_np_scores = q_tf_scores.detach().numpy()[:, 0]

                doc_weights = PL_rank_3(
                    q_metric_weights, q_labels, q_np_scores, n_samples=num_samples
                )
                batch_doc_weights[batch_ranges[i] : batch_ranges[i + 1]] = doc_weights

        loss = -torch.sum(batch_tf_scores[:, 0] * torch.from_numpy(batch_doc_weights))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_train_time = time.time() - start_time
    with torch.no_grad():
        test_result = evl.compute_results(
            data.test, model, metric_weights, test_labels, ideal_test_metrics,
        )
    epoch_results.append(
        {
            "epoch": epoch_i + 1,
            "total time": total_train_time,
            "test result": test_result,
        }
    )

print(epoch_results)