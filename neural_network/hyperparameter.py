# License: MIT License

import sys
import time

import numpy as np
import optuna
import torch
import torch.optim as optim
import utils.dataset as dataset
import utils.evaluation as evl
import utils.plackettluce as pl
from torch import nn


def define_model(trial):
    # We optimize the number of layers and hidden units in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layers = []

    in_features = 519
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features, dtype=torch.float64))
        layers.append(nn.Sigmoid())
        in_features = out_features
    layers.append(nn.Linear(in_features, 1, dtype=torch.float64))

    return nn.Sequential(*layers)


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


data = dataset.get_dataset_from_json_info("Webscope_C14_Set1", "local_dataset_info.txt")
fold_id = (1 - 1) % data.num_folds()
data = data.get_data_folds()[fold_id]

data.read_data()

max_ranking_size = np.min((cutoff, data.max_query_size()))

longest_possible_metric_weights = 1.0 / np.log2(np.arange(data.max_query_size()) + 2)
metric_weights = longest_possible_metric_weights[:max_ranking_size]
train_labels = 2 ** data.train.label_vector - 1
vali_labels = 2 ** data.validation.label_vector - 1
test_labels = 2 ** data.test.label_vector - 1
ideal_train_metrics = evl.ideal_metrics(data.train, metric_weights, train_labels)
ideal_vali_metrics = evl.ideal_metrics(data.validation, metric_weights, vali_labels)
ideal_test_metrics = evl.ideal_metrics(data.test, metric_weights, test_labels)

n_queries = data.train.num_queries()


def objective(trial):
    # Generate the model.
    model = define_model(trial)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    # Training of the model.
    n_epochs = 200
    best_score = 0
    patience = 20
    counter = 0
    for epoch_i in range(n_epochs):
        last_method_train_time = time.time()
        query_permutation = np.random.permutation(n_queries)
        model.train()
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
                    batch_doc_weights[
                        batch_ranges[i] : batch_ranges[i + 1]
                    ] = doc_weights

            loss = -torch.sum(
                batch_tf_scores[:, 0] * torch.from_numpy(batch_doc_weights)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        with torch.no_grad():
            vali_result = evl.compute_results(
                data.validation, model, metric_weights, vali_labels, ideal_vali_metrics,
            )
        if best_score < vali_result["dataset normalized maximum likelihood"]:
            best_score = vali_result["dataset normalized maximum likelihood"]
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                return best_score

    return best_score


# Run hyperparameter search
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=12 * 60 * 60)

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  NDCG@{}: {}".format(int(sys.argv[1]), trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))