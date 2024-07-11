# License: MIT License

import sys
import time

import numpy as np
import utils.dataset as dataset
import utils.plackettluce as pl
import xgboost as xgb
from scipy.sparse import csr_matrix
from xgboost import DMatrix


def PL_rank_3_grad(rank_weights, labels, scores, sampled_rankings):
    n_docs = labels.shape[0]
    cutoff = min(rank_weights.shape[0], n_docs)

    n_samples = sampled_rankings.shape[0]

    cutoff_sampled_rankings = sampled_rankings[:, :cutoff]

    scores = scores - np.max(scores) + 10.0

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

    return -(result1 + np.mean(second_part, axis=0))


def PL_rank_3_hess(rank_weights, labels, scores, sampled_rankings):
    n_docs = labels.shape[0]
    # second order
    result2 = np.ones(n_docs, dtype=np.float64)

    return result2


def plrank3obj(preds, dtrain):
    group_ptr = dtrain.get_uint_info("group_ptr")
    labels = dtrain.get_label()
    labels = 2 ** labels - 1

    # number of rankings
    n_samples = 200

    grad = np.zeros(len(labels), dtype=np.float64)
    hess = np.zeros(len(labels), dtype=np.float64)

    group = np.diff(group_ptr)
    max_query_size = max(group)
    longest_metric_weights = 1.0 / np.log2(np.arange(max_query_size) + 2)

    # number of docs to display
    cutoff = int(sys.argv[1])
    # share rankings or not
    share_rankings = True

    max_ranking_size = np.min((cutoff, max_query_size))
    metric_weights = longest_metric_weights[:max_ranking_size]

    for q in range(len(group_ptr) - 1):
        q_l = labels[group_ptr[q] : group_ptr[q + 1]]
        scores = preds[group_ptr[q] : group_ptr[q + 1]]

        sampled_rankings = pl.gumbel_sample_rankings(
            scores, n_samples, cutoff=cutoff, return_full_rankings=True
        )
        if share_rankings == True:
            # first order
            grad[group_ptr[q] : group_ptr[q + 1]] = PL_rank_3_grad(
                metric_weights, q_l, scores.astype(np.float64), sampled_rankings
            )
            # second order
            hess[group_ptr[q] : group_ptr[q + 1]] = PL_rank_3_hess(
                metric_weights, q_l, scores.astype(np.float64), sampled_rankings
            )
        else:
            # first order
            grad[group_ptr[q] : group_ptr[q + 1]] = PL_rank_3_grad(
                metric_weights, q_l, scores.astype(np.float64), sampled_rankings[:100]
            )
            # second order
            hess[group_ptr[q] : group_ptr[q + 1]] = PL_rank_3_hess(
                metric_weights, q_l, scores.astype(np.float64), sampled_rankings[100:]
            )

    return grad, hess


def dcg_at_k(rel, k):
    rel = np.asfarray(rel)[:k]
    if rel.size:
        return np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    return 0.0


def ideal_metrics(dtrain, k):
    group_ptr = dtrain.get_uint_info("group_ptr")
    labels = dtrain.get_label()
    idcg_results = []

    for q in range(len(group_ptr) - 1):
        relevance_labels = labels[group_ptr[q] : group_ptr[q + 1]]
        idcg_results.append(dcg_at_k(sorted(relevance_labels, reverse=True), k))

    return np.mean(np.array(idcg_results))


def ndcg_dataset(preds, dtrain):
    group_ptr = dtrain.get_uint_info("group_ptr")
    labels = dtrain.get_label()
    results = []
    k_value = int(sys.argv[1])
    idcg = ideal_metrics(dtrain, k_value)

    for q in range(len(group_ptr) - 1):
        relevance_labels = labels[group_ptr[q] : group_ptr[q + 1]]
        document_scores = preds[group_ptr[q] : group_ptr[q + 1]]
        document_data = list(zip(document_scores, relevance_labels))
        document_data.sort(reverse=True, key=lambda x: x[0])
        sorted_relevance = [item[1] for item in document_data]
        results.append(dcg_at_k(sorted_relevance, k_value) / idcg)

    return "NDCG@{}".format(int(sys.argv[1])), float(np.mean(np.array(results)))


# Custom callback to record running time for each round
class TimingCallback(xgb.callback.TrainingCallback):
    def before_training(self, model):
        self.results = []
        return model

    def after_iteration(self, model, epoch, evals_log):
        elapsed_time = time.time() - start_time
        round_time = {"iteration": epoch + 1, "time": elapsed_time}
        self.results.append(round_time)
        return False


data = dataset.get_dataset_from_json_info("Webscope_C14_Set1", "local_dataset_info.txt")
fold_id = (1 - 1) % data.num_folds()
data = data.get_data_folds()[fold_id]

data.read_data()

train_n_queries = data.train.num_queries()
train_array = np.concatenate(
    [data.train.query_feat(i) for i in range(train_n_queries)], axis=0
)
train_sparse_matrix = csr_matrix(train_array)
train_labels = data.train.label_vector
new_train = DMatrix(train_sparse_matrix, train_labels)
new_train.set_group([data.train.query_feat(i).shape[0] for i in range(train_n_queries)])

test_n_queries = data.test.num_queries()
test_array = np.concatenate(
    [data.test.query_feat(i) for i in range(test_n_queries)], axis=0
)
test_sparse_matrix = csr_matrix(test_array)
test_labels = data.test.label_vector
new_test = DMatrix(test_sparse_matrix, test_labels)
new_test.set_group([data.test.query_feat(i).shape[0] for i in range(test_n_queries)])

validation_n_queries = data.validation.num_queries()
validation_array = np.concatenate(
    [data.validation.query_feat(i) for i in range(validation_n_queries)], axis=0
)
validation_sparse_matrix = csr_matrix(validation_array)
validation_labels = data.validation.label_vector
new_valid = DMatrix(validation_sparse_matrix, validation_labels)
new_valid.set_group(
    [data.validation.query_feat(i).shape[0] for i in range(validation_n_queries)]
)


# plrank3
params = {
    "verbosity": 0,
    "learning_rate": 0.0001471490665625388,
    "max_depth": 8,
    "min_child_weight": 6,
    "gamma": 0.002185898177940798,
    "lambda": 0.02913371587094604,
    "alpha": 5.94134670500851e-05,
    "disable_default_eval_metric": 1,
}

timing_callback = TimingCallback()
ndcg_result = {}

start_time = time.time()
model = xgb.train(
    params,
    new_train,
    num_boost_round=1000,
    evals=[(new_test, "test")],
    obj=plrank3obj,
    custom_metric=ndcg_dataset,
    evals_result=ndcg_result,
    verbose_eval=False,
    callbacks=[timing_callback],
)

print(ndcg_result)
print(timing_callback.results)