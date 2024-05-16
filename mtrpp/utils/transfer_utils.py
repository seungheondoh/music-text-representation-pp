import os
import torch
from torch import nn
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer, set_seed
from sklearn import metrics


def single_query_evaluation(targets, logits, save_dir, labels):
    """
    target = Pandas DataFrame Binary Mtrix( track x label )
    logits = Pandas DataFrame Logit Mtrix( track x label )
    label = tag list
    """
    targets = targets[labels]
    logits = logits[labels]
    roc_auc = metrics.roc_auc_score(targets, logits, average='macro')
    pr_auc = metrics.average_precision_score(targets, logits, average='macro')
    results = {
        'roc_auc' :roc_auc,
        'pr_auc': pr_auc
    }
    # tag wise score
    roc_aucs = metrics.roc_auc_score(targets, logits, average=None)
    pr_aucs = metrics.average_precision_score(targets, logits, average=None)
    tag_wise = {}
    for i in range(len(labels)):
        tag_wise[labels[i]] = {
            "roc_auc":roc_aucs[i], 
            "pr_auc":pr_aucs[i]
    }
    results['tag_wise'] = tag_wise
    label_len = len(labels)
    with open(os.path.join(save_dir, f"{label_len}_results.json"), mode="w") as io:
        json.dump(results, io, indent=4)
        
def compute_accuracy_metrics(ground_truth, predicted, threshold=0.5):
    decisions = predicted > threshold
    binary_pred = decisions.astype(np.int16)
    return metrics.classification_report(ground_truth, binary_pred, output_dict=True)

def get_binary_decisions(ground_truth, predicted):
    """https://github.com/MTG/mtg-jamendo-dataset/blob/31507d6e9a64da11471bb12168372db4f98d7783/scripts/mediaeval/calculate_decisions.py#L8"""
    thresholds = {}
    avg_fscore_macro, avg_fscore_weighted = [], []
    for idx in range(len(ground_truth[0])):
        precision, recall, threshold = metrics.precision_recall_curve(
            ground_truth[:, idx], predicted[:, idx])
        f_score = np.nan_to_num(
            (2 * precision * recall) / (precision + recall))
        thresholds[idx] = threshold[np.argmax(f_score)]

        results = compute_accuracy_metrics(ground_truth[:, idx], predicted[:, idx], threshold=0.5)
        avg_fscore_weighted.append(results['weighted avg']['f1-score'])
        avg_fscore_macro.append(results['macro avg']['f1-score'])
    avg_macro_f1 = np.array(avg_fscore_macro).mean()
    avg_fscore_weighted = np.array(avg_fscore_weighted).mean()
    bestF1_decisions = predicted > np.array(list(thresholds.values()))
    return thresholds, avg_macro_f1, avg_fscore_weighted, bestF1_decisions

def get_evaluation(binary, logit, labels, task_type):
    if task_type == "multilabel":
        _, avg_macro_f1, avg_fscore_weighted, bestF1_decisions = get_binary_decisions(binary, logit)
        roc_auc = metrics.roc_auc_score(binary, logit, average='macro')
        pr_auc = metrics.average_precision_score(binary, logit, average='macro')
        results = {
            "macro_roc": roc_auc,
            "macro_pr": pr_auc,
            "macro_fl": metrics.f1_score(binary, bestF1_decisions, average='macro'),
            "avg_macro_f1": avg_macro_f1,
            "avg_fscore_weighted": avg_fscore_weighted
        }
        # tag wise score
        roc_aucs = metrics.roc_auc_score(binary, logit, average=None)
        pr_aucs = metrics.average_precision_score(binary, logit, average=None)
        tag_wise = {}
        for i in range(len(labels)):
            tag_wise[labels[i]] = {
                "roc_auc":roc_aucs[i], 
                "pr_auc":pr_aucs[i]
        }
        results['tag_wise'] = tag_wise
    else:
        results = {
            "acc": metrics.accuracy_score(binary.argmax(axis=1), logit.argmax(axis=1))
        }
    return results

def get_cls_config(args):
    task_type = "multilabel"
    loss_fn = nn.BCELoss()
    print(args.eval_dataset)
    if args.eval_dataset in ["msd", "mtat", 'mtg_top50tags']:
        output_dim = 50
    elif args.eval_dataset == "kvt":
        output_dim = 42
    elif args.eval_dataset == "mtg_genre":
        output_dim = 87
    elif args.eval_dataset == "mtg_instrument":
        output_dim = 40
    elif args.eval_dataset == "mtg_moodtheme":
        output_dim = 56
    elif args.eval_dataset == "openmic":
        output_dim = 20
    elif args.eval_dataset == "gtzan":
        task_type = "multiclass"
        output_dim = 10
        loss_fn = nn.CrossEntropyLoss()
    elif args.eval_dataset == "emotify":
        task_type = "multiclass"
        output_dim = 9
        loss_fn = nn.CrossEntropyLoss()
    elif args.eval_dataset == "fma":
        task_type = "multiclass"
        output_dim = 8
        loss_fn = nn.CrossEntropyLoss()
    return task_type, output_dim, loss_fn


def print_model_params(args, model):
    n_parameters = sum(p.numel() for p in model.parameters())
    train_n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("============")
    print("lr: %.2e" % (args.lr))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    print('number train of params (M): %.2f' % (train_n_parameters / 1.e6))
    print("============")