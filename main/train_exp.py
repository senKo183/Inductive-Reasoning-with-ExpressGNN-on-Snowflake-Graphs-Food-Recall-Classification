"""Experimental training and ablation runner for V_1/ExpressGNN (Step 2 & Step 4 of Roadmap).

Leaves original main/train.py and model/gcn.py 100% untouched.
Supports:
  -gcn_mode {original,degnorm}  : selects model.gcn.GCN vs model.gcn_degnorm.GCN
  -obs_loss_weight <float>      : optional supervised BCE weight on observed target facts (0.0 = pure ELBO)
  -fix_mln_opt {0,1}            : whether to include mln.parameters() in Adam when rule_weights_learning=1
  -eval_every <int>             : evaluate every N epochs
Computes:
  - Pooled AUC-ROC, Pooled AUC-PR
  - Within-class AUC-ROC (Class_I, Class_II, Class_III)
  - 3-class Multiclass Accuracy, Macro-F1, Balanced Accuracy (evaluating all 3 classes per test record)
  - Prediction min/max/mean/std, positive/negative means, unique scores
"""
import sys, os, math, random, json, time
from os.path import join as joinpath
from itertools import chain
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, accuracy_score

from common.cmd_args import cmd_opt, cmd_args
from common.predicate import PRED_DICT
from common.constants import const_dict, TYPE_SET
from common.utils import EarlyStopMonitor, get_lr, count_parameters
from data_process.dataset import Dataset
from model.graph import KnowledgeGraph
from model.mean_field_posterior import FactorizedPosterior
from model.mln import ConditionalMLN
from model.gcn import GCN as OriginalGCN, TrainableEmbedding
from model.gcn_degnorm import GCN as DegNormGCN


def compute_entropy(posterior_prob):
    eps = 1e-6
    p = posterior_prob.clamp(eps, 1 - eps)
    compl = 1.0 - p
    return -(p * torch.log(p) + compl * torch.log(compl)).sum()


def parse_extra_args():
    cmd_opt.add_argument('-gcn_mode', default='original', choices=['original', 'degnorm'])
    cmd_opt.add_argument('-obs_loss_weight', default=0.0, type=float)
    cmd_opt.add_argument('-obs_batchsize', default=128, type=int)
    cmd_opt.add_argument('-mask_target_graph', default=1, type=int,
                         help='1: exclude target query predicates from GCN input graph so train and test hub-and-spoke nodes have identical degree and edge types')
    cmd_opt.add_argument('-fix_mln_opt', default=0, type=int)
    cmd_opt.add_argument('-json_out', default=None, type=str)
    args, _ = cmd_opt.parse_known_args()
    return args


def get_query_class_and_record(q_tuple):
    # q_tuple is (label, pred_name, args_tuple)
    label, pred_name, args = q_tuple
    if len(args) == 2:
        return args[0], args[1], False
    else:
        # unary predicate e.g. class_I(record_000003)
        cls_map = {'class_I': 'Class_I', 'class_II': 'Class_II', 'class_III': 'Class_III'}
        return args[0], cls_map.get(pred_name, pred_name), True


def evaluate_full(dataset, kg, gcn, posterior_model):
    with torch.no_grad():
        node_embeds = gcn(dataset)
        q_inputs = [(e[1], e[2]) for e in dataset.test_fact_ls]
        scores = posterior_model(q_inputs, node_embeds).detach().cpu().numpy().reshape(-1)
        labels = np.asarray([e[0] for e in dataset.test_fact_ls]).reshape(-1)

        auc_roc = float(roc_auc_score(labels, scores))
        auc_pr = float(average_precision_score(labels, scores))

        parsed = [get_query_class_and_record(e) for e in dataset.test_fact_ls]
        q_recs = np.array([p[0] for p in parsed])
        q_cls = np.array([p[1] for p in parsed])
        is_unary = parsed[0][2]

        within_auc = {}
        class_means = {}
        for c in ['Class_I', 'Class_II', 'Class_III']:
            m = (q_cls == c)
            if np.any(m) and len(set(labels[m])) == 2:
                within_auc[c] = float(roc_auc_score(labels[m], scores[m]))
                class_means[c] = float(scores[m].mean())

        # Multiclass evaluation across all 3 classes for every unique test record
        pos_mask = (labels == 1)
        if np.any(pos_mask) and set(q_cls).issubset({'Class_I', 'Class_II', 'Class_III'}):
            test_rec_list = list(q_recs[pos_mask])
            true_cls_list = list(q_cls[pos_mask])
            classes = ['Class_I', 'Class_II', 'Class_III']
            unary_names = {'Class_I': 'class_I', 'Class_II': 'class_II', 'Class_III': 'class_III'}
            mc_scores = np.zeros((len(test_rec_list), 3), dtype=np.float32)
            for j, c in enumerate(classes):
                if is_unary:
                    cand = [(unary_names[c], (r,)) for r in test_rec_list]
                else:
                    cand = [('has_classification', (r, c)) for r in test_rec_list]
                mc_scores[:, j] = posterior_model(cand, node_embeds).detach().cpu().numpy().reshape(-1)
            pred_cls = [classes[i] for i in mc_scores.argmax(axis=1)]
            mc_acc = float(accuracy_score(true_cls_list, pred_cls))
            mc_f1 = float(f1_score(true_cls_list, pred_cls, labels=classes, average='macro', zero_division=0))
            mc_bal = float(balanced_accuracy_score(true_cls_list, pred_cls))
        else:
            mc_acc = mc_f1 = mc_bal = float('nan')

        return {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'within_auc': within_auc,
            'class_means': class_means,
            'mc_acc': mc_acc,
            'mc_macro_f1': mc_f1,
            'mc_bal_acc': mc_bal,
            's_min': float(scores.min()),
            's_max': float(scores.max()),
            's_mean': float(scores.mean()),
            's_std': float(scores.std()),
            'uniq6': int(len(np.unique(np.round(scores, 6)))),
            'mean_pos': float(scores[labels == 1].mean()),
            'mean_neg': float(scores[labels == 0].mean()),
        }


def main():
    args = parse_extra_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = Dataset(args.data_root, args.batchsize, args.shuffle_sampling, load_method=args.load_method)
    target_preds = {e[1] for e in dataset.test_fact_ls}
    if args.mask_target_graph == 1:
        gcn_facts = {k: (set() if k in target_preds else v) for k, v in dataset.fact_dict.items()}
    else:
        gcn_facts = dataset.fact_dict
    kg = KnowledgeGraph(gcn_facts, PRED_DICT, dataset)

    GCNClass = OriginalGCN if args.gcn_mode == 'original' else DegNormGCN
    if args.use_gcn == 1:
        gcn = GCNClass(kg, args.embedding_size - args.gcn_free_size, args.gcn_free_size,
                       num_hops=args.num_hops, num_layers=args.num_mlp_layers,
                       transductive=(args.trans == 1)).to(args.device)
    else:
        gcn = TrainableEmbedding(kg, args.embedding_size).to(args.device)

    posterior_model = FactorizedPosterior(kg, args.embedding_size, args.slice_dim).to(args.device)
    mln = ConditionalMLN(args, dataset.rule_ls)

    param_groups = [posterior_model.parameters(), gcn.parameters()]
    if args.fix_mln_opt == 1 and args.rule_weights_learning == 1:
        param_groups.append(mln.parameters())
    optimizer = optim.Adam(chain.from_iterable(param_groups), lr=args.learning_rate, weight_decay=args.l2_coef)

    # Collect observed target facts for optional supervised cross-entropy (if obs_loss_weight > 0)
    obs_target_facts = []
    for p in sorted(target_preds):
        for val, consts in dataset.fact_dict.get(p, set()):
            obs_target_facts.append((val, p, consts))

    history = []
    init_metrics = evaluate_full(dataset, kg, gcn, posterior_model)
    init_metrics['epoch'] = -1
    init_metrics['train_loss'] = float('nan')
    history.append(init_metrics)
    print(f"[Init] AUC-ROC={init_metrics['auc_roc']:.4f} AUC-PR={init_metrics['auc_pr']:.4f} "
          f"MacroF1={init_metrics['mc_macro_f1']:.4f} BalAcc={init_metrics['mc_bal_acc']:.4f} "
          f"within={init_metrics['within_auc']} pos/neg={init_metrics['mean_pos']:.4f}/{init_metrics['mean_neg']:.4f} "
          f"uniq={init_metrics['uniq6']}", flush=True)

    for epoch in range(args.num_epochs):
        acc_loss = 0.0
        t0 = time.time()
        for k in range(args.num_batches):
            node_embeds = gcn(dataset)
            bnm, flat, blv, orc, bov = dataset.get_batch_rnd(
                observed_prob=args.observed_prob,
                filter_latent=(args.filter_latent == 1),
                closed_world=(args.closed_world == 1),
                filter_observed=1
            )
            if len(flat) > 0:
                posterior_prob = posterior_model(flat, node_embeds)
                entropy = 0.0 if args.no_entropy == 1 else compute_entropy(posterior_prob) / args.entropy_temp
                potential = mln(bnm, blv, orc, posterior_prob.to('cpu'), flat, bov)
                elbo_loss = -(potential + entropy.to('cpu')) / args.batchsize
            else:
                elbo_loss = torch.tensor(0.0)

            if args.obs_loss_weight > 0.0 and len(obs_target_facts) > 0:
                batch_obs = random.sample(obs_target_facts, min(args.obs_batchsize, len(obs_target_facts)))
                obs_inputs = [(p, c) for _, p, c in batch_obs]
                obs_labels = torch.tensor([float(v) for v, _, _ in batch_obs], device=args.device)
                obs_probs = posterior_model(obs_inputs, node_embeds).clamp(1e-6, 1.0 - 1e-6)
                obs_bce = F.binary_cross_entropy(obs_probs, obs_labels)
                loss = elbo_loss + args.obs_loss_weight * obs_bce.to('cpu')
            else:
                loss = elbo_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_loss += float(loss.item())

        m = evaluate_full(dataset, kg, gcn, posterior_model)
        m['epoch'] = epoch
        m['train_loss'] = acc_loss / max(1, args.num_batches)
        m['epoch_sec'] = round(time.time() - t0, 2)
        history.append(m)
        print(f"[Epoch {epoch:2d} ({m['epoch_sec']}s)] loss={m['train_loss']:.4f} "
              f"AUC-ROC={m['auc_roc']:.4f} AUC-PR={m['auc_pr']:.4f} "
              f"MacroF1={m['mc_macro_f1']:.4f} BalAcc={m['mc_bal_acc']:.4f} "
              f"within={m['within_auc']} pos/neg={m['mean_pos']:.4f}/{m['mean_neg']:.4f} "
              f"uniq={m['uniq6']}", flush=True)

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, 'w') as f:
            json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()

