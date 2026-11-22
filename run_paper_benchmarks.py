"""
Fast, Exact Benchmark & Ablation Script on food_v4_unary for the ExpressGNN Paper.
Computes and prints all requested metrics on food_v4_unary:
1. Graph Topology Statistics: event_id degree (mean, std, median, P95, max) & component size distribution.
2. 1x2 Table on food_v4_unary: Transductive (-trans 1) vs. Inductive (-trans 0) over 3 seeds (mean ± std),
   reporting AUC-ROC, AUC-PR, Macro-F1, and Class-Specific AUC-PR (especially Class III).
3. Receptive Field (-num_hops in {1, 2, 3}) on food_v4_unary: metrics, time per epoch (sec/epoch),
   and average hidden embedding L2-norm ||h^(K)||_2 across event-hub degree buckets.
4. Rule Weights Learning (-rule_weights_learning 0 vs. 1) on food_v4_unary:
   epoch-by-epoch AUC-ROC / AUC-PR curve and initial vs. final weights of 6 representative rules.
5. Baselines & Component Ablations on food_v4_unary (3 seeds, mean ± std):
   - Logistic Regression before and after event pooling
   - Full food_v4_unary (Event Pooling + Degree Unification deg=5 + 20-Tier Quantization)
   - food_v4_unary w/o Event Pooling
   - food_v4_unary w/o Degree Unification (with train target label edges: deg 8 vs. 5)
   - food_v4_unary w/ 10-Tier Quantization
   - food_v4_unary w/ 5-Tier Quantization
"""

import math
import random
import re
import time
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

from common.cmd_args import cmd_args
from common.constants import TYPE_SET, const_dict
from common.predicate import PRED_DICT
from data_process.dataset import Dataset
from model.graph import KnowledgeGraph
from model.mean_field_posterior import FactorizedPosterior
from model.mln import ConditionalMLN
from model.mlp import MLP

_DATASET_CACHE = {}


def get_dataset_and_kg(data_root: str, batchsize: int = 128):
    key = str(data_root)
    if key not in _DATASET_CACHE:
        PRED_DICT.clear()
        TYPE_SET.clear()
        const_dict.constants.clear()
        ds = Dataset(key, batchsize, shuffle_sampling=1, load_method=0)
        kg = KnowledgeGraph(ds.fact_dict, dict(PRED_DICT), ds)
        _DATASET_CACHE[key] = (ds, kg, dict(PRED_DICT))
    ds, kg, pred_dict_snapshot = _DATASET_CACHE[key]
    PRED_DICT.clear()
    PRED_DICT.update(pred_dict_snapshot)
    ds.batchsize = batchsize
    return ds, kg


class FastExactGCN(nn.Module):
    """
    Mathematically 100% equivalent to model.gcn.GCN, but replaces the dense
    (num_nodes x (num_ents + num_rels)) one-hot matrix multiplication in init_node_linear
    with an O(num_nodes) one-hot column lookup and caches the fixed initial projection
    (since init_node_linear has requires_grad=False in model/gcn.py).
    """

    def __init__(
        self, graph, latent_dim, free_dim, num_hops=2, num_layers=2, transductive=True
    ):
        super().__init__()
        self.graph = graph
        self.latent_dim = latent_dim
        self.free_dim = free_dim
        self.num_hops = num_hops
        self.num_layers = num_layers

        self.num_ents = graph.num_ents
        self.num_rels = graph.num_rels
        self.num_nodes = graph.num_nodes
        self.num_edges = graph.num_edges
        self.num_edge_types = len(graph.edge_type2idx)

        (
            self.edge2node_in,
            self.edge2node_out,
            self.node_degree,
            self.edge_type_masks,
            self.edge_direction_masks,
        ) = self.gen_edge2node_mapping()

        const_nodes = []
        col_indices = torch.zeros(graph.num_nodes, dtype=torch.long)
        if transductive:
            self.node_feat_dim = self.num_ents + self.num_rels
            for i in graph.idx2node:
                if isinstance(graph.idx2node[i], str):
                    const_nodes.append(i)
                    col_indices[i] = i
                elif isinstance(graph.idx2node[i], tuple):
                    rel, _ = graph.idx2node[i]
                    col_indices[i] = self.num_ents + graph.rel2idx[rel]
        else:
            self.node_feat_dim = 1 + self.num_rels
            for i in graph.idx2node:
                if isinstance(graph.idx2node[i], str):
                    const_nodes.append(i)
                    col_indices[i] = 0
                elif isinstance(graph.idx2node[i], tuple):
                    rel, _ = graph.idx2node[i]
                    col_indices[i] = 1 + graph.rel2idx[rel]

        self.const_nodes = torch.LongTensor(const_nodes).to(cmd_args.device)

        # Identical uniform initialization to nn.Linear(node_feat_dim, latent_dim, bias=False)
        bound = 1.0 / math.sqrt(self.node_feat_dim)
        weight_T = torch.empty(self.node_feat_dim, latent_dim).uniform_(-bound, bound)
        self.register_buffer("init_node_embeds", weight_T[col_indices].to(cmd_args.device))

        self.edge2node_in = self.edge2node_in.to(cmd_args.device)
        self.edge2node_out = self.edge2node_out.to(cmd_args.device)
        self.edge_type_masks = [
            m.to(cmd_args.device).unsqueeze(1) for m in self.edge_type_masks
        ]
        self.edge_direction_masks = [
            m.to(cmd_args.device).unsqueeze(1) for m in self.edge_direction_masks
        ]

        self.MLPs = nn.ModuleList(
            [
                MLP(
                    input_size=self.latent_dim,
                    num_layers=self.num_layers,
                    hidden_size=self.latent_dim,
                    output_size=self.latent_dim,
                )
                for _ in range(self.num_hops)
            ]
        )

        self.edge_type_W = nn.ModuleList()
        for _ in range(self.num_edge_types):
            ml_edge_type = nn.ModuleList()
            for _ in range(self.num_hops):
                ml_hop = nn.ModuleList(
                    [nn.Linear(latent_dim, latent_dim, bias=False) for _ in range(2)]
                )
                ml_edge_type.append(ml_hop)
            self.edge_type_W.append(ml_edge_type)

        if free_dim > 0:
            self.const_nodes_free_params = nn.Parameter(
                nn.init.kaiming_uniform_(torch.zeros(self.num_ents, free_dim))
            )
        else:
            self.register_buffer(
                "const_nodes_free_params", torch.zeros(self.num_ents, 0)
            )

    def gen_edge2node_mapping(self):
        ei = 0
        edge_idx = 0
        edge2node_in = torch.zeros(self.num_edges * 2, dtype=torch.long)
        edge2node_out = torch.zeros(self.num_edges * 2, dtype=torch.long)
        node_degree = torch.zeros(self.num_nodes)

        edge_type_masks = [
            torch.zeros(self.num_edges * 2) for _ in range(self.num_edge_types)
        ]
        edge_direction_masks = [torch.zeros(self.num_edges * 2) for _ in range(2)]

        for ni, nj in torch.as_tensor(self.graph.edge_pairs):
            edge_type = self.graph.edge_types[edge_idx]
            edge_idx += 1

            edge2node_in[ei] = nj
            edge2node_out[ei] = ni
            node_degree[ni] += 1
            edge_type_masks[edge_type][ei] = 1
            edge_direction_masks[0][ei] = 1
            ei += 1

            edge2node_in[ei] = ni
            edge2node_out[ei] = nj
            node_degree[nj] += 1
            edge_type_masks[edge_type][ei] = 1
            edge_direction_masks[1][ei] = 1
            ei += 1

        edge2node_in = edge2node_in.view(-1, 1).expand(-1, self.latent_dim)
        edge2node_out = edge2node_out.view(-1, 1).expand(-1, self.latent_dim)
        node_degree = node_degree.view(-1, 1)
        return (
            edge2node_in,
            edge2node_out,
            node_degree,
            edge_type_masks,
            edge_direction_masks,
        )

    def forward(self, batch_data):
        node_embeds = self.init_node_embeds
        for hop in range(self.num_hops):
            edge_sources = torch.gather(node_embeds, 0, self.edge2node_out)
            agg_msgs = torch.zeros(
                edge_sources.size(), device=node_embeds.device, dtype=node_embeds.dtype
            )
            for et in range(self.num_edge_types):
                et_mask = self.edge_type_masks[et]
                for ed in range(2):
                    mask = et_mask * self.edge_direction_masks[ed]
                    agg_msgs = agg_msgs + self.edge_type_W[et][hop][ed](
                        edge_sources * mask
                    )
            node_Update = torch.zeros(
                node_embeds.size(), device=node_embeds.device, dtype=node_embeds.dtype
            )
            node_Update.scatter_add_(0, self.edge2node_in, agg_msgs)
            node_embeds = self.MLPs[hop](node_embeds + node_Update)

        node_embeds_const = node_embeds[self.const_nodes]
        if self.free_dim > 0:
            return torch.cat([node_embeds_const, self.const_nodes_free_params], dim=1)
        return node_embeds_const


def compute_entropy(posterior_prob):
    eps = 1e-6
    p = torch.clamp(posterior_prob, eps, 1.0 - eps)
    compl_p = 1.0 - p
    return -(p * torch.log(p) + compl_p * torch.log(compl_p)).sum()


def load_fda_split(root: Path):
    raw = pd.read_csv(
        root / "fda/food_recall.csv", dtype={"event_id": "string"}
    ).reset_index(names="raw_idx")
    rmap = pd.read_csv(root / "fda/record_id_mapping.csv")
    rmap["raw_idx"] = rmap["source_csv_row"] - 2
    raw = raw.merge(
        rmap[["record_id", "event_id", "raw_idx"]],
        on="raw_idx",
        how="left",
        suffixes=("", "_m"),
    )
    raw["event_id"] = raw["event_id_m"].astype("string")

    rc, cc, fm, ev = {}, {}, {}, {}
    for line in (root / "fda/parser_test/facts.txt").read_text().splitlines():
        m = re.fullmatch(r"(\w+)\(([^,]+),\s*([^\)]+)\)", line.strip())
        if not m:
            continue
        pred, arg1, arg2 = m.group(1), m.group(2), m.group(3)
        if pred == "has_reason_category":
            rc[arg1] = arg2
        elif pred == "has_classification":
            cc[arg1] = arg2
        elif pred == "recalled_by":
            fm[arg1] = arg2
        elif pred == "belongs_to_event":
            ev[arg1] = arg2

    raw["rc"] = raw["record_id"].map(rc)
    raw["cc"] = raw["record_id"].map(cc)
    raw["fm"] = raw["record_id"].map(fm)
    raw["ev"] = raw["record_id"].map(ev)

    valid = (
        raw[raw["cc"].isin(["Class_I", "Class_II", "Class_III"]) & raw["rc"].notna()]
        .copy()
        .reset_index(drop=True)
    )
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, te_idx = next(gss.split(valid, groups=valid["event_id"]))
    return valid, tr_idx, te_idx


def evaluate_predictions_detailed(
    y_query_true: np.ndarray,
    y_query_prob: np.ndarray,
    query_preds: list,
    te_df: pd.DataFrame,
    record_class_probs: dict,
):
    auc_roc = roc_auc_score(y_query_true, y_query_prob)
    auc_pr = average_precision_score(y_query_true, y_query_prob)
    log_prob = float(
        np.sum(
            np.log(
                np.clip(
                    np.abs((1.0 - y_query_true) - y_query_prob), 1e-6, 1.0 - 1e-6
                )
            )
        )
    )

    q_preds = np.array(query_preds)
    pr_by_class_query = {}
    for c_pred in ["class_I", "class_II", "class_III"]:
        mask = q_preds == c_pred
        pr_by_class_query[c_pred] = average_precision_score(
            y_query_true[mask], y_query_prob[mask]
        )

    classes = ["Class_I", "Class_II", "Class_III"]
    unary_cl = ["class_I", "class_II", "class_III"]
    y_true_idx = np.array([classes.index(c) for c in te_df["cc"].values])
    P_mat = np.zeros((len(te_df), 3), dtype=np.float64)
    for i, rid in enumerate(te_df["record_id"].values):
        for j, uc in enumerate(unary_cl):
            P_mat[i, j] = record_class_probs.get((uc, rid), 0.333)

    y_pred_idx = np.argmax(P_mat, axis=1)
    macro_f1 = f1_score(y_true_idx, y_pred_idx, average="macro")
    f1_per_class = f1_score(y_true_idx, y_pred_idx, average=None)

    ovr_pr = {}
    for j, uc in enumerate(unary_cl):
        y_bin = (y_true_idx == j).astype(int)
        ovr_pr[uc] = average_precision_score(y_bin, P_mat[:, j])

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "log_prob": log_prob,
        "macro_f1": macro_f1,
        "f1_I": float(f1_per_class[0]),
        "f1_II": float(f1_per_class[1]),
        "f1_III": float(f1_per_class[2]),
        "q_pr_I": pr_by_class_query["class_I"],
        "q_pr_II": pr_by_class_query["class_II"],
        "q_pr_III": pr_by_class_query["class_III"],
        "ovr_pr_I": ovr_pr["class_I"],
        "ovr_pr_II": ovr_pr["class_II"],
        "ovr_pr_III": ovr_pr["class_III"],
    }


def section_1_graph_statistics(valid: pd.DataFrame):
    print("\n" + "=" * 88)
    print("PHẦN 1: THỐNG KÊ ĐỒ THỊ FOOD RECALL (BẬC TÂM EVENT_ID & KÍCH THƯỚC CỤM BÔNG TUYẾT)")
    print("=" * 88)

    ev_deg = valid.groupby("ev")["record_id"].count().values
    print(f"- Tổng số bản ghi (recall_record): {len(valid):,}")
    print(f"- Tổng số sự kiện (event_id hubs): {len(ev_deg):,}")
    print(f"- Tổng số doanh nghiệp (firm_id):  {valid['fm'].nunique():,}")
    print("\n[1.1] Phân bố bậc của tâm sự kiện deg(event_id):")
    print(f"  + Trung bình (Mean ± Std): {np.mean(ev_deg):.2f} ± {np.std(ev_deg):.2f}")
    print(f"  + Trung vị (Median / P50): {np.median(ev_deg):.1f}")
    print(f"  + Phân vị 90 (P90):        {np.percentile(ev_deg, 90):.1f}")
    print(f"  + Phân vị 95 (P95):        {np.percentile(ev_deg, 95):.1f}")
    print(f"  + Phân vị 99 (P99):        {np.percentile(ev_deg, 99):.1f}")
    print(f"  + Cực đại (Max):           {np.max(ev_deg)}")

    b1 = np.sum((ev_deg >= 1) & (ev_deg <= 5))
    b2 = np.sum((ev_deg >= 6) & (ev_deg <= 20))
    b3 = np.sum((ev_deg >= 21) & (ev_deg <= 100))
    b4 = np.sum(ev_deg > 100)
    print("  + Phân nhóm bậc tâm sự kiện:")
    print(f"    * Bậc nhỏ [1, 5]:        {b1:,} events ({b1 / len(ev_deg) * 100:.1f}%)")
    print(f"    * Bậc vừa [6, 20]:       {b2:,} events ({b2 / len(ev_deg) * 100:.1f}%)")
    print(f"    * Bậc lớn [21, 100]:     {b3:,} events ({b3 / len(ev_deg) * 100:.1f}%)")
    print(f"    * Bậc rất lớn (> 100):   {b4:,} events ({b4 / len(ev_deg) * 100:.1f}%)")

    adj = defaultdict(list)
    for row in valid.itertuples(index=False):
        r, e, f = str(row.record_id), str(row.ev), str(row.fm)
        adj[r].append(e)
        adj[e].append(r)
        if pd.notna(row.fm):
            adj[r].append(f)
            adj[f].append(r)

    visited = set()
    comp_sizes = []
    comp_records = []
    for node in adj:
        if node in visited:
            continue
        queue = [node]
        visited.add(node)
        c_nodes = []
        while queue:
            cur = queue.pop()
            c_nodes.append(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        comp_sizes.append(len(c_nodes))
        comp_records.append(sum(1 for u in c_nodes if u.startswith("record_")))

    comp_sizes = np.array(comp_sizes)
    comp_records = np.array(comp_records)
    print("\n[1.2] Phân bố kích thước cụm liên thông (Connected Components):")
    print(f"  + Tổng số cụm liên thông (Components):     {len(comp_sizes):,}")
    print(f"  + Tổng số nút thực thể (Entities):         {len(adj):,}")
    print("  + Kích thước cụm (tổng thực thể record + event + firm):")
    print(f"    * Trung bình (Mean ± Std): {np.mean(comp_sizes):.2f} ± {np.std(comp_sizes):.2f}")
    print(f"    * Trung vị (Median / P50): {np.median(comp_sizes):.1f}")
    print(f"    * Phân vị 90 (P90):        {np.percentile(comp_sizes, 90):.1f}")
    print(f"    * Phân vị 95 (P95):        {np.percentile(comp_sizes, 95):.1f}")
    print(f"    * Phân vị 99 (P99):        {np.percentile(comp_sizes, 99):.1f}")
    print(f"    * Cực đại (Max):           {np.max(comp_sizes)}")
    print("  + Kích thước cụm (riêng số bản ghi recall_record mỗi cụm):")
    print(f"    * Trung bình (Mean ± Std): {np.mean(comp_records):.2f} ± {np.std(comp_records):.2f}")
    print(f"    * Trung vị (Median / P50): {np.median(comp_records):.1f}")
    print(f"    * Phân vị 95 (P95):        {np.percentile(comp_records, 95):.1f}")
    print(f"    * Cực đại (Max):           {np.max(comp_records)}")


def section_2_lr_baselines_and_build_ablation_datasets(
    root: Path, valid: pd.DataFrame, tr_idx: np.ndarray, te_idx: np.ndarray
):
    print("\n" + "=" * 88)
    print("PHẦN 2: BASELINE LOGISTIC REGRESSION (TRƯỚC VÀ SAU KHI GỘP THEO EVENT_ID)")
    print("=" * 88)

    tr = valid.iloc[tr_idx].copy()
    te = valid.iloc[te_idx].copy()
    classes = ["Class_I", "Class_II", "Class_III"]
    unary_cl = {
        "Class_I": "class_I",
        "Class_II": "class_II",
        "Class_III": "class_III",
    }

    vec_r = TfidfVectorizer(
        max_features=1500, ngram_range=(1, 2), min_df=3, sublinear_tf=True
    )
    vec_p = TfidfVectorizer(
        max_features=1500, ngram_range=(1, 2), min_df=3, sublinear_tf=True
    )
    cat_cols = [
        "rc",
        "initial_firm_notification",
        "voluntary_mandated",
        "status",
        "state",
        "country",
        "fm",
    ]
    ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=2)

    X_all_r = vec_r.fit(tr["reason_for_recall"].fillna("")).transform(
        valid["reason_for_recall"].fillna("")
    )
    X_all_p = vec_p.fit(
        tr["product_description"].fillna("")
        + " "
        + tr["distribution_pattern"].fillna("")
    ).transform(
        valid["product_description"].fillna("")
        + " "
        + valid["distribution_pattern"].fillna("")
    )
    X_all_c = ohe.fit(tr[cat_cols].fillna("NA").astype(str)).transform(
        valid[cat_cols].fillna("NA").astype(str)
    )
    X_all = hstack([X_all_r, X_all_p, X_all_c]).tocsr()

    clf = LogisticRegression(max_iter=1000, C=0.3, random_state=42).fit(
        X_all[tr_idx], tr["cc"]
    )
    p_raw = pd.DataFrame(clf.predict_proba(X_all), index=valid.index, columns=classes)
    p_raw["ev"] = valid["event_id"].values
    p_pooled = p_raw.groupby("ev")[classes].transform("mean")

    query_lines = (root / "food_v4_unary/queries").read_text().splitlines()
    y_q_true, q_preds, q_rids = [], [], []
    for line in query_lines:
        line = line.strip()
        if not line:
            continue
        is_neg = line.startswith("!")
        body = line[1:] if is_neg else line
        m = re.fullmatch(r"(\w+)\(([^)]+)\)", body)
        pred_name, rid = m.group(1), m.group(2)
        y_q_true.append(0 if is_neg else 1)
        q_preds.append(pred_name)
        q_rids.append(rid)
    y_q_true = np.array(y_q_true)

    for name, p_df in [
        ("Baseline LR (Trước khi gộp event / Record-level)", p_raw),
        ("Baseline LR + Event Pooling (Sau khi gộp theo event_id)", p_pooled),
    ]:
        rec_probs = {}
        for idx in te_idx:
            rid = valid.loc[idx, "record_id"]
            for c, uc in unary_cl.items():
                rec_probs[(uc, rid)] = float(p_df.loc[idx, c])
        y_q_prob = np.array([rec_probs[(p, r)] for p, r in zip(q_preds, q_rids)])
        res = evaluate_predictions_detailed(y_q_true, y_q_prob, q_preds, te, rec_probs)
        print(
            f"- {name}:\n"
            f"  + Query AUC-ROC: {res['auc_roc']:.4f} | Query AUC-PR: {res['auc_pr']:.4f} | "
            f"Macro-F1: {res['macro_f1']:.4f} | Log-Prob: {res['log_prob']:.2f}\n"
            f"  + Per-Class AUC-PR (Query): Class_I={res['q_pr_I']:.4f}, Class_II={res['q_pr_II']:.4f}, Class_III={res['q_pr_III']:.4f}\n"
            f"  + Per-Class AUC-PR (OvR):   Class_I={res['ovr_pr_I']:.4f}, Class_II={res['ovr_pr_II']:.4f}, Class_III={res['ovr_pr_III']:.4f}"
        )

    ablation_dirs = {}
    edges_20 = [
        0.0, 0.005, 0.015, 0.03, 0.06, 0.10, 0.16, 0.24, 0.34, 0.45,
        0.55, 0.66, 0.76, 0.84, 0.90, 0.94, 0.965, 0.98, 0.99, 0.996, 1.01,
    ]
    edges_10 = [0.0, 0.02, 0.08, 0.20, 0.38, 0.50, 0.62, 0.80, 0.92, 0.98, 1.01]
    edges_5 = [0.0, 0.15, 0.40, 0.60, 0.85, 1.01]

    def build_variant(
        variant_name: str,
        prob_df: pd.DataFrame,
        edges: list,
        include_train_target_facts: bool,
    ):
        vdir = root / f"food_v4_ablation_{variant_name}"
        vdir.mkdir(parents=True, exist_ok=True)
        n_tiers = len(edges) - 1
        facts = []
        preds = {
            "belongs_to_event(recall_record, event)",
            "recalled_by(recall_record, firm)",
            "class_I(recall_record)",
            "class_II(recall_record)",
            "class_III(recall_record)",
        }
        rules = []
        tier_counts = {(c, t): 0 for c in classes for t in range(n_tiers)}
        tr_set = set(tr_idx.tolist())

        for idx, row in valid.iterrows():
            rid = row["record_id"]
            facts.append(f"belongs_to_event({rid}, {row['ev']})")
            facts.append(f"recalled_by({rid}, {row['fm']})")
            for c, uc in unary_cl.items():
                pr = float(prob_df.loc[idx, c])
                t = min(
                    n_tiers - 1,
                    max(0, int(np.searchsorted(edges[1:], pr, side="right"))),
                )
                tier_counts[(c, t)] += 1
                facts.append(f"tier_{uc}_{t}({rid})")
            if include_train_target_facts and idx in tr_set:
                true_c = row["cc"]
                for c, uc in unary_cl.items():
                    if c == true_c:
                        facts.append(f"{uc}({rid})")
                    else:
                        facts.append(f"!{uc}({rid})")

        for c, uc in unary_cl.items():
            for t in range(n_tiers):
                if tier_counts[(c, t)] == 0:
                    continue
                pname = f"tier_{uc}_{t}"
                preds.add(f"{pname}(recall_record)")
                p_mid = min(
                    0.995, max(0.003, (edges[t] + min(1.0, edges[t + 1])) / 2.0)
                )
                if p_mid >= 0.50:
                    w = max(0.10, math.log(p_mid / (1.0 - p_mid)))
                    rules.append(f"{w:.3f} !{pname}(x) v {uc}(x)")
                else:
                    w = max(0.10, math.log((1.0 - p_mid) / p_mid))
                    rules.append(f"{w:.3f} !{pname}(x) v !{uc}(x)")

        (vdir / "facts").write_text("\n".join(facts) + "\n")
        (vdir / "predicates").write_text("\n".join(sorted(preds)) + "\n")
        (vdir / "rules").write_text("\n".join(rules) + "\n")
        (vdir / "queries").write_text("\n".join(query_lines) + "\n")
        return vdir

    ablation_dirs["no_event_pool"] = build_variant(
        "no_event_pool", p_raw, edges_20, include_train_target_facts=False
    )
    ablation_dirs["with_deg_shift"] = build_variant(
        "with_deg_shift", p_pooled, edges_20, include_train_target_facts=True
    )
    ablation_dirs["coarse_10tier"] = build_variant(
        "coarse_10tier", p_pooled, edges_10, include_train_target_facts=False
    )
    ablation_dirs["coarse_5tier"] = build_variant(
        "coarse_5tier", p_pooled, edges_5, include_train_target_facts=False
    )
    return ablation_dirs


def run_single_expressgnn(
    data_root: str,
    te_df: pd.DataFrame,
    trans: int = 0,
    num_hops: int = 2,
    rule_weights_learning: int = 0,
    seed: int = 10,
    num_epochs: int = 6,
    num_batches: int = 40,
    batchsize: int = 128,
    lr: float = 0.0005,
    lr_rule_weights: float = 0.005,
    record_epoch_curve: bool = False,
    compute_degree_norms: bool = False,
    valid_df: pd.DataFrame = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cmd_args.data_root = str(data_root)
    cmd_args.load_method = 0
    cmd_args.trans = trans
    cmd_args.gcn_free_size = 64 if trans == 1 else 0
    cmd_args.embedding_size = 64
    cmd_args.slice_dim = 16
    cmd_args.num_hops = num_hops
    cmd_args.num_mlp_layers = 2
    cmd_args.batchsize = batchsize
    cmd_args.rule_weights_learning = rule_weights_learning
    cmd_args.learning_rate = lr
    cmd_args.learning_rate_rule_weights = lr_rule_weights
    cmd_args.observed_prob = 0.9
    cmd_args.filter_latent = 0
    cmd_args.closed_world = 0
    cmd_args.no_entropy = 0
    cmd_args.entropy_temp = 1.0

    dataset, kg = get_dataset_and_kg(cmd_args.data_root, cmd_args.batchsize)

    gcn = FastExactGCN(
        kg,
        cmd_args.embedding_size - cmd_args.gcn_free_size,
        cmd_args.gcn_free_size,
        num_hops=cmd_args.num_hops,
        num_layers=cmd_args.num_mlp_layers,
        transductive=(cmd_args.trans == 1),
    ).to(cmd_args.device)
    posterior_model = FactorizedPosterior(
        kg, cmd_args.embedding_size, cmd_args.slice_dim
    ).to(cmd_args.device)
    mln = ConditionalMLN(cmd_args, dataset.rule_ls)

    all_params = chain.from_iterable([posterior_model.parameters(), gcn.parameters()])
    optimizer = optim.Adam(all_params, lr=cmd_args.learning_rate, weight_decay=0.0)
    rule_opt = None
    if rule_weights_learning == 1:
        rule_opt = optim.SGD(
            mln.rule_weights_lin.parameters(), lr=cmd_args.learning_rate_rule_weights
        )

    init_weights = mln.rule_weights_lin.weight.detach().cpu().numpy().flatten().copy()

    all_test_rids = te_df["record_id"].values
    all_test_triples = []
    for rid in all_test_rids:
        for uc in ["class_I", "class_II", "class_III"]:
            all_test_triples.append((uc, (rid,)))

    epoch_times = []
    epoch_curve = []

    for ep in range(num_epochs):
        t0 = time.perf_counter()
        for _ in range(num_batches):
            node_embeds = gcn(dataset)
            (
                batch_neg_mask,
                flat_list,
                batch_latent_var_inds,
                observed_rule_cnts,
                batch_observed_vars,
            ) = dataset.get_batch_rnd(
                observed_prob=cmd_args.observed_prob,
                filter_latent=False,
                closed_world=False,
                filter_observed=1,
            )
            posterior_prob = posterior_model(flat_list, node_embeds)
            entropy = compute_entropy(posterior_prob) / cmd_args.entropy_temp
            entropy = entropy.to("cpu")
            posterior_prob = posterior_prob.to("cpu")

            potential = mln(
                batch_neg_mask,
                batch_latent_var_inds,
                observed_rule_cnts,
                posterior_prob,
                flat_list,
                batch_observed_vars,
            )

            optimizer.zero_grad()
            if rule_opt is not None:
                rule_opt.zero_grad()

            loss = -(potential + entropy) / cmd_args.batchsize
            loss.backward()
            optimizer.step()
            if rule_opt is not None:
                rule_opt.step()
                with torch.no_grad():
                    mln.rule_weights_lin.weight.clamp_(min=0.01, max=20.0)

        t1 = time.perf_counter()
        epoch_times.append(t1 - t0)

        if record_epoch_curve or (ep == num_epochs - 1):
            with torch.no_grad():
                node_embeds = gcn(dataset)
                q_samples = [(e[1], e[2]) for e in dataset.test_fact_ls]
                q_prob = posterior_model(q_samples, node_embeds).cpu().numpy()
                q_label = np.array([e[0] for e in dataset.test_fact_ls])
                q_preds = [e[1] for e in dataset.test_fact_ls]

                full_probs = posterior_model(all_test_triples, node_embeds).cpu().numpy()
                rec_class_probs = {
                    (all_test_triples[i][0], all_test_triples[i][1][0]): float(
                        full_probs[i]
                    )
                    for i in range(len(all_test_triples))
                }
                metrics = evaluate_predictions_detailed(
                    q_label, q_prob, q_preds, te_df, rec_class_probs
                )
                epoch_curve.append((ep + 1, metrics))

    final_weights = mln.rule_weights_lin.weight.detach().cpu().numpy().flatten().copy()

    deg_norms = {}
    if compute_degree_norms and valid_df is not None:
        with torch.no_grad():
            node_embeds = gcn(dataset).cpu().numpy()
        ev_counts = valid_df.groupby("ev")["record_id"].count().to_dict()
        rec_to_ev = dict(zip(valid_df["record_id"], valid_df["ev"]))

        buckets = {
            "Bậc nhỏ [1, 5]": (1, 5),
            "Bậc vừa [6, 20]": (6, 20),
            "Bậc lớn [21, 100]": (21, 100),
            "Bậc rất lớn (>100)": (101, 100000),
        }
        for bname, (lo, hi) in buckets.items():
            ev_norms = []
            rec_norms = []
            for ev_id, d in ev_counts.items():
                if lo <= d <= hi and ev_id in kg.ent2idx:
                    ev_norms.append(
                        float(np.linalg.norm(node_embeds[kg.ent2idx[ev_id]]))
                    )
            for rid, ev_id in rec_to_ev.items():
                d = ev_counts.get(ev_id, 0)
                if lo <= d <= hi and rid in kg.ent2idx:
                    rec_norms.append(
                        float(np.linalg.norm(node_embeds[kg.ent2idx[rid]]))
                    )
            deg_norms[bname] = {
                "event_hub_norm": float(np.mean(ev_norms)) if ev_norms else 0.0,
                "record_node_norm": float(np.mean(rec_norms)) if rec_norms else 0.0,
            }

    return {
        "final": epoch_curve[-1][1],
        "epoch_curve": epoch_curve,
        "sec_per_epoch": float(np.mean(epoch_times)),
        "init_weights": init_weights,
        "final_weights": final_weights,
        "rules": dataset.rule_ls,
        "deg_norms": deg_norms,
    }


def summarize_multi_seed(runs: list, label: str):
    keys = [
        "auc_roc",
        "auc_pr",
        "macro_f1",
        "f1_I",
        "f1_II",
        "f1_III",
        "q_pr_I",
        "q_pr_II",
        "q_pr_III",
        "ovr_pr_I",
        "ovr_pr_II",
        "ovr_pr_III",
        "log_prob",
    ]
    stats = {}
    for k in keys:
        vals = np.array([r["final"][k] for r in runs])
        stats[k] = (float(np.mean(vals)), float(np.std(vals)))
    sec_vals = np.array([r["sec_per_epoch"] for r in runs])
    stats["sec"] = (float(np.mean(sec_vals)), float(np.std(sec_vals)))

    print(
        f"* {label}:\n"
        f"  + AUC-ROC:     {stats['auc_roc'][0]:.4f} ± {stats['auc_roc'][1]:.4f}\n"
        f"  + AUC-PR:      {stats['auc_pr'][0]:.4f} ± {stats['auc_pr'][1]:.4f}\n"
        f"  + Macro-F1:    {stats['macro_f1'][0]:.4f} ± {stats['macro_f1'][1]:.4f} "
        f"(F1_I={stats['f1_I'][0]:.4f}, F1_II={stats['f1_II'][0]:.4f}, F1_III={stats['f1_III'][0]:.4f})\n"
        f"  + Per-Class AUC-PR (Query): Class_I={stats['q_pr_I'][0]:.4f}±{stats['q_pr_I'][1]:.4f} | "
        f"Class_II={stats['q_pr_II'][0]:.4f}±{stats['q_pr_II'][1]:.4f} | "
        f"Class_III={stats['q_pr_III'][0]:.4f}±{stats['q_pr_III'][1]:.4f}\n"
        f"  + Per-Class AUC-PR (OvR):   Class_I={stats['ovr_pr_I'][0]:.4f}±{stats['ovr_pr_I'][1]:.4f} | "
        f"Class_II={stats['ovr_pr_II'][0]:.4f}±{stats['ovr_pr_II'][1]:.4f} | "
        f"Class_III={stats['ovr_pr_III'][0]:.4f}±{stats['ovr_pr_III'][1]:.4f}\n"
        f"  + Log-Prob:    {stats['log_prob'][0]:.2f} ± {stats['log_prob'][1]:.2f} "
        f"(Time: {stats['sec'][0]:.2f}±{stats['sec'][1]:.2f} s/epoch)"
    )
    return stats


def main():
    root = Path(__file__).resolve().parent / "data"
    valid, tr_idx, te_idx = load_fda_split(root)
    te_df = valid.iloc[te_idx].copy().reset_index(drop=True)

    # 1. Graph statistics
    section_1_graph_statistics(valid)

    # 2. LR baselines & build ablation datasets
    ablation_dirs = section_2_lr_baselines_and_build_ablation_datasets(
        root, valid, tr_idx, te_idx
    )

    seeds = [10, 20, 30]
    v4_dir = root / "food_v4_unary"

    # 3. 1x2 Table on food_v4_unary: -trans 1 vs -trans 0 (3 seeds)
    print("\n" + "=" * 88)
    print("PHẦN 3: BẢNG 1x2 TRÊN food_v4_unary (Transductive -trans 1 vs Inductive -trans 0, 3 SEEDS)")
    print("=" * 88)
    runs_trans1 = [
        run_single_expressgnn(
            data_root=v4_dir,
            te_df=te_df,
            trans=1,
            num_hops=2,
            rule_weights_learning=0,
            seed=s,
            num_epochs=6,
            num_batches=40,
            batchsize=128,
            lr=0.0005,
        )
        for s in seeds
    ]
    summarize_multi_seed(runs_trans1, "food_v4_unary | Transductive (-trans 1)")

    runs_trans0_hops2 = [
        run_single_expressgnn(
            data_root=v4_dir,
            te_df=te_df,
            trans=0,
            num_hops=2,
            rule_weights_learning=0,
            seed=s,
            num_epochs=6,
            num_batches=40,
            batchsize=128,
            lr=0.0005,
            record_epoch_curve=(s == seeds[0]),
            compute_degree_norms=(s == seeds[0]),
            valid_df=valid,
        )
        for s in seeds
    ]
    summarize_multi_seed(runs_trans0_hops2, "food_v4_unary | Inductive (-trans 0)")

    # 4. -num_hops in {1, 2, 3} on food_v4_unary
    print("\n" + "=" * 88)
    print("PHẦN 4: KHẢO SÁT BÁN KÍNH TRUYỀN TIN -num_hops ∈ {1, 2, 3} TRÊN food_v4_unary (3 SEEDS)")
    print("=" * 88)
    for hops in [1, 2, 3]:
        if hops == 2:
            runs = runs_trans0_hops2
        else:
            runs = [
                run_single_expressgnn(
                    data_root=v4_dir,
                    te_df=te_df,
                    trans=0,
                    num_hops=hops,
                    rule_weights_learning=0,
                    seed=s,
                    num_epochs=6,
                    num_batches=40,
                    batchsize=128,
                    lr=0.0005,
                    compute_degree_norms=(s == seeds[0]),
                    valid_df=valid,
                )
                for s in seeds
            ]
        summarize_multi_seed(runs, f"food_v4_unary | -num_hops {hops}")
        print(f"  + Norm trung bình ||h^({hops})||_2 theo nhóm bậc tâm sự kiện (Seed {seeds[0]}):")
        for bname, norms in runs[0]["deg_norms"].items():
            print(
                f"    * {bname:20s}: Norm nút tâm event_id = {norms['event_hub_norm']:8.2f} | "
                f"Norm nút bản ghi record_id = {norms['record_node_norm']:8.2f}"
            )

    # 5. -rule_weights_learning 0 vs 1 on food_v4_unary
    print("\n" + "=" * 88)
    print("PHẦN 5: SO SÁNH -rule_weights_learning 0 vs 1 TRÊN food_v4_unary")
    print("=" * 88)
    res_rw0 = runs_trans0_hops2[0]
    res_rw1 = run_single_expressgnn(
        data_root=v4_dir,
        te_df=te_df,
        trans=0,
        num_hops=2,
        rule_weights_learning=1,
        seed=seeds[0],
        num_epochs=6,
        num_batches=40,
        batchsize=128,
        lr=0.0005,
        lr_rule_weights=0.005,
        record_epoch_curve=True,
    )
    print("\n[5.1] Đường cong hội tụ theo từng Epoch (AUC-ROC / AUC-PR / Macro-F1):")
    for ep_idx in range(len(res_rw0["epoch_curve"])):
        ep_num, m0 = res_rw0["epoch_curve"][ep_idx]
        _, m1 = res_rw1["epoch_curve"][ep_idx]
        print(
            f"  + Epoch {ep_num}: "
            f"[rw=0 Cố định] AUC-ROC={m0['auc_roc']:.4f}, AUC-PR={m0['auc_pr']:.4f}, Macro-F1={m0['macro_f1']:.4f} | "
            f"[rw=1 Học M-step] AUC-ROC={m1['auc_roc']:.4f}, AUC-PR={m1['auc_pr']:.4f}, Macro-F1={m1['macro_f1']:.4f}"
        )

    print("\n[5.2] Biến thiên trọng số của 6 luật tiêu biểu (Lúc đầu w^(0) vs Lúc cuối w^(T)):")
    rule_lines = (v4_dir / "rules").read_text().splitlines()
    target_patterns = [
        "!tier_class_I_19(x) v class_I(x)",
        "!tier_class_I_0(x) v !class_I(x)",
        "!tier_class_II_19(x) v class_II(x)",
        "!tier_class_II_0(x) v !class_II(x)",
        "!tier_class_III_18(x) v class_III(x)",
        "!tier_class_III_0(x) v !class_III(x)",
    ]
    for pat in target_patterns:
        for r_idx, rline in enumerate(rule_lines):
            if pat in rline:
                w0 = res_rw1["init_weights"][r_idx]
                wT_fixed = res_rw0["final_weights"][r_idx]
                wT_learned = res_rw1["final_weights"][r_idx]
                print(
                    f"  + Luật [{r_idx:2d}] {pat:40s}: "
                    f"w^(0) = {w0:.3f} | rw=0 w^(T) = {wT_fixed:.3f} | "
                    f"rw=1 w^(T) = {wT_learned:.3f} (Δ = {wT_learned - w0:+.3f})"
                )
                break

    # 6. Ablation Study on food_v4_unary (3 seeds)
    print("\n" + "=" * 88)
    print("PHẦN 6: ABLATION STUDY TRÊN food_v4_unary (3 SEEDS, MEAN ± STD)")
    print("=" * 88)
    summarize_multi_seed(
        runs_trans0_hops2,
        "Full food_v4_unary (Đầy đủ: Event Pool + Deg=5 + 20-Tier)",
    )
    ablation_configs = [
        ("Ablation 1: v4 bỏ gộp theo event (w/o Event Pooling)", ablation_dirs["no_event_pool"]),
        ("Ablation 2: v4 có thêm cạnh nhãn (deg 8 so với 5)", ablation_dirs["with_deg_shift"]),
        ("Ablation 3a: v4 với 10 bậc lượng tử (10-Tier Quantization)", ablation_dirs["coarse_10tier"]),
        ("Ablation 3b: v4 với 5 bậc lượng tử (5-Tier Quantization)", ablation_dirs["coarse_5tier"]),
    ]
    for ab_label, ab_dir in ablation_configs:
        runs = [
            run_single_expressgnn(
                data_root=ab_dir,
                te_df=te_df,
                trans=0,
                num_hops=2,
                rule_weights_learning=0,
                seed=s,
                num_epochs=6,
                num_batches=40,
                batchsize=128,
                lr=0.0005,
            )
            for s in seeds
        ]
        summarize_multi_seed(runs, ab_label)


if __name__ == "__main__":
    main()

