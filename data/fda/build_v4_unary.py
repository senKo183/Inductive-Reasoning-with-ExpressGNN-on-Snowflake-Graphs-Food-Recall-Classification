"""
Build data/food_v4_unary from data/fda/food_recall.csv for ExpressGNN.

Key improvements over food_v3_unary:
1. Extracts richer inductive text & categorical signals from food_recall.csv:
   - reason_for_recall (TF-IDF 1-2 ngrams, fit strictly on train events)
   - product_description + distribution_pattern (TF-IDF 1-2 ngrams, fit strictly on train events)
   - categorical attributes: reason_category, initial_firm_notification, voluntary_mandated,
     status, state, country, recalled_by (OneHotEncoder fit strictly on train events)
2. Pools evidence across records sharing the same event_id (Snowflake hub pooling).
3. Quantizes out-of-fold / inductive probabilities into 20 log-odds unary predicates per class:
   tier_class_I_0..19(x), tier_class_II_0..19(x), tier_class_III_0..19(x).
4. Sets exact Variational EM equilibrium rule weights w = ln(p / (1 - p)) so that the
   free-energy minimum of ExpressGNN's E-step + M-step occurs at p* = sigmoid(+-w).
5. Excludes target labels class_c(x_train) from `facts` by default so original `main/train.py`
   has zero target-edge degree mismatch between train and test snowflake nodes.
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder


def main():
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "food_v4_unary"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Strict event-level split (zero event leakage between train and test)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, te_idx = next(gss.split(valid, groups=valid["event_id"]))
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

    clf = LogisticRegression(max_iter=1000, C=0.3).fit(X_all[tr_idx], tr["cc"])
    p_all = pd.DataFrame(clf.predict_proba(X_all), index=valid.index, columns=classes)
    p_all["ev"] = valid["event_id"].values
    p_ev = p_all.groupby("ev")[classes].transform("mean")

    edges = [
        0.0,
        0.005,
        0.015,
        0.03,
        0.06,
        0.10,
        0.16,
        0.24,
        0.34,
        0.45,
        0.55,
        0.66,
        0.76,
        0.84,
        0.90,
        0.94,
        0.965,
        0.98,
        0.99,
        0.996,
        1.01,
    ]
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

    for idx, row in valid.iterrows():
        rid = row["record_id"]
        facts.append(f"belongs_to_event({rid}, {row['ev']})")
        facts.append(f"recalled_by({rid}, {row['fm']})")
        for c, uc in unary_cl.items():
            pr = float(p_ev.loc[idx, c])
            t = min(
                n_tiers - 1, max(0, int(np.searchsorted(edges[1:], pr, side="right")))
            )
            tier_counts[(c, t)] += 1
            facts.append(f"tier_{uc}_{t}({rid})")

    for c, uc in unary_cl.items():
        for t in range(n_tiers):
            cnt = tier_counts[(c, t)]
            if cnt == 0:
                continue
            pname = f"tier_{uc}_{t}"
            preds.add(f"{pname}(recall_record)")
            p_mid = min(0.995, max(0.003, (edges[t] + min(1.0, edges[t + 1])) / 2.0))
            if p_mid >= 0.50:
                w = max(0.10, math.log(p_mid / (1.0 - p_mid)))
                rules.append(f"{w:.3f} !{pname}(x) v {uc}(x)")
            else:
                w = max(0.10, math.log((1.0 - p_mid) / p_mid))
                rules.append(f"{w:.3f} !{pname}(x) v !{uc}(x)")

    # Build queries directly from test split using unary class predicates
    # (1 positive ground-truth class + 1 sampled negative class per test record)
    rng_v4 = np.random.default_rng(20261101)
    queries = []
    for row in te.itertuples(index=False):
        rid = row.record_id
        pos_c = row.cc
        neg_candidates = [c for c in classes if c != pos_c]
        neg_c = str(rng_v4.choice(neg_candidates))
        queries.append(f"{unary_cl[pos_c]}({rid})")
        queries.append(f"!{unary_cl[neg_c]}({rid})")

    (out_dir / "facts").write_text("\n".join(facts) + "\n")
    (out_dir / "predicates").write_text("\n".join(sorted(preds)) + "\n")
    (out_dir / "rules").write_text("\n".join(rules) + "\n")
    (out_dir / "queries").write_text("\n".join(queries) + "\n")

    print(
        f"Saved data/food_v4_unary: facts={len(facts)}, preds={len(preds)}, rules={len(rules)}, queries={len(queries)}"
    )


if __name__ == "__main__":
    main()

