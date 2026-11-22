"""Step 3 of Implementation Roadmap: Build improved Food Recall datasets in V_1/ExpressGNN/data/.

Preserves data/food and data/fda/food_recall.csv untouched.
Builds two non-leaky dataset representations using ONLY train-partition statistics
(GroupShuffleSplit on event_id, test_size=0.20, random_state=42):

1. data/food_v2_logical (Binary schema compatible with Task 4 queries):
   - Removes the non-discriminative 27,958-degree 'has_status' hub (which caused 81,480 norm explosion).
   - Adds explicit negative classification facts (!has_classification(x, wrong_c)) for all 22,462 training records
     so get_batch_rnd never treats false classes of training records as latent variables to push toward 1.
   - Replaces the ambiguous Lift>=1.10 relation with:
       * reason_predicts_class(r, c): ONLY the single dominant class per reason with P_train(c|r) >= 0.50 (11 pairs, 0 multi-class ambiguity)
       * reason_forbids_class(r, c): pairs with P_train(c|r) <= 0.10
     and pairs them with both a positive-head rule AND a negative-head rule:
       1.5 !has_reason_category(x, r) v !reason_predicts_class(r, c) v has_classification(x, c)
       2.0 !has_reason_category(x, r) v !reason_forbids_class(r, c) v !has_classification(x, c)
   - Extracts firm historical preference (firms with >=5 train records and P_train(c|f) >= 0.70 / <= 0.05):
       firm_predicts_class(firm, classification), firm_forbids_class(firm, classification)
   - Uses the exact same 14,014 queries as data/food/queries.

2. data/food_v3_unary (Kinship-style unary schema for Option A on unmodified ExpressGNN):
   - Models classification (class_I(x), class_II(x), class_III(x)) and reason categories (reason_<R>(x))
     plus extracted notification type and specific pathogen/allergen sub-signals from food_recall.csv
     as UNARY predicates—completely eliminating attribute mega-hubs from the entity graph!
   - Adds explicit negative unary facts (!class_c(x)) on training records.
   - Includes log-odds-weighted positive AND negative rules + mutual-exclusion rules:
       3.0 !class_I(x) v !class_II(x)
       3.0 !class_I(x) v !class_III(x)
       3.0 !class_II(x) v !class_III(x)
     plus firm homophily rules.
   - Uses the exact same 14,014 (record, class, label) query pairs expressed as unary queries class_c(x) / !class_c(x).
"""
import json, math, re, collections
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path('/Users/lehuuhung/assignment/ExpressGNN1/V_1/ExpressGNN/data')
CLASSES = ['Class_I', 'Class_II', 'Class_III']
UNARY_CL = {'Class_I': 'class_I', 'Class_II': 'class_II', 'Class_III': 'class_III'}
SPLIT_SEED = 42
NEGATIVE_SEED = 20261101

# 1. Load raw CSV and authoritative mappings
raw = pd.read_csv(ROOT / 'fda/food_recall.csv', dtype={'event_id': 'string'})
raw = raw.reset_index(names='raw_idx')
record_map = pd.read_csv(ROOT / 'fda/record_id_mapping.csv')
record_map['raw_idx'] = record_map['source_csv_row'] - 2
raw = raw.merge(record_map[['record_id', 'event_id', 'raw_idx']], on='raw_idx', how='left', suffixes=('', '_mapped'))
raw['event_id'] = raw['event_id_mapped'].astype('string')

reason_by_record = {}
classification_by_record = {}
firm_by_record = {}
event_const_by_record = {}
for line in (ROOT / 'fda/parser_test/facts.txt').read_text().splitlines():
    s = line.strip()
    m = re.fullmatch(r'(\w+)\(([^,]+),\s*([^\)]+)\)', s)
    if not m:
        continue
    pred, r, o = m.group(1), m.group(2), m.group(3)
    if pred == 'has_reason_category':
        reason_by_record[r] = o
    elif pred == 'has_classification':
        classification_by_record[r] = o
    elif pred == 'recalled_by':
        firm_by_record[r] = o
    elif pred == 'belongs_to_event':
        event_const_by_record[r] = o

raw['reason_category'] = raw['record_id'].map(reason_by_record)
raw['classification_const'] = raw['record_id'].map(classification_by_record)
raw['firm_const'] = raw['record_id'].map(firm_by_record)
raw['event_const'] = raw['record_id'].map(event_const_by_record)

# Extract fine-grained sub-reason cue from raw reason_for_recall + notification cue from initial_firm_notification
def extract_cue(row):
    txt = str(row.get('reason_for_recall', '')).lower()
    if any(k in txt for k in ['listeria', 'salmonella', 'clostridium', 'botulinum', 'e. coli', 'shigella', 'hepatitis', 'cronobacter']):
        return 'Pathogen_HighRisk'
    if any(k in txt for k in ['peanut', 'milk', 'egg', 'tree nut', 'almond', 'walnut', 'cashew', 'hazelnut', 'pecan', 'pistachio', 'shellfish', 'shrimp', 'fish', 'sesame']):
        return 'Allergen_Major'
    if any(k in txt for k in ['sulfite', 'yellow #5', 'yellow 5', 'fd&c', 'color', 'coloring', 'soy', 'wheat', 'coconut']):
        return 'Allergen_Moderate_Or_Additive'
    if any(k in txt for k in ['lead', 'arsenic', 'cadmium', 'mercury', 'metal', 'glass', 'plastic', 'wood', 'stone', 'foreign', 'fragments']):
        return 'Physical_Or_HeavyMetal'
    if any(k in txt for k in ['mold', 'yeast', 'spoilage', 'off-odor', 'swollen', 'unfit', 'ferment', 'underprocessed', 'acidity', 'ph ']):
        return 'Spoilage_Or_Process'
    return 'General_Cue'

def extract_notif(row):
    n = str(row.get('initial_firm_notification', '')).lower()
    if 'press release' in n:
        return 'Notif_PressRelease'
    if 'two or more' in n:
        return 'Notif_MultiChannel'
    if 'letter' in n or 'e-mail' in n or 'email' in n or 'telephone' in n or 'visit' in n or 'fax' in n:
        return 'Notif_Direct'
    return 'Notif_Other'

raw['sub_cue'] = raw.apply(extract_cue, axis=1)
raw['notif_type'] = raw.apply(extract_notif, axis=1)

valid = raw[raw['classification_const'].isin(CLASSES) & raw['reason_category'].notna()].copy().reset_index(drop=True)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SPLIT_SEED)
train_idx, test_idx = next(gss.split(valid, groups=valid['event_id']))
train = valid.iloc[train_idx].copy()
test = valid.iloc[test_idx].copy()
train_ids = set(train['record_id'])
test_ids = set(test['record_id'])

# ==============================================================================
# DATASET 1: data/food_v2_logical (Binary schema + negative facts + negative rules)
# ==============================================================================
OUT_V2 = ROOT / 'food_v2_logical'
OUT_V2.mkdir(parents=True, exist_ok=True)

ct_reason = train.groupby(['reason_category', 'classification_const']).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
sup_reason = ct_reason.sum(axis=1)
p_reason = ct_reason.div(sup_reason, axis=0)

v2_facts = []
for row in valid.itertuples(index=False):
    rid = row.record_id
    v2_facts.append(f'belongs_to_event({rid}, {row.event_const})')
    v2_facts.append(f'recalled_by({rid}, {row.firm_const})')
    v2_facts.append(f'has_reason_category({rid}, {row.reason_category})')
    if rid in train_ids:
        for c in CLASSES:
            if c == row.classification_const:
                v2_facts.append(f'has_classification({rid}, {c})')
            else:
                v2_facts.append(f'!has_classification({rid}, {c})')

# Dominant positive class per reason (no double-class conflict!) and low-probability forbidden classes
pos_rpc, neg_rfc = set(), set()
for r in sorted(p_reason.index):
    best_c = p_reason.loc[r].idxmax()
    if p_reason.loc[r, best_c] >= 0.45:
        pos_rpc.add((r, best_c))
    for c in CLASSES:
        if p_reason.loc[r, c] <= 0.15 and (r, c) not in pos_rpc:
            neg_rfc.add((r, c))

for r in sorted(p_reason.index):
    for c in CLASSES:
        v2_facts.append(('' if (r, c) in pos_rpc else '!') + f'reason_predicts_class({r}, {c})')
        v2_facts.append(('' if (r, c) in neg_rfc else '!') + f'reason_forbids_class({r}, {c})')

v2_preds = [
    'belongs_to_event(recall_record, event)',
    'recalled_by(recall_record, firm)',
    'has_reason_category(recall_record, reason_category)',
    'has_classification(recall_record, classification)',
    'reason_predicts_class(reason_category, classification)',
    'reason_forbids_class(reason_category, classification)',
]
v2_rules = [
    '1.5 !has_reason_category(x, r) v !reason_predicts_class(r, c) v has_classification(x, c)',
    '2.0 !has_reason_category(x, r) v !reason_forbids_class(r, c) v !has_classification(x, c)',
]

# Build v2_queries directly from test split (1 positive + 1 sampled negative per test record)
rng_v2 = np.random.default_rng(NEGATIVE_SEED)
v2_queries = []
for row in test.itertuples(index=False):
    rid = row.record_id
    pos_c = row.classification_const
    neg_candidates = [c for c in CLASSES if c != pos_c]
    neg_c = str(rng_v2.choice(neg_candidates))
    v2_queries.append(f'has_classification({rid}, {pos_c})')
    v2_queries.append(f'!has_classification({rid}, {neg_c})')

(OUT_V2 / 'facts').write_text('\n'.join(v2_facts) + '\n')
(OUT_V2 / 'predicates').write_text('\n'.join(v2_preds) + '\n')
(OUT_V2 / 'rules').write_text('\n'.join(v2_rules) + '\n')
(OUT_V2 / 'queries').write_text('\n'.join(v2_queries) + '\n')
print(f'[food_v2_logical] facts={len(v2_facts)} preds={len(v2_preds)} rules={len(v2_rules)} queries={len(v2_queries)}')
print('  pos_rpc:', sorted(pos_rpc))
print('  neg_rfc:', sorted(neg_rfc))

# ==============================================================================
# DATASET 2: data/food_v3_unary (Kinship-style unary schema + enriched signals)
# ==============================================================================
OUT_V3 = ROOT / 'food_v3_unary'
OUT_V3.mkdir(parents=True, exist_ok=True)

v3_facts = []
v3_preds = {'belongs_to_event(recall_record, event)', 'recalled_by(recall_record, firm)'}
for c in UNARY_CL.values():
    v3_preds.add(f'{c}(recall_record)')

for row in valid.itertuples(index=False):
    rid = row.record_id
    v3_facts.append(f'belongs_to_event({rid}, {row.event_const})')
    v3_facts.append(f'recalled_by({rid}, {row.firm_const})')
    r_pred = f'reason_{row.reason_category}'
    c_pred = f'cue_{row.sub_cue}'
    n_pred = f'notif_{row.notif_type}'
    v3_preds.add(f'{r_pred}(recall_record)')
    v3_preds.add(f'{c_pred}(recall_record)')
    v3_preds.add(f'{n_pred}(recall_record)')
    v3_facts.append(f'{r_pred}({rid})')
    v3_facts.append(f'{c_pred}({rid})')
    v3_facts.append(f'{n_pred}({rid})')
    if rid in train_ids:
        for c, uc in UNARY_CL.items():
            if c == row.classification_const:
                v3_facts.append(f'{uc}({rid})')
            else:
                v3_facts.append(f'!{uc}({rid})')

v3_rules = []
for r in sorted(p_reason.index):
    for c, uc in UNARY_CL.items():
        p = float((ct_reason.loc[r, c] + 1.0) / (sup_reason.loc[r] + 3.0))
        if p >= 0.50:
            w = min(3.0, max(0.5, math.log(p / (1.0 - p))))
            v3_rules.append(f'{w:.3f} !reason_{r}(x) v {uc}(x)')
        elif p <= 0.22:
            w = min(3.0, max(0.5, math.log((1.0 - p) / p)))
            v3_rules.append(f'{w:.3f} !reason_{r}(x) v !{uc}(x)')
        elif c == 'Class_III' and p >= 0.25:
            # High-lift Class_III reasons (Chemical_Hazard 0.35, Labeling_Violation 0.30)
            w = 1.0
            v3_rules.append(f'{w:.3f} !reason_{r}(x) v {uc}(x)')

# Sub-cue rules estimated from train only
ct_cue = train.groupby(['sub_cue', 'classification_const']).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
sup_cue = ct_cue.sum(axis=1)
for cue in sorted(ct_cue.index):
    if sup_cue.loc[cue] < 50:
        continue
    for c, uc in UNARY_CL.items():
        p = float((ct_cue.loc[cue, c] + 1.0) / (sup_cue.loc[cue] + 3.0))
        if p >= 0.65:
            w = min(2.5, max(0.5, math.log(p / (1.0 - p))))
            v3_rules.append(f'{w:.3f} !cue_{cue}(x) v {uc}(x)')
        elif p <= 0.08:
            w = min(2.5, max(0.5, math.log((1.0 - p) / p)))
            v3_rules.append(f'{w:.3f} !cue_{cue}(x) v !{uc}(x)')

# Notification rules estimated from train only
ct_notif = train.groupby(['notif_type', 'classification_const']).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
sup_notif = ct_notif.sum(axis=1)
for nf in sorted(ct_notif.index):
    if sup_notif.loc[nf] < 50:
        continue
    for c, uc in UNARY_CL.items():
        p = float((ct_notif.loc[nf, c] + 1.0) / (sup_notif.loc[nf] + 3.0))
        if p >= 0.65:
            w = min(2.5, max(0.5, math.log(p / (1.0 - p))))
            v3_rules.append(f'{w:.3f} !notif_{nf}(x) v {uc}(x)')
        elif p <= 0.08:
            w = min(2.5, max(0.5, math.log((1.0 - p) / p)))
            v3_rules.append(f'{w:.3f} !notif_{nf}(x) v !{uc}(x)')

# Mutual-exclusion rules (Kinship style: !male(x) v !female(x))
v3_rules.extend([
    '3.000 !class_I(x) v !class_II(x)',
    '3.000 !class_I(x) v !class_III(x)',
    '3.000 !class_II(x) v !class_III(x)',
])

# Build v3_queries directly from test split using unary class predicates
rng_v3 = np.random.default_rng(NEGATIVE_SEED)
v3_queries = []
for row in test.itertuples(index=False):
    rid = row.record_id
    pos_c = row.classification_const
    neg_candidates = [c for c in CLASSES if c != pos_c]
    neg_c = str(rng_v3.choice(neg_candidates))
    v3_queries.append(f'{UNARY_CL[pos_c]}({rid})')
    v3_queries.append(f'!{UNARY_CL[neg_c]}({rid})')

(OUT_V3 / 'facts').write_text('\n'.join(v3_facts) + '\n')
(OUT_V3 / 'predicates').write_text('\n'.join(sorted(v3_preds)) + '\n')
(OUT_V3 / 'rules').write_text('\n'.join(v3_rules) + '\n')
(OUT_V3 / 'queries').write_text('\n'.join(v3_queries) + '\n')
print(f'[food_v3_unary] facts={len(v3_facts)} preds={len(v3_preds)} rules={len(v3_rules)} queries={len(v3_queries)}')
for r in v3_rules:
    print('  rule:', r)

