"""Build an ExpressGNN FDA dataset for Reason Category -> Classification.

Run from a workspace containing:
  files/food_recall(1).csv
  task2_food_recall/record_id_mapping.csv
  task2_food_recall/facts.txt

The split is event-level (GroupShuffleSplit, test_size=0.20, random_state=42).
Classification labels for every record in held-out events are removed from facts.
Queries contain one positive and one sampled negative classification per test record.
A derived predicate reason_predicts_class is estimated from TRAIN events only.
"""
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path('/Users/lehuuhung/assignment/ExpressGNN1/ExpressGNN/data')
OUT = ROOT / 'task4_food_recall_reason_classification'
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ['Class_I', 'Class_II', 'Class_III']
CLASS_SPACES = {c: c.replace('_', ' ') for c in CLASSES}
MIN_PAIR_SUPPORT = 50
MIN_LIFT = 1.10
SPLIT_SEED = 42
NEGATIVE_SEED = 20261101

# Load raw records and align them to record IDs used in task2 facts.
raw = pd.read_csv(ROOT / 'fda/food_recall.csv', dtype={'event_id': 'string'})
raw = raw.reset_index(names='raw_idx')
record_map = pd.read_csv(ROOT / 'fda/record_id_mapping.csv')
record_map['raw_idx'] = record_map['source_csv_row'] - 2
raw = raw.merge(record_map[['record_id', 'event_id', 'raw_idx']], on='raw_idx', how='left', suffixes=('', '_mapped'))
# The mapping file is authoritative for record_id/event_id.
raw['event_id'] = raw['event_id_mapped'].astype('string')

reason_by_record = {}
classification_by_record = {}
for line in (ROOT / 'fda/parser_test/facts.txt').read_text().splitlines():
    s = line.strip()
    m = re.fullmatch(r'has_reason_category\(([^,]+),\s*([^\)]+)\)', s)
    if m:
        reason_by_record[m.group(1)] = m.group(2)
    m = re.fullmatch(r'has_classification\(([^,]+),\s*([^\)]+)\)', s)
    if m:
        classification_by_record[m.group(1)] = m.group(2)

raw['reason_category'] = raw['record_id'].map(reason_by_record)
raw['classification_const'] = raw['record_id'].map(classification_by_record)
raw['classification'] = raw['classification'].astype(str).str.strip()
# Keep only the three target classes for supervised task and rule estimation.
valid = raw[raw['classification_const'].isin(CLASSES) & raw['reason_category'].notna()].copy()
valid = valid.reset_index(drop=True)

# Event-level split: no event represented in both train and test.
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SPLIT_SEED)
train_idx, test_idx = next(gss.split(valid, groups=valid['event_id']))
train = valid.iloc[train_idx].copy()
test = valid.iloc[test_idx].copy()
train_ids = set(train['record_id'])
test_ids = set(test['record_id'])
test_events = set(test['event_id'].astype(str))

# Learn the reason-category/class associations using TRAIN records only.
counts = (train.groupby(['reason_category', 'classification_const']).size()
          .unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0))
counts['support'] = counts[CLASSES].sum(axis=1)
priors = train['classification_const'].value_counts(normalize=True).reindex(CLASSES, fill_value=0)
selected_pairs = []
for reason in counts.index:
    support = int(counts.loc[reason, 'support'])
    if support < MIN_PAIR_SUPPORT:
        continue
    for cls in CLASSES:
        p_cond = float(counts.loc[reason, cls] / support)
        lift = p_cond / float(priors[cls]) if priors[cls] > 0 else 0.0
        if lift >= MIN_LIFT:
            selected_pairs.append({
                'reason_category': reason,
                'classification': cls,
                'support_train': support,
                'class_count_train': int(counts.loc[reason, cls]),
                'conditional_probability_train': p_cond,
                'class_prior_train': float(priors[cls]),
                'lift_train': lift,
            })
selected_df = pd.DataFrame(selected_pairs).sort_values(['reason_category', 'lift_train'], ascending=[True, False])
selected_set = {(r['reason_category'], r['classification']) for r in selected_pairs}

# Build a closed-world binary relation for all 11 x 3 possible reason/class pairs.
# Positive = training lift >= threshold; negative = all other pairs.
meta_facts = []
all_reasons = sorted(valid['reason_category'].dropna().unique())
for reason in all_reasons:
    for cls in CLASSES:
        val = (reason, cls) in selected_set
        meta_facts.append(('' if val else '!') + f'reason_predicts_class({reason}, {cls})')

# Preserve the original observed facts, but remove classification labels from:
# (a) every record in held-out events; (b) the two Not Yet Classified records.
base_lines = (ROOT / 'fda/parser_test/facts.txt').read_text().splitlines()
not_target_ids = set(raw.loc[~raw['classification_const'].isin(CLASSES), 'record_id'].dropna())
new_fact_lines = []
removed_classification = 0
for line in base_lines:
    s = line.strip()
    m = re.fullmatch(r'has_classification\(([^,]+),\s*([^\)]+)\)', s)
    if m:
        rid = m.group(1)
        if rid in test_ids or rid in not_target_ids:
            removed_classification += 1
            continue
    if s:
        new_fact_lines.append(s)
new_fact_lines.extend(meta_facts)

# Queries: one positive true class and one reproducibly sampled negative class per test record.
rng = np.random.default_rng(NEGATIVE_SEED)
query_lines = []
for row in test.sort_values('record_id').itertuples(index=False):
    rid = row.record_id
    true_cls = row.classification_const
    query_lines.append(f'has_classification({rid}, {true_cls})')
    wrong = [c for c in CLASSES if c != true_cls]
    negative_cls = str(rng.choice(wrong))
    query_lines.append(f'!has_classification({rid}, {negative_cls})')

# Predicate for derived train-only statistical relation.
predicate_lines = [
    'belongs_to_event(recall_record, event)',
    'recalled_by(recall_record, firm)',
    'has_reason_category(recall_record, reason_category)',
    'has_classification(recall_record, classification)',
    'has_status(recall_record, status)',
    'reason_predicts_class(reason_category, classification)',
]

# One general rule; its relation facts encode associations estimated from train only.
rules = [
    '1.0 !has_reason_category(x, r) v !reason_predicts_class(r, c) v has_classification(x, c)'
]

(OUT / 'facts').write_text('\n'.join(new_fact_lines) + '\n')
(OUT / 'predicates').write_text('\n'.join(predicate_lines) + '\n')
(OUT / 'rules').write_text('\n'.join(rules) + '\n')
(OUT / 'queries').write_text('\n'.join(query_lines) + '\n')
selected_df.to_csv(OUT / 'reason_predicts_class_train_only.csv', index=False)
counts.reset_index().to_csv(OUT / 'train_reason_classification_crosstab.csv', index=False)

# Metadata makes the split and rule construction auditable.
meta = {
    'task': 'predict has_classification(record, class) using reason category associations',
    'split_unit': 'event_id', 'test_size': 0.20, 'split_seed': SPLIT_SEED,
    'train_records': int(len(train)), 'test_records': int(len(test)),
    'train_events': int(train['event_id'].nunique()), 'test_events': int(test['event_id'].nunique()),
    'train_record_ids': len(train_ids), 'test_record_ids': len(test_ids),
    'query_positive': int(len(test)), 'query_negative': int(len(test)), 'query_total': int(len(query_lines)),
    'facts_total': int(len(new_fact_lines)), 'base_facts_total': int(len(base_lines)),
    'classification_facts_removed': int(removed_classification),
    'derived_relation_facts': int(len(meta_facts)),
    'derived_relation_positive': int(len(selected_set)),
    'derived_relation_negative': int(len(meta_facts) - len(selected_set)),
    'rule': rules[0], 'min_train_pair_support': MIN_PAIR_SUPPORT,
    'min_train_lift': MIN_LIFT, 'negative_sampling_seed': NEGATIVE_SEED,
    'note': 'belongs_to_event facts are retained as context; the final rule no longer uses event to infer classification. Held-out event classification facts are removed from facts.'
}
(OUT / 'split_metadata.json').write_text(json.dumps(meta, indent=2))

print(json.dumps(meta, indent=2))
print('\nSelected train-only reason/class associations:')
print(selected_df.to_string(index=False))
