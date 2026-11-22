# Inductive Reasoning with ExpressGNN on Snowflake Graphs: Food Recall Classification

> **Research Scope & Domain Clarification:** This repository adapts **ExpressGNN** (originally proposed in *"Efficient Probabilistic Logic Reasoning with Graph Neural Networks"*) to a new real-world domain: **U.S. FDA Food Recall severity classification**. Unlike tightly connected benchmark knowledge graphs (such as Kinship or FB15K-237), tabular food recall records decompose into thousands of disconnected hub-and-spoke (**"snowflake"**) subgraphs with zero entity overlap between training and test splits. While default Transductive ExpressGNN degrades to near-random performance on unseen snowflakes, this work demonstrates that **ExpressGNN generalizes effectively to snowflake knowledge graphs** when paired with **Inductive mode**, a **unary-predicate schema**, and **empirical log-odds rule weights**.

---

## Paper Summary

* **Title:** *Inductive Reasoning with ExpressGNN on Snowflake Graphs: Food Recall Classification* (*Suy luận quy nạp với ExpressGNN trên đồ thị bông tuyết: Phân loại thu hồi thực phẩm*)
* **Authors:** Le Huu Hung, Dang Tuan Minh, and Tran Nhat Hoang Anh (Industrial University of Ho Chi Minh City)

### Problem & Snowflake Graph Topology
Classifying U.S. FDA food recalls into **Class I** (severe health hazard or death, 44.0%), **Class II** (temporary/reversible health effects, 50.0%), and **Class III** (unlikely to cause adverse health effects, 6.0%) requires both predictive accuracy and interpretable logical rules. When converted into a knowledge graph (`food_v4_unary`), the dataset comprises **42,882 entities** (29,469 recall records, 7,896 recall events, and 5,517 firms) fragmented into **5,517 isolated snowflake components** with 0 triangles and **0% test entities seen during training**.

### Key Architectural Solutions
1. **Unary-Predicate Schema (Mega-Hub Elimination):** Categorical attributes are modeled as unary leaf predicates `a(x)` rather than binary relations linked to shared constant nodes. This eliminates high-degree attribute mega-hubs, prevents activation saturation and over-smoothing, and maintains a uniform node degree across train and test sets.
2. **Empirical Log-Odds Rule Weights:** Event-pooled probabilities from an L2-regularized logistic regression model are quantized into probability tiers (`tier_c_t(x)`). Rule weights are analytically set to the empirical log-odds (`w_t = ln(p_mid / (1 - p_mid))` for positive clauses and `ln((1 - p_mid) / p_mid)` for negative clauses), proven to be the unique global maximum of the variational objective (ELBO) per ground clause, and kept fixed to prevent M-step weight drift.
3. **Inductive Mode & Snowflake Isomorphism Invariance:** Removing entity-specific One-Hot identifiers (`-trans 0`, `-gcn_free_size 0`) reduces the input space to relation/entity types (`1 + N_R`), guaranteeing identical hidden representations for isomorphic `K`-hop snowflake motifs across unseen test clusters.

### Summary of Experimental Results

| Model / Configuration | Mode | AUC-ROC | AUC-PR | Macro-F1 | Class III AUC-PR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ExpressGNN (`K = 2`) | Transductive (`-trans 1`) | 0.664 ± 0.026 | 0.616 ± 0.016 | 0.308 | 0.063 |
| ExpressGNN (`K = 2`) | **Inductive (`-trans 0`)** | 0.892 ± 0.012 | 0.881 ± 0.009 | 0.533 | 0.140 |
| ExpressGNN (`K = 1`) | **Inductive (`-trans 0`)** | **0.902 ± 0.001** | **0.892 ± 0.002** | **0.565** | **0.188** |
| Logistic Regression (Event-Pooled) | Linear Baseline | 0.914 | 0.902 | 0.570 | 0.280 |

* **Key Takeaways:** Inductive mode restores performance on unseen snowflakes (peaking at **0.905 AUC-ROC**); a single propagation hop (`K = 1`) is sufficient and **3.4×** faster once evidence is pooled at the event level; and ExpressGNN approaches the linear baseline while offering explainable first-order logic rules.

---

## Requirements
- Python 3.7+
- PyTorch 1.1+
- scikit-learn
- networkx
- tqdm

---

## Quick Start

### 1. Run a Single Experiment (Inductive Mode)
To train and evaluate ExpressGNN on the `food_v4_unary` dataset in **Inductive mode**, set `-trans 0` and `-gcn_free_size 0`:

```bash
python main/train.py \
  -data_root data/food_v4_unary \
  -load_method 0 \
  -trans 0 \
  -gcn_free_size 0 \
  -embedding_size 64 \
  -slice_dim 16 \
  -num_hops 2 \
  -learning_rate 0.001 \
  -num_batches 50 \
  -num_epochs 10
```

> **Configuration Notes:**
> - `-trans 0` & `-gcn_free_size 0`: Enables **Inductive mode** (removes per-entity One-Hot embeddings so the model generalizes across unseen snowflake subgraphs). Set `-trans 1` to reproduce the **Transductive** baseline.
> - `-num_hops`: Use `-num_hops 2` for full event-firm structural propagation, or `-num_hops 1` for faster convergence (`22.5 s/epoch`) when event-pooled features are already present.

### 2. Run Full Paper Benchmarks & Statistical Reports
To reproduce all experimental configurations, multi-seed evaluations (seeds `10, 20, 30`), ablation studies, and summary statistics reported in the paper:

```bash
python run_paper_benchmarks.py
```

---

## References
1. **This Work:** L. H. Hung, D. T. Minh, and T. N. H. Anh, *"Inductive Reasoning with ExpressGNN on Snowflake Graphs: Food Recall Classification,"* Student Scientific Research Communication, Vol. I, 2025.
2. **Base Architecture:** Y. Zhang, X. Chen, Y. Yang, A. Ramamurthy, B. Li, Y. Qi, and L. Song, *"Efficient Probabilistic Logic Reasoning with Graph Neural Networks,"* in Proc. ICLR, 2020.
