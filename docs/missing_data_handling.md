# Summary: Handling Systematic Missingness for Prudential with KAN (Tabular)

## Key idea
For Prudential-style **systematic / structured missingness**, the strongest “SOTA-default” approach for **KANs** is to **treat missingness as signal**:
- **Do not rely on imputation alone.**
- Encode missingness explicitly and let the KAN learn how it affects risk.

Concretely, model each feature with:
- a (possibly imputed) **value channel** $x_i$
- a **missingness mask** $m_i \in \{0,1\}$ (1 observed, 0 missing)

This is very KAN-compatible because KAN edges learn **univariate** functions: you can give the network separate univariate inputs for value and mask.

---

## Recommended “SOTA-default” recipe (KAN-focused)

1. **Create missingness masks**
   - For every feature $i$, build $m_i \in \{0,1\}$.

2. **Numeric features**
   - Compute scaling (e.g., z-score) using **observed** values only.
   - **Baseline:** impute missing numeric values to 0 **after scaling**, and feed both $x_i$ and $m_i$.
   - **Upgrade for systematic missingness:** learn a per-feature numeric “missing token” $a_i$:
     $$
     x_i^{*} = m_i \, x_i^{\mathrm{obs}} + (1-m_i)\, a_i
     $$
     and still feed $m_i$.

3. **Categorical features**
   - Add an explicit **“MISSING”** category (embedding/code).
   - Optionally also feed a categorical missingness mask (often redundant if “MISSING” is explicit, but sometimes helpful).

4. **KAN inputs**
   - Concatenate the **value vector** and **mask vector** (and categorical embeddings as usual):
     $$
     \mathrm{input} = [x^{*}, m]
     $$
   - This lets the KAN learn separate univariate functions for “value effect” and “missingness effect”.

5. **Missingness-aware training (strongly recommended)**
   - Apply **feature masking augmentation** during training:
     - randomly (or pattern-aware) drop a subset of *observed* features
     - simulate Prudential-like missingness patterns
   - This improves robustness and reduces over-reliance on any single feature block.

6. **Optional: high-quality imputation (only if needed)**
   - If you truly need a fully imputed table, pretrain a modern masked-autoencoder-style imputer.
   - Still keep and feed masks $m$ to the KAN afterward (to preserve informative missingness).

---

## What to avoid (common pitfall)
- **Label-conditioned imputation** (imputing using the target label) is not deployable at test time and risks leakage.
- Prefer label-free imputers and/or the $(\mathrm{value} + \mathrm{mask})$ modeling setup above.

---

## Minimal KAN-compatible formulation (conceptual)
A KAN can naturally express:
$$
f(x,m)=\sum_{i=1}^{d} \phi_i\!\left(x_i^{*}\right) \;+\; \sum_{i=1}^{d} \psi_i(m_i) \;+\; \cdots
$$

Where:
- $\phi_i$ learns the effect of the (filled) feature value
- $\psi_i$ learns the effect of the feature being present/absent
- $\cdots$ denotes higher-layer compositions (depending on your KAN depth)