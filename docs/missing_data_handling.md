# Prudential Life Insurance Assessment + KAN: Handling Systematic Missingness (Practical SOTA Playbook)

## Context: why missingness matters on Prudential
The **Prudential Life Insurance Assessment** dataset (Kaggle) is widely treated as a tabular benchmark with **heavy, structured / systematic missingness** (e.g., whole “blocks” of features missing for many applicants, especially in medical-history style fields).  
- Competition page: https://www.kaggle.com/c/prudential-life-insurance-assessment  
- Example EDA noting widespread missingness patterns: https://www.kaggle.com/code/ithesisart/prudential-life-insurance-assessment-edaml  

For models like **KANs** (Kolmogorov–Arnold Networks) where learnable *univariate* functions live on edges, the missing-data strategy should be **KAN-native**:
> **Do not try to “erase” missingness via imputation alone. Encode missingness explicitly, and let the model learn how missingness correlates with risk.**

---

## SOTA-ish default for KAN: value + mask (and often a trainable missing token)
### 1) Always add a missingness indicator (mask)
For each original feature `i`, create:
- `x_i` = numeric value after preprocessing/imputation (or categorical encoding)
- `m_i ∈ {0,1}` = 1 if observed, 0 if missing

This “Missing Indicator Method” is strongly supported in modern supervised learning practice: indicator + simple imputation is often competitive and captures *informative missingness* when it exists.
- Paper: **The Missing Indicator Method: From Low to High Dimensions** (Van Ness et al.)  
  https://arxiv.org/abs/2211.09259

**KAN mapping (conceptually):**
\[
f(x,m)=\sum_i \phi_i(x_i) + \sum_i \psi_i(m_i) + \text{(compositions over layers)}
\]
Because KAN edges are univariate, feeding `(x_i, m_i)` as separate inputs fits naturally.

---

### 2) Use “zero/median + mask” as the baseline (it’s hard to beat)
A strong, simple baseline for deep tabular is:
- **numeric**: standardize (on observed values), then **impute 0**, and feed the mask  
- **categorical**: treat missing as its own level (“MISSING”), optionally also feed the mask

This is consistent with empirical evidence that **zero imputation** can be as effective as more complex deep impute-then-predict pipelines (when combined with sensible modeling).
- Paper: **In Defense of Zero Imputation for Tabular Deep Learning**  
  https://openreview.net/forum?id=H0gENXL7F2

**Practical note for KANs:** if your spline/edge functions are sensitive to out-of-domain spikes, zero-imputing *after standardization* keeps missing values in a stable, “central” region.

---

### 3) Upgrade for systematic missingness: learn a numeric “missing token” per feature
When missingness is systematic (often closer to MAR/MNAR than MCAR), replace fixed imputation with a **trainable scalar** per feature:

\[
x_i^\* = m_i \cdot x_i^{obs} + (1-m_i)\cdot a_i
\]
- `a_i` is a learned parameter (“numeric missing token”)
- still feed `m_i`

This lets the KAN represent “missing for feature i” as a meaningful latent value rather than assuming it equals the mean/median.

**Minimal PyTorch sketch:**
```python
# x: [batch, d] with NaNs
# m: [batch, d] mask (1 observed, 0 missing)
# a: [d] trainable missing token
x_filled = torch.nan_to_num(x, nan=0.0)             # temporary fill
x_star = m * x_filled + (1 - m) * a                 # learned replacement
# model_input = concat([x_star, m], dim=-1)