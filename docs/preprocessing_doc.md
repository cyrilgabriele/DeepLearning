# Preprocessing Video Notes

Curated transcript-based notes for the requested encoding videos so the preprocessing discussion stays grounded in the original explanations.

## Target-Encoding Foundations

### [StatQuest: One-Hot, Label, and Target Encoding](https://www.youtube.com/watch?v=589nCGeWG1w)
- Walks through why discrete insurance-style features often need to become numeric via one-hot, label, or target encoding, using the Troll 2 toy dataset from the transcript.
- Highlights strengths/weaknesses: one-hot is safe but explodes dimensionality for high-cardinality codes; label encoding injects arbitrary ordering that can mislead trees.
- Derives vanilla target encoding, then the Bayesian/weighted variant (combining category-level mean with the global target mean through the `m` prior) to stabilize rare categories.
- Ends by showing how k-fold target encoding breaks leakage loops by computing encodings for each fold from out-of-fold targets—exactly the leakage guard we rely on before any KAN training.

### [Target Encoding Explained with Day/Temperature Toy Data](https://www.youtube.com/watch?v=hGAbsHWitwo)
- Re-derives mean/likelihood encoding straight from the transcript’s day–temperature–play dataset: encode each category as the empirical probability of the outcome.
- Extends the idea to multi-class targets (duplicate encoded columns per class) and regression targets (replace probability with average numeric response), reinforcing when each variant is useful.
- Cautions that blindly averaging the full dataset leaks answers into the features, causing overfitting and drift when future distributions shift; tees up Bayesian smoothing and cross-validation as the remedy.
- Useful as a “pen-and-paper” sanity check for our Prudential preprocessing because it shows exactly what the encodings should equal before batching/spline-fitting inside a KAN.

## CatBoost / Ordered Target Encoding

### [StatQuest: CatBoost Part 1 – Ordered Target Encoding](https://www.youtube.com/watch?v=KXOTSkPL2X4)
- Transcript explains how CatBoost simulates online learning: iterate through rows in a random order, encode each categorical value using only the targets seen earlier, and default to a prior (0.05 in the example) when nothing precedes it.
- Shows how the ordered formula simply adds 1 to the denominator (instead of an arbitrary weight) and stores running option counts so leakage is impossible—even if a category only appears once.
- Clarifies that production CatBoost averages many random permutations and then uses the full dataset to encode brand-new rows; that mirrors our need to freeze encoders after the training phase before pushing inputs into a KAN.
- Takeaway for KAN preprocessing: ordered/Bayesian encodings plus [-1, 1] scaling (as done in `prudential_paper_preprocessing.py`) give spline layers stable, leakage-free categorical signals.
