# Viva guide — explain the project in 5 minutes

**One sentence:** *From raw multivariate sensor time series to explainable machine-health intelligence: we predict how many cycles an engine can still run (RUL), flag abnormal behaviour, and explain why.*

## Story (follow the three notebooks)
1. **Problem** – unplanned failures are expensive; fixed-schedule maintenance is too early or too late. We need *RUL*, not just "normal/failure".
2. **Phase 1 – Data audit & EDA** – NASA C-MAPSS: 100–260 engines run to failure, 21 sensors + 3 settings. Findings: constant sensors carry no information; FD004 has 6 operating conditions that hide the degradation signal.
3. **Phase 2 – Preprocessing** – RUL label = last cycle − current cycle, capped at 125; split by engine; per-condition normalisation; drop constant sensors; causal smoothing; windows of the last 30/40 cycles; healthy baseline for anomaly detection.
4. **Phase 3 – Models** – classical baselines vs MLP / LSTM / GRU / BiLSTM / CNN-LSTM / CNN-BiLSTM-Attention; ensembles; selection on validation only; test on the official protocol; XAI; compression; export to API + dashboard.

## Likely questions
* **Why not just use the best model on the test set?** Test is touched once, after selection on validation engines; otherwise the score would be optimistic.
* **Is the high R² over-fitting?** Splits are by engine; validation and test engines (never seen in training) give the same level (FD001 0.945 / 0.923). Early stopping, dropout, L2, noise, ensembles, cross-validated preprocessing.
* **Why cap RUL at 125?** Early in life a healthy engine shows no degradation, so "250 vs 150 cycles left" is unknowable from sensors. Capping is the standard C-MAPSS practice; we report raw-label metrics too.
* **Why did the old R² (0.17) happen?** Uncapped validation labels vs capped training target, a narrow validation range, and global scaling of a 6-condition dataset (see README).
* **Deep learning vs classical?** Reported side by side. Classical window-statistics models are strong on C-MAPSS; the notebook lets the validation score decide and says so. (Do not claim deep learning wins unless your table shows it.)
* **Why LSTM + CNN + attention?** CNN = local sensor patterns, LSTM = temporal dependence, attention = which time steps matter.
* **NASA score?** Asymmetric: predicting failure *later* than reality (dangerous) costs more than predicting it earlier.
* **How is anomaly detection evaluated?** Unsupervised Mahalanobis distance from healthy behaviour; "degraded" := true RUL ≤ 50; precision/recall/F1/ROC-AUC reported.
* **Edge deployment?** Distillation (tiny GRU) + float16/TFLite quantisation; size, latency and error are tabulated (needs TensorFlow).

## The web app (questions your professor may ask)
* **Why HTML/CSS/JS instead of Streamlit?** Full control of the design, instant interaction (replay slider, model switching), no build step, and a clean split: the Python API (`src/api_core.py`) serves JSON, the browser draws the charts.
* **How do you compare deep learning and classical models fairly?** Every model is evaluated on the same held-out engines and the same official test machines (`src/zoo.py`). The recommended model is chosen on validation error only; the test set is never used for selection.
* **Where are the deep models?** Train them with `python tools/train_deep_models.py` (TensorFlow/Keras). Until then they show as "not trained yet" - the app never fabricates numbers.
* **What does the photo upload do?** It identifies the machine (file name, QR code, or image-hash match with a registered photo) and opens that machine's sensor-based dashboard. A picture alone cannot reveal remaining life, and we say so in the app.
* **How is overfitting checked?** Train / validation / test R² per model on the Compare page ("Overfitting check"); a small gap means the model generalises.
