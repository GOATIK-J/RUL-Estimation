# Deep Learning-Based Predictive Maintenance & Remaining Useful Life (RUL) Estimation
**NASA C-MAPSS turbofan engines · TensorFlow / Keras only (no PyTorch anywhere) · web app (HTML/CSS/JS front end + Python API)**

## Results (real C-MAPSS data, official test protocol)
| Dataset | Selected model | Val R² | **Test R²** (capped RUL) | Test R² (raw NASA labels) | Test RMSE | Test MAE | Anomaly F1 |
|---|---|---|---|---|---|---|---|
| **FD001** (1 condition) | Classical ensemble (ExtraTrees + GBM) | 0.945 | **0.923** | 0.910 | 11.1 cycles | 8.0 cycles | 0.82 |
| **FD004** (6 conditions, hardest) | HistGradientBoosting | 0.891 | **0.913** | 0.789 | 12.7 cycles | 8.7 cycles | 0.76 |

Previous project: R² = 0.17 (validation) / 0.68 (test, FD004). The numbers above come from the classical models, which run
anywhere. When you run Phase 3 with TensorFlow installed, the 6 Keras architectures are trained as well and the notebook
**automatically picks the best model on validation engines** (the test set is never used for selection). Numbers printed by your
run are the ones to quote.

*Capped vs raw:* models are trained on RUL capped at 125 (standard). "Test R² (capped)" compares with the capped NASA label;
"raw" compares with NASA's uncapped label (FD004 has labels up to 195 which no capped model can reach). Both are reported.

## Why the old R² was only 0.17 — and what changed
1. **Validation compared against the *uncapped* RUL while the model learns the *capped* RUL** → unfixable errors. Now both use the capped target; raw is reported separately.
2. **Validation snapshots only from 60-90 % of life** → tiny RUL range → R² deflated. Now: all windows of held-out engines.
3. **FD004 sensors were scaled globally** although it has 6 operating conditions. Now: operating condition detection (k-means) + per-condition z-score.
4. **No noise suppression / no age information.** Now: causal exponential smoothing (span 20, chosen by 5-fold cross-validation over *training engines only*) + operating-cycle feature.
5. **Tiny model, quick run.** Now: 6 architectures × 3 seeds, ensembles, early stopping, dropout, L2, input-noise augmentation.
6. **No safety net.** Now: if a deep model fails or TensorFlow is missing, the best classical model is exported, so every phase always finishes.

## How over-fitting is controlled
Split by **engine** (never by row) · scalers/clusters/healthy baseline fitted on training engines only · causal smoothing (no future data) ·
early stopping with best-weight restore · dropout, L2, Gaussian input noise · seed ensembles · model chosen on **validation only** ·
final check table *train vs validation vs test* (Phase 3 §7). Training R² of tree ensembles is always optimistic (neighbouring windows of one engine
are near-copies); the honest evidence is that **validation and test engines agree** (e.g. FD001: 0.945 vs 0.923).

## Project layout
```
01_Phase1_Data_Audit_and_EDA.ipynb          data download/audit, EDA, design decisions        (standalone)
02_Phase2_Preprocessing_and_Windowing.ipynb RUL labels, split, scaling, windows, anomaly baseline (standalone)
03_Phase3_Models_Evaluation_XAI_Deployment.ipynb  baselines, Keras models, selection, XAI, compression, export (standalone)
src/      rul_core.py (data, preprocessing, metrics, anomaly) · rul_models.py (Keras) · inference.py · zoo.py (model zoo) · api_core.py (API) · imageid.py (photo identification) · server_std.py · viz.py
web/      index.html · css/app.css · js/ (charts.js, core.js, views.js, app.js)  - the web app, no build step, no CDN
run.py    starts the web app      backend/main.py  FastAPI adapter      tests/  checks      tools/  notebook builder, train_deep_models.py, build_zoo.py
artifacts/ results, figures, exported models, dashboard data (created by the notebooks; a ready-made copy is included)
```
Each notebook is **self-contained**: its first cells write the `src/*.py` files it needs, so a single notebook can be uploaded to Google Colab alone.

## Run it
```bash
pip install -r requirements.txt            # numpy, pandas, scikit-learn, tensorflow, fastapi, uvicorn, pillow ...
python run.py                              # opens the web app at http://127.0.0.1:8000  (a ready-made model zoo is included)
python run.py --stdlib                     # force Python's built-in server (no FastAPI needed)
python tests/test_pipeline.py              # automatic checks
```
To reproduce everything: put `CMAPSSData.zip` in `data/raw/`, run notebooks 01 -> 02 -> 03 (Phase 3: 20-60 min on CPU; `QUICK_RUN=True` gives a 5-min test).

### Train the six deep-learning models (needs TensorFlow)
The zip ships with the classical models evaluated, because TensorFlow was not available where it was prepared. Nothing in the app is invented:
untrained deep models appear greyed out as "not trained yet". To fill them in:
```bash
pip install tensorflow
python tools/train_deep_models.py --quick          # a few minutes, first test
python tools/train_deep_models.py                  # full run: 6 architectures x 3 seeds per dataset
python run.py                                      # the Compare page now shows deep vs classical side by side
```
(Running notebook 03 with TensorFlow installed does the same and also builds the zoo.)

## Web app pages
* **Fleet** - every test machine as a coloured tile (predicted RUL), KPI counts, predicted-vs-actual, maintenance priority list.
* **Machine** - the health monitor card (health %, anomaly score, RUL, status, trends, action), RUL/anomaly timelines with a replay slider, "why" chart, sensor trends.
* **Compare models** - leaderboard of every model on the same unseen machines (R², RMSE, MAE, NASA score, size, latency), bar chart, error spread, predicted-vs-actual, overfitting check, feature reliance, one machine / every model.
* **Live prediction** - upload a sensor CSV (or take a sample machine), pick the model in the top bar, get the dashboard; "Compare all models" runs every model on the same data.
* **Photo** - upload a machine photo to open its dashboard (see scope below).
* **How it works** - plain-language pipeline and metric glossary.
The **model selector** in the top bar changes the model used by Fleet, Machine, Live and Photo, so the difference between deep and classical models is visible everywhere.

### What the photo feature does (and does not do)
A photo cannot show how many cycles an engine has left; that comes from sensor history. The photo **identifies the machine**: machine ID in the file name (`M-023.jpg`), a QR code in the picture (Chrome/Edge), or a perceptual-hash match against photos registered earlier (12 sample photos are included; unknown photos can be registered to a machine). The dashboard is then built from that machine's sensor data.

## Honest notes
* C-MAPSS has no vibration sensor; fan/core speed sensors are used as the rotating-machinery analogue.
* The "why" chart uses occlusion (all models) and Integrated Gradients (Keras models). The `cycle` feature (engine age) often ranks high - that is real information, not a bug.
* Anomaly labels: "degraded" = true RUL ≤ 50 cycles; the detector itself is unsupervised (Mahalanobis distance from healthy behaviour).
