# AIGIS-IDS-SCADA-IEC104

AIGIS is an Intrusion Detection System based on Machine Learning (ML) and Deep Learning (DL) algorithms for SCADA networks.

The models were trained on the ITHACA lab's IEC 60870-5-104 Intrusion Detection Dataset: https://zenodo.org/records/7108614

The project is separated into two parts:

1. *Supervised Learning with boosted-tree algorithms*
2. *Anomaly Detection algorithms*

---

# Supervised Learning Version

The classes of the dataset are:

| Class | IEC 104 command |
|---|---|
| `NORMAL` | Normal traffic |
| `c_ci_na_1` | Counter Interrogation |
| `c_ci_na_1_DoS` | Counter Interrogation (DoS) |
| `c_rd_na_1` | Read Command |
| `c_rd_na_1_DoS` | Read Command (DoS) |
| `c_rp_na_1` | Reset Process |
| `c_rp_na_1_DoS` | Reset Process (DoS) |
| `c_sc_na_1` | Single Command |
| `c_sc_na_1_DoS` | Single Command (DoS) |
| `c_se_na_1` | Set-point Command |
| `c_se_na_1_DoS` | Set-point Command (DoS) |
| `m_sp_na_1_DoS` | Single-point Information (DoS) |

To understand why the supervised tree-based models plateau, their performance was analysed and visualized with SHAP, ROC-AUC, PR-AUC, calibration curves and UMAP. Finally, a two-stage classifier was added, with a model specifically trained on the classes that did worse (c_rd_na_1, c_rd_na_1_DoS, c_rp_na_1, c_rp_na_1_DoS) so that it specializes in predicting only them.

To achieve this, the 4 worst-performing classes were merged into one `c_rd_rp_family` label, numbered as class `3`, for the first stage, the second-stage model was then trained on a train/test set containing only those 4 classes.

Models used:

- XGBoost
- LightGBM
- CatBoost

GridSearchCV and RFECV were used to find the best hyperparameters of the models and drop the unnecessary features.

### Results of the Supervised Learning version

**Stage 1 — XGBoost, 9 classes, accuracy 0.82, macro f1 0.76:**

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| NORMAL | 1.00 | 1.00 | 1.00 | 210 |
| c_ci_na_1 | 0.48 | 0.61 | 0.54 | 210 |
| c_ci_na_1_DoS | 0.80 | 0.18 | 0.29 | 210 |
| c_rd_rp_family | 1.00 | 0.96 | 0.98 | 840 |
| c_sc_na_1 | 0.51 | 0.86 | 0.64 | 210 |
| c_sc_na_1_DoS | 0.91 | 0.96 | 0.93 | 210 |
| c_se_na_1 | 0.68 | 0.78 | 0.73 | 210 |
| c_se_na_1_DoS | 0.81 | 0.63 | 0.71 | 210 |
| m_sp_na_1_DoS | 1.00 | 1.00 | 1.00 | 210 |
| **macro avg** | 0.80 | 0.77 | 0.76 | 2520 |
| **weighted avg** | 0.85 | 0.82 | 0.81 | 2520 |

**Stage 2 — quad XGB model, 4 classes, accuracy 0.36, macro f1 0.35:**

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| c_rd_na_1 | 0.24 | 0.19 | 0.21 | 181 |
| c_rd_na_1_DoS | 0.46 | 0.32 | 0.38 | 204 |
| c_rp_na_1 | 0.36 | 0.45 | 0.40 | 210 |
| c_rp_na_1_DoS | 0.37 | 0.45 | 0.41 | 210 |
| **macro avg** | 0.36 | 0.35 | 0.35 | 805 |
| **weighted avg** | 0.36 | 0.36 | 0.35 | 805 |

The model lost 29 samples from c_rd_na_1 and 6 samples from c_rd_na_1_DoS, which means the first-stage classifier couldn't even recognize those samples as the `c_rd_rp_family`.

### Single-stage XGBoost — all 12 classes (accuracy 0.60)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| NORMAL | 1.00 | 1.00 | 1.00 | 210 |
| c_ci_na_1 | 0.47 | 0.60 | 0.53 | 210 |
| c_ci_na_1_DoS | 0.86 | 0.18 | 0.30 | 210 |
| c_rd_na_1 | 0.18 | 0.13 | 0.15 | 210 |
| c_rd_na_1_DoS | 0.49 | 0.21 | 0.29 | 210 |
| c_rp_na_1 | 0.33 | 0.44 | 0.38 | 210 |
| c_rp_na_1_DoS | 0.36 | 0.50 | 0.42 | 210 |
| c_sc_na_1 | 0.47 | 0.76 | 0.58 | 210 |
| c_sc_na_1_DoS | 0.83 | 0.95 | 0.89 | 210 |
| c_se_na_1 | 0.68 | 0.78 | 0.73 | 210 |
| c_se_na_1_DoS | 0.80 | 0.62 | 0.70 | 210 |
| m_sp_na_1_DoS | 1.00 | 1.00 | 1.00 | 210 |
| **Macro avg** | 0.62 | 0.60 | 0.58 | 2520 |
| **Weighted avg** | 0.62 | 0.60 | 0.58 | 2520 |

### Comparison

| class | single-stage XGB | two-stage specialist | difference f1 |
|---|---|---|---|
| c_rd_na_1 | 0.15 | 0.21 | +0.06 |
| c_rd_na_1_DoS | 0.29 | 0.38 | +0.09 |
| c_rp_na_1 | 0.38 | 0.40 | +0.02 |
| c_rp_na_1_DoS | 0.42 | 0.41 | −0.01 |


# Anomaly Detection Version


models used:

- Isolation Forest
- One class Support Vector Machine
- Autoencoder (PyTorch)

### Results of the Anomaly Detection Version

**Isolation Forest — accuracy 0.9980** (contamination 0.02, n_estimators 100)

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Attack | 1.00 | 1.00 | 1.00 | 2310 |
| Normal | 1.00 | 0.98 | 0.99 | 210 |
| **macro avg** | 1.00 | 0.99 | 0.99 | 2520 |
| **weighted avg** | 1.00 | 1.00 | 1.00 | 2520 |

**One-Class SVM — accuracy 0.9980** (nu 0.01, gamma 0.001)

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Attack | 1.00 | 1.00 | 1.00 | 2310 |
| Normal | 1.00 | 0.98 | 0.99 | 210 |
| **macro avg** | 1.00 | 0.99 | 0.99 | 2520 |
| **weighted avg** | 1.00 | 1.00 | 1.00 | 2520 |

*** Autoencoder — accuracy 0.9988*** (epochs 1000)

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Attack | 1.00 | 1.00 | 1.00 | 2310 |
| Normal | 1.00 | 0.99 | 0.99 | 210 |
| **macro avg** | 1.00 | 0.99 | 0.99 | 2520 |
| **weighted avg** | 1.00 | 1.00 | 1.00 | 2520 |

---


# Conclusion
### The Anomaly detection methods were almost 100% accurate while the Supervised version is dealing with classes that look very similar to each other because of their features. The anomaly detection method is more useful on a IDS since the main goal is to identify the attack rather than classify it 
More inforamtion about everything shown here is marked down in the .ipynb files 

There is only a gain on c_rd_na_1 
