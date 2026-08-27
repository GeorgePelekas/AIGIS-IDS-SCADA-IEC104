# AIGIS-IDS-SCADA-IEC104

AIGIS is an Intrusion Detection System based on Machine Learning (ML) and Deep Learning (DL) algorithms for SCADA networks.

The models were trained on the ITHACA lab's IEC 60870-5-104 Intrusion Detection Dataset: https://zenodo.org/records/7108614 on the `iec104_custom_script_train_final.csv` and `iec104_custom_script_test_final.csv`

The project showcases 2 jupyter notebook files.
The **AIGIS.ipynb** file cointaining sections from Anomaly detectors, multilabel classifiers and data visualization.
The **AIGIS_UNSUPERVISED.ipynb** containing only the anomaly detection part and the search of the models hyperparameters.


---

# Requirments
## Setup

Requires Python 3.14.

```bash
pip install -r requirements.txt
```

The dataset is not included in the repository. Download the balanced
CSV files from the [Zenodo record](https://zenodo.org/records/7108614)
and place `iec104_custom_script_train_final.csv` and
`iec104_custom_script_test_final.csv` in the project root.

Main dependencies: scikit-learn, XGBoost, LightGBM, CatBoost, SHAP,
umap-learn, PyTorch.

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

To understand why the supervised tree-based models plateau, their
performance was analysed and visualized with confusion matrices, SHAP,
ROC-AUC, PR-AUC, calibration curves and UMAP.

The final architecture is a three-stage cascade. The first stage is an
anomaly detector that separates normal network flows from anything that
deviates from them. The second stage is a 9-class classifier, in which
the four most-confused classes (`c_sc_na_1`, `c_sc_na_1_DoS`,
`c_se_na_1`, `c_se_na_1_DoS`) are merged into a single `c_sc_and_c_se`
label. Flows routed to that label are then passed to a third-stage model
trained only on those four classes, so that it specializes in telling
them apart.

Models used:

- **Random Forest**
- XGBoost
- LightGBM
- CatBoost

GridSearchCV and RFECV were used to find the best hyperparameters of the models and drop the unnecessary features.


### Random Forest FLAT

| class | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| NORMAL | 1.00 | 1.00 | 1.00 | 169 |
| c_ci_na_1 | 0.99 | 1.00 | 1.00 | 169 |
| c_ci_na_1_DoS | 0.78 | 1.00 | 0.87 | 169 |
| c_rd_na_1 | 0.99 | 1.00 | 1.00 | 169 |
| c_rd_na_1_DoS | 0.94 | 1.00 | 0.97 | 169 |
| c_rp_na_1 | 1.00 | 0.99 | 1.00 | 169 |
| c_rp_na_1_DoS | 0.97 | 1.00 | 0.99 | 169 |
| c_sc_na_1 | 0.58 | 0.37 | 0.45 | 169 |
| c_sc_na_1_DoS | 0.80 | 0.76 | 0.78 | 169 |
| c_se_na_1 | 0.46 | 0.56 | 0.51 | 169 |
| c_se_na_1_DoS | 0.50 | 0.41 | 0.45 | 169 |
| m_sp_na_1_DoS | 1.00 | 1.00 | 1.00 | 169 |
| **accuracy** | | | **0.84** | **2028** |
| **macro avg** | **0.84** | **0.84** | **0.83** | **2028** |
| **weighted avg** | **0.84** | **0.84** | **0.83** | **2028** |

### 3-stage Cascade (Isolation Forest → Random Forest → CatBoost)

**Stage 1 — Isolation Forest** (refitted on all 400 NORMAL training flows):

| class | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| Attack | 1.00 | 1.00 | 1.00 | 1859 |
| Normal | 1.00 | 0.99 | 0.99 | 169 |
| **accuracy** | | | **1.00** | **2028** |
| **macro avg** | 1.00 | 0.99 | 1.00 | 2028 |
| **weighted avg** | 1.00 | 1.00 | 1.00 | 2028 |

Every attack flow is routed onward; two of the 169 normal flows are
misrouted. No attack is labelled NORMAL at this stage.

**End-to-end, all 12 classes:**

| class | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| NORMAL | 1.00 | 0.99 | 1.00 | 169 |
| c_ci_na_1 | 0.99 | 1.00 | 1.00 | 169 |
| c_ci_na_1_DoS | 0.81 | 0.99 | 0.89 | 169 |
| c_rd_na_1 | 0.99 | 0.98 | 0.98 | 169 |
| c_rd_na_1_DoS | 0.98 | 1.00 | 0.99 | 169 |
| c_rp_na_1 | 0.99 | 0.99 | 0.99 | 169 |
| c_rp_na_1_DoS | 0.99 | 0.99 | 0.99 | 169 |
| c_sc_na_1 | 0.68 | 0.60 | 0.64 | 169 |
| c_sc_na_1_DoS | 0.80 | 0.78 | 0.79 | 169 |
| c_se_na_1 | 0.55 | 0.66 | 0.60 | 169 |
| c_se_na_1_DoS | 0.66 | 0.47 | 0.55 | 169 |
| m_sp_na_1_DoS | 1.00 | 1.00 | 1.00 | 169 |
| **accuracy** | | | **0.87** | **2028** |
| **macro avg** | **0.87** | **0.87** | **0.87** | **2028** |
| **weighted avg** | **0.87** | **0.87** | **0.87** | **2028** |

### Comparison

| class | Random Forest FLAT | Random Forest + CatBoost hybrid | difference f1 |
|---|---|---|---|
| NORMAL | 1.00 | 1.00 | +0.00 |
| c_ci_na_1 | 1.00 | 1.00 | +0.00 |
| c_ci_na_1_DoS | 0.87 | 0.89 | +0.02 |
| c_rd_na_1 | 1.00 | 0.98 | −0.02 |
| c_rd_na_1_DoS | 0.97 | 0.99 | +0.02 |
| c_rp_na_1 | 1.00 | 0.99 | −0.01 |
| c_rp_na_1_DoS | 0.99 | 0.99 | +0.00 |
| c_sc_na_1 | 0.45 | 0.64 | +0.19 |
| c_sc_na_1_DoS | 0.78 | 0.79 | +0.01 |
| c_se_na_1 | 0.51 | 0.60 | +0.09 |
| c_se_na_1_DoS | 0.45 | 0.55 | +0.10 |
| m_sp_na_1_DoS | 1.00 | 1.00 | +0.00 |
| **Macro avg** | **0.83** | **0.87** | **+0.04** |
| **Weighted avg** | **0.83** | **0.87** | **+0.04** |


A diagram of how it works:

                     Data
                      |
                      v
             +------------------+
             |   Stage 1        |
             | Isolation Forest  |
             +------------------+
                /            \
           Normal           Attack
                              |
                              v
                   +------------------+
                   |    Stage 2       |
                   | Random Forest    |
                   |    9 classes     |
                   +------------------+
                              |
                    c_sc_and_c_se
                              |
                              v
                   +------------------+
                   |    Stage 3       |
                   |    CatBoost      |
                   |    4 classes     |
                   +------------------+

# Anomaly Detection Version


models used:

- **Isolation Forest**
- One class Support Vector Machine
- Autoencoder (PyTorch)

### Results of the Anomaly Detection Version


All three models were fit on the same 320 NORMAL training flows.
Hyperparameters were selected on a held-out validation set; the test
set was used once. All timings are CPU timings (Ryzen 7 9800X3D),
averaged over 5 runs.

| Model | ROC-AUC | Detection rate | False alarms | False alarm rate | Fit | Predict | Parameters |
|---|---|---|---|---|---|---|---|
| **Isolation Forest** | 1.0000 | 1.00 | **3 / 169** | **0.02** | 54.1 ms | 6.16 ms | contamination 0.033, n_estimators 100 |
| One-Class SVM | 1.0000 | 1.00 | 5 / 169 | 0.03 | **0.32 ms** | 1.57 ms | nu 0.01, gamma 0.001 |
| Autoencoder | 1.0000 | 1.00 | 13 / 169 | 0.08 | 730 ms | **0.68 ms** | 1000 epochs, 95th-percentile threshold |

### Isolation Forest

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Attack | 1.00 | 1.00 | 1.00 | 1859 |
| Normal | 1.00 | 0.98 | 0.99 | 169 |
| **macro avg** | 1.00 | 0.99 | 1.00 | 2028 |
| **weighted avg** | 1.00 | 1.00 | 1.00 | 2028 |

### One-Class SVM

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Attack | 1.00 | 1.00 | 1.00 | 1859 |
| Normal | 1.00 | 0.97 | 0.98 | 169 |
| **macro avg** | 1.00 | 0.99 | 0.99 | 2028 |
| **weighted avg** | 1.00 | 1.00 | 1.00 | 2028 |

### Autoencoder

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Attack | 0.99 | 1.00 | 1.00 | 1859 |
| Normal | 1.00 | 0.92 | 0.96 | 169 |
| **macro avg** | 1.00 | 0.96 | 0.98 | 2028 |
| **weighted avg** | 0.99 | 0.99 | 0.99 | 2028 |


ROC-AUC is 1.0000 for all three models. The models differ only in how many normal flows they wrongly flag as attacks.

All three detect every attack in the test set. The Isolation Forest
produces the fewest false alarms and needs no feature scaling, so it was
selected for the first stage of the cascade. It is not the cheapest to
run — the One-Class SVM fits 170× faster and the autoencoder predicts
9× faster — but at this data scale the costs are negligible.
the false alarm rate matters more.

The model deployed in the cascade is refitted on all 400 NORMAL training
flows once the hyperparameters are fixed, which brings the false alarms
on the test set down from 3 to 2 (Normal recall 0.99 instead of 0.98).
The figures in the table above are those of the tuning fit (320 flows).

---

## Dataset & Citation

This project uses the **IEC 60870-5-104 Intrusion Detection Dataset**, created by the ITHACA lab at the University of Western Macedonia (https://ithaca.ece.uowm.gr/).

- Zenodo record: https://zenodo.org/records/7108614
- DOI: [10.21227/fj7s-f281](https://doi.org/10.21227/fj7s-f281)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Authors: P. Radoglou-Grammatikis, K. Rompolos, T. Lagkas, V. Argyriou, P. Sarigiannidis

As requested by the dataset authors, please cite:

> P. Radoglou-Grammatikis, K. Rompolos, P. Sarigiannidis, V. Argyriou, T. Lagkas, A. Sarigiannidis, S. Goudos and S. Wan, "Modeling, Detecting, and Mitigating Threats Against Industrial Healthcare Systems: A Combined Software Defined Networking and Reinforcement Learning Approach," *IEEE Transactions on Industrial Informatics*, vol. 18, no. 3, pp. 2041–2052, March 2022, doi: 10.1109/TII.2021.3093905.
> https://ieeexplore.ieee.org/document/9470933

The dataset was produced within the H2020 projects **ELECTRON** (grant agreement No 101021936) and **SDN-microSENSE** (No 833955).

The IEC 60870-5-104 flow statistics used here come from the dataset's Custom IEC 60870-5-104 Python Parser (Scapy-based); the TCP/IP flow statistics come from CICFlowMeter.
