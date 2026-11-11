# Bank Marketing: Campaign Analysis & Subscription Prediction

**Which customers are most likely to subscribe to a term deposit, and can a model help a bank decide who to call first?**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

A customer analytics case study on **11,162 bank marketing records**: segment-level subscription analysis, followed by a classification model that ranks customers for outreach.

## Key findings

**KPI:** subscription rate = customers with `deposit = yes` ÷ customers in the segment. Overall sample rate: **~47%**.

| Segment insight | Subscription rate | Recommended action |
|---|---:|---|
| Previous campaign was a success | **~90%** | Prioritize past converters in follow-up campaigns |
| Students / retired customers | ~75% / ~66% | Test dedicated offers for these segments |
| Customers under 30 vs. aged 40–50 | ~59% vs. ~40% | Pilot age-targeted messaging |
| Cellular contact vs. unknown contact type | ~54% vs. ~23% | Improve contact-data quality before the next campaign |

These are observed associations. Each recommendation should be validated with a controlled campaign test before rollout.

## Model: who to call first

Two models were trained with scikit-learn pipelines on a stratified 80/20 split (8,929 train / 2,233 test). **Call duration was deliberately excluded**, because it is only known after the call and would leak the outcome.

| Model | Accuracy | Precision (subscribers) | Recall (subscribers) | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 69.6% | 0.74 | 0.56 | 0.64 |
| **Random Forest** | **70.4%** | 0.72 | **0.61** | **0.66** |

**Recommendation:** Random Forest. On the test set it finds **52 more actual subscribers** (646 vs. 594) at the cost of 35 extra unnecessary calls. That's a good trade when a missed subscriber costs more than one extra call. The decision threshold should be tuned against real call costs and deposit value.

## Notebooks

1. [**Exploratory analysis**](notebooks/01_eda.ipynb): data checks, customer segments, contact channels, prior campaign outcomes
2. [**Modeling**](notebooks/02_modeling.ipynb): preprocessing pipeline, model comparison, confusion matrices, recommendations

## Run it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Download the Bank Marketing dataset (the 11,162-row `bank.csv` version with a `deposit` target, available on Kaggle) and save it as `data/bank.csv`. The notebooks also run in Google Colab: they mount Google Drive automatically, and you set `DATA_PATH` to your file's location.

## Limitations & next steps

- Treat `unknown` values as a data-quality issue in their own right, not as missing data to drop.
- Use cross-validation and a separate holdout for threshold tuning.
- Check model performance across customer groups for stability and fairness before any deployment.

## Author

**Antonio Jabbour**, Data Analyst · [GitHub](https://github.com/AntonioJabbour)

*Developed as a project following the IBM Data Science Professional Certificate.*
