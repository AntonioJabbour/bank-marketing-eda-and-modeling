# Bank Marketing: Campaign Analysis & Subscription Prediction

A customer analytics case study exploring term deposit subscriptions, campaign response patterns, and the trade-off between finding potential subscribers and contacting non-subscribers.

**Tools:** Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn

## Business question

Which customer segments show higher subscription rates in this sample, and can a model help prioritize outreach?

The analysis combines descriptive customer segmentation with a comparison of Logistic Regression and Random Forest. Recommendations below are proposals for validation, rather than measured improvements to a live campaign.

## Explore the analysis

- [Exploratory analysis notebook](EDA_Project%28Marketing%29.ipynb): data checks, customer segments, contact channels, and prior campaign outcomes.
- [Modeling notebook](bank_marketing_modeling.ipynb): preprocessing, model comparison, confusion matrices, and recommendations.

## Data and KPI definition

The saved notebook outputs describe **11,162 records and 17 original columns**. The target is `deposit` (`yes` / `no`).

**Subscription rate** = records with `deposit = yes` ÷ records in the selected segment.

The sample subscription rate is approximately **47%**. This is a descriptive rate for the analyzed sample; its representativeness of a live campaign has not been established.

The original README identifies the data as a Portuguese bank marketing dataset obtained through Kaggle / UCI, but does not record the exact source URL or version. **The CSV is not included.** Reproducing these results requires the same 11,162-row `bank.csv` variant with the `deposit` target; other Bank Marketing variants may have different schemas and class distributions.

## Findings and business implications

Approximate rates below come from the saved EDA notebook.

| Observation in this sample | Proposed next step |
|---|---|
| Customers under 30 show about 59% subscription, compared with about 40% for ages 40–50. | Compare segment sizes and uncertainty before designing a segmented campaign test. |
| Students and retired customers show about 75% and 66% subscription, respectively. | Check sample sizes and whether the pattern holds in new campaign data. |
| Cellular contacts show about 54% subscription, versus about 23% for unknown contact types. | Investigate recording quality and customer mix before attributing the difference to the channel. |
| Previous campaign success is associated with roughly 90% subscription. | Validate whether this historical signal can support a controlled follow-up campaign. |

These are observational associations. They do not establish that changing a channel or targeting a demographic group will cause more subscriptions.

## Model comparison

Both models use an **80/20 stratified split** with `random_state=42`: 8,929 training records and 2,233 test records. Numeric scaling and categorical encoding are fitted within scikit-learn pipelines.

The selected 14 features **exclude call duration**, which would not be known before a call, and exclude the original target column.

| Model | Test accuracy | Subscriber precision | Subscriber recall | Subscriber F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 69.64% | 0.74 | 0.56 | 0.64 |
| Random Forest | 70.40% | 0.72 | 0.61 | 0.66 |

Values are transcribed from saved notebook outputs, not a new execution. Subscriber metrics are rounded as displayed in the classification reports.

The saved confusion matrix interpretations report **646 correctly identified subscribers** for Random Forest versus **594** for Logistic Regression, alongside **249 false positives** versus **214**. That is 52 more identified subscribers and 35 more false positives on this test set.

**Decision:** Random Forest is the stronger candidate if subscriber recall is the priority. Before operational use, compare contact costs and subscription value, tune the threshold on validation data, and evaluate on a fresh holdout. This comparison does not demonstrate revenue uplift.

## Run the notebooks

The existing notebooks use **Google Colab and a Google Drive CSV path**.

1. Open each notebook in Google Colab.
2. Obtain the matching dataset described above and place `bank.csv` in a Google Drive folder.
3. Run the Drive mount cell and authorize access.
4. Change the `pd.read_csv(...)` path in each notebook to your CSV location.
5. Run the EDA notebook, then the modeling notebook, from top to bottom.

Imports used by the notebooks: `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`, plus Colab's `google.colab` module. Dependency versions are not recorded, and the notebooks have not been independently rerun as part of this documentation update.

## Limitations and next steps

- Recover and document the exact dataset source, version, and usage terms.
- Validate segment rates with sample counts and uncertainty estimates.
- Treat `unknown` values as data quality questions; they are not equivalent to missing Python values.
- Check the prediction-time availability of all retained features, including current-campaign contact information.
- Use cross-validation and a separate final holdout when tuning models or thresholds.
- Assess stability and fairness across customer groups before any deployment.

## Author

Antonio Jabbour

Developed as a learning project following the IBM Data Science Professional Certificate.
