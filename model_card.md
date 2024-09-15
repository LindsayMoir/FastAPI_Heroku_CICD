# Model Card

## Model Details

- **Developer:** Lindsay Moir for Udacity Machine Learning DevOps Engineer course, Project FastAPI
- **Model Date:** September 30, 2024
- **Model Version:** 1.0
- **Model Type:** Gradient Boosting Classifier
- **Training Algorithms:** Gradient Boosting Classifier
- **Parameters:**
  - `n_estimators`: 200
  - `learning_rate`: 0.1
  - `max_depth`: 4
  - `min_samples_split`: 5
  - `min_samples_leaf`: 4
- **Features:**
  - Categorical features were one-hot encoded.
  - Class imbalance was handled using SMOTENC.
- **Fairness Constraints:** Not specified.
- **License:** [Model License](https://github.com/LindsayMoir/FastAPI_Heroku_CICD/blob/main/starter/model_card.md)

---

## Intended Use

- **Primary Intended Uses:**
  - Predicting whether an individual's salary is `<=50K` or `>50K` based on demographic and other related data.
- **Primary Intended Users:**
  - Data scientists, machine learning practitioners, and organizations interested in income prediction.
- **Out-of-Scope Use Cases:**
  - Predictions outside of income classification.
  - Use cases requiring real-time predictions in high-stakes environments where bias and fairness need rigorous auditing.

---

## Factors

- **Relevant Factors:**
  - Demographic factors such as age, work class, occupation, and marital status.
  - The model's performance may vary across different demographic groups.
- **Evaluation Factors:**
  - Evaluations considered the class distribution and potential biases in the dataset.
  - Cross-validation was used to ensure robustness across different data splits.

---

## Metrics

- **Model Performance Measures:**
  - Accuracy: 87%
  - F1-Score: 
    - Class 0: 0.91 
    - Class 1: 0.74 
    - Weighted Average: 0.87
- **Decision Thresholds:** The model uses default decision thresholds optimized by GridSearchCV.
- **Variation Approaches:** Cross-validation with `StratifiedKFold` to ensure consistent performance across different subsets of the data.

---

## Training Data

- **Datasets:** Train and test datasets were created from the original `census.csv` dataset.
- **Motivation:** The dataset's characteristics match the model's intended use cases.

---

## Evaluation Data

- **Datasets:** The dataset (`census.csv`) was provided by Udacity for this project.
- **Motivation:** The dataset was selected for its relevance to income prediction and demographic factors.
- **Preprocessing:**
  - Data was cleaned by removing duplicates, spaces, and irrelevant features.
  - Feature importance was calculated, leading to the removal of several features (`sex`, `native_country`, `fnlgt`, `race`).
  - The `education` feature was dropped as it provided the same information as the ordinal column `education-num`.
  - Categorical features were one-hot encoded, and class imbalance was addressed using SMOTENC.

---

## Metrics

- **Unitary Results:**
  - Accuracy, F1-Score, precision, recall, and ROC-AUC scores were calculated to evaluate model performance. 
  - The F1 score was used to choose the best model.
- **Intersectional Results:**
  - Performance metrics were analyzed across different demographic groups to assess fairness.
  - Data was sliced on all features and unique values within the features. Results of this data slicing can be found in `ml/slice_results.csv`.

---

## Ethical Considerations

- **Bias:**
  - Potential biases in demographic factors were considered. However, as the data is provided for learning purposes, there was no opportunity to address bias.
- **Fairness:**
  - The model may produce different outcomes for different demographic groups. 
  - Data was sliced on all demographic subcategories to reveal different performance characteristics.
  - Except for the `workclass` categories "Without-pay" and "Never-worked", there were no glaring examples of unfairness. These categories had no class scores for `>50K`, which aligns with the expectation that individuals not working would not have an income from work.
- **Transparency:**
  - Model decisions are based on features such as age, occupation, and work class, which should be carefully considered in context.

---

## Caveats and Recommendations

- **Model Limitations:** The model may not generalize well to populations significantly different from the training data.
- **Recommendations for Use:** 
  - Users should be aware of potential biases and ensure that the model is used in a context-appropriate manner.
  - Additional fairness constraints may be necessary for high-stakes applications.
