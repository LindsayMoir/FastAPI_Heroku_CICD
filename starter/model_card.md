# Model Card
Very strange!

## Model Details
Person or organization developing model: Lindsay Moir for Udacity Machine Learning DevOps Engineer course, Project FastAPI
Model date: September 30, 2024
Model version: 1.0
Model type: Gradient Boosting Classifier
Training Algorithms: Gradient Boosting Classifier.
Parameters:
n_estimators: 200
learning_rate: 0.1
max_depth: 4
min_samples_split: 5
min_samples_leaf: 4
Features:
Categorical features were one-hot encoded.
Class imbalance was handled using SMOTENC.
Fairness Constraints: Not specified.
License: https://github.com/LindsayMoir/FastAPI_CICD_Machine_Learning/blob/master/LICENSE.txt

## Intended Use
Primary intended uses:
Predicting whether an individual's salary is <=50K or >50K based on demographic and other related data.
Primary intended users:
Data scientists, machine learning practitioners, and organizations interested in income prediction.
Out-of-scope use cases:
Predictions outside of income classification.
Use cases requiring real-time predictions in high-stakes environments where bias and fairness need rigorous auditing.
Factors
Relevant factors:
Demographic factors such as age, work class, occupation, and marital status.
The model's performance may vary across different demographic groups.
Evaluation factors:
Evaluations considered the class distribution and potential biases in the dataset.
Cross-validation was used to ensure robustness across different data splits.

## Metrics
Model performance measures:
Accuracy: 87%
F1-Score: Class 0 .91, Class 1 .74, Weighted Average .83
Decision thresholds:
The model uses default decision thresholds optimized by GridSearchCV.
Variation approaches:
Cross-validation with StratifiedKFold to ensure consistent performance across different subsets of the data.
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

## Training Data
Datasets:
Train and test datasets were created from the original census.csv dataset. 
Motivation:
The dataset's characteristics match the model's intended use cases.

## Evaluation Data
Datasets:
The dataset (census.csv) was provided by Udacity for this project.
Motivation:
The dataset was selected for its relevance to income prediction and demographic factors.
Preprocessing:
Data was cleaned by removing duplicates, spaces, and irrelevant features. Feature importance was calcluated and as a 
results a number of the features were removed ('sex', 'native_country', 'fnlgt', 'race'). education was also dropped 
since it was identical to the ordinal column education-num in terms of the information that it provided to the model.
Categorical features were one-hot encoded, and class imbalance was addressed using SMOTENC.

## Metrics
Unitary results:
Accuracy, F1-Score, precision, recall, and ROC-AUC scores were calculated to evaluate model performance. I used f1 score 
to choose the best model.
Intersectional results:
Performance metrics were analyzed across different demographic groups to assess fairness. Data was sliced on all features and 
unique values within the feature. The results of this data slicing can be found in ml/slice_results.csv.

## Ethical Considerations
Bias:
Potential biases in demographic factors were considered. Since the data is provided as project data for learning purposes 
there is no opportunity to address bias.
Fairness:
The model may produce different outcomes for different demographic groups; this was acknowledged, but no specific fairness constraints were applied. Data was sliced on all demographic subcategories to reveal different performance characteristics. 
Except for workclass, Without-pay and workclass, Never-worked there were no glaring examples of unfairness. These had no class 
scores for >50K. This makes sense as somebody not working would not  be making any income from work.
Transparency:
Model decisions are based on features such as age, occupation, and work class, which should be carefully considered in context.

## Caveats and Recommendations
Model Limitations:
The model may not generalize well to populations significantly different from the training data.
Recommendations for use:
Users should be aware of potential biases and ensure that the model is used in a context-appropriate manner.
Additional fairness constraints may be necessary for high-stakes applications.
