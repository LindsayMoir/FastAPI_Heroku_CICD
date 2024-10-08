# model.py
"""
This is the module that reads the data, cleans it, feature engineers it,
trains the model,  performs inference on the test set, visualizes the
feature importance, and produces quality metrics.
"""

# import libraries
import argparse
from datetime import datetime
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import yaml

config = None


def setup_env(working_directory):
    """
    Sets up the environment by changing the working directory to the provided
    path and loading the configuration from the 'config/config.yaml' file. It
    also sets up logging with the specified log file.

    Args:
        working_directory (str): The path to the working directory.

    Returns:
        config (dict): The loaded configuration.

    Raises:
        FileNotFoundError: If the 'config/config.yaml' file is not found.
    """
    global config

    # Change the working directory to the one provided via command line or
    os.chdir(working_directory)

    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Extract the log file path from the configuration
    log_file_path = config['files']['log_file']

    # Setup logging
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        force=True,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s'
    )

    # Access the root logger and retrieve its handlers
    root_logger = logging.getLogger()
    handlers = root_logger.handlers

    # Print the absolute path of the log file by inspecting the FileHandler
    for handler in handlers:
        if isinstance(handler, logging.FileHandler):
            print(f"Log file path set to: {handler.baseFilename}")
        else:
            print("No file handler found in the root logger")

    return config


def load_data():
    """Reads the data from the data source.

    Returns:
        pandas.DataFrame: The loaded data.
    """
    data = pd.read_csv(config['data']['raw_data_path'])

    return data


def clean_data(df):
    """
    Clean the given DataFrame by performing the following steps:

    1. Drop duplicates from the DataFrame.
    2. Remove spaces from column names.
    3. Remove spaces from values in the columns.
    4. Replace '-' with '_' in column names.
    5. Drop 'education', 'sex', 'native_country', 'fnlgt', and 'race' columns.
    6. Write the cleaned data to a new file.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be cleaned.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.

    """

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    logging.info(
        f"SUCCESS: Dropped duplicates. \nThe shape of the data is: {df.shape}")

    # Remove all spaces from the column names
    df.columns = df.columns.str.replace(' ', '')

    # Remove all spaces from the values in the columns
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Also replace the '-' with '_'
    df.columns = df.columns.str.replace('-', '_')

    # Drop the education column and columns that were not important based on
    # feature importance
    df.drop(columns=['education', 'sex', 'native_country', 'fnlgt', 'race'],
            axis=1, inplace=True)

    # Write the cleaned data to a new file
    df.to_csv(config['data']['cleaned_data_path'], index=False)

    return df


def feature_engineering(data):
    """
    Sets up and runs the pipeline for feature engineering and model training.

    Parameters:
        data (pandas.DataFrame): The input data containing features and target
        variable.

    Returns:
        tuple: A tuple containing the preprocessed features, encoded target
        variable, and the pipeline.

    Raises:
        None

    Example:
        X, y_encoded, pipeline = feature_engineering(data)
    """

    # Identify categorical features excluding the target variable
    categorical_features = data.select_dtypes(
        include='object').columns.tolist()
    categorical_features.remove('salary')  # Exclude the target variable

    # Separate features and target variable
    X = data.drop(columns=['salary'])
    y = data['salary']

    # Label encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Define preprocessing for categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'),
             categorical_features)
             ],
        remainder='passthrough'  # Passes quantitative features as they are
    )

    # SMOTENC expects the indices of the categorical features, not their names
    categorical_indices = [X.columns.get_loc(
        col) for col in categorical_features]

    # Create the pipeline with SMOTENC
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTENC(categorical_features=categorical_indices,
                          random_state=42)),
        ('classifier', GradientBoostingClassifier())
    ])

    # Save the label encoder
    joblib.dump(label_encoder, config['models']['label_encoder'])

    return X, y_encoded, pipeline


def find_best_model(X, y_encoded, pipeline):
    """
    Perform train/test split, k-fold cross-validation with stratification, and
    return the best model.

    Parameters:
        X (array-like): The input features.
        y_encoded (array-like): The encoded target variable.
        pipeline (object): The pipeline object containing the classifier.

    Returns:
        tuple: A tuple with the test features and test target variables.

    Raises:
        None
    """

    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'classifier__n_estimators': config['hyperparameters']
                                          ['n_estimators'],
        'classifier__learning_rate': config['hyperparameters']
                                           ['learning_rate'],
        'classifier__max_depth': config['hyperparameters']
                                       ['max_depth'],
        'classifier__min_samples_split': config['hyperparameters']
                                               ['min_samples_split'],
        'classifier__min_samples_leaf': config['hyperparameters']
                                              ['min_samples_leaf']
    }

    # Define stratified k-fold cross-validation
    stratified_kfold = StratifiedKFold(
        n_splits=4, shuffle=True, random_state=42)

    # Perform GridSearchCV with stratification
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               cv=stratified_kfold,
                               scoring='f1',  # Evaluation metric
                               n_jobs=-1,  # Use all available cores
                               verbose=4)  # Verbosity level

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    logging.info(f"SUCCESS: Best model found with the following parameters: \
    {grid_search.best_params_}")

    # Extract the fitted SMOTENC and transform the training data
    smote = best_model.named_steps['smote']
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Log the shape of the resampled data
    logging.info(f"Shape of X_resampled after SMOTENC: {X_resampled.shape}")
    logging.info(f"Shape of y_resampled after SMOTENC: {y_resampled.shape}")
    logging.info(
        f"Percentage of 0s: {y_resampled.sum() / y_resampled.shape[0]}")
    logging.info(
        f"Percentage of 1s: \
        {(y_resampled.shape[0] - y_resampled.sum()) / y_resampled.shape[0]}")

    # Write the resampled data to disk. We will use it for data distribution
    # monitoring.
    pd.DataFrame(X_resampled).to_csv(
        config['data']['resampled_data'], index=False)

    # Save the best model
    joblib.dump(best_model, config['models']['best_model'])

    return X_test, y_test


def inference_batch(X_test, y_test):
    """
    This function performs inference on the test set using the trained model.

    Parameters:
    - X_test: DataFrame containing the dependent features of the test set.
    - y_test: The target variable of the test set.

    Returns:
    - predictions: The predicted labels for the test set.
    - best_model: The best model that has been loaded from disk.
    """

    # Load the best model from disk
    best_model = joblib.load(config['models']['best_model'])

    # Perform predictions
    predictions = best_model.predict(X_test)

    # Log the predictions and classification report
    logging.info("SUCCESS: Predictions made on the test set")
    logging.info("SUCCESS: Classification Report:")
    logging.info(f"{classification_report(y_test, predictions)}")

    return predictions, best_model


def inference(features):
    """
    Performs inference on a single data sample using the trained model.

    Parameters:
    - features: Dictionary containing the dependent features.

    Returns:
    - prediction: The predicted label for the input data.
    """
    # We are running this standalone. It needs the config file.
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load the best_model, pipeline, and label encoder from disk
    best_model = joblib.load(config['models']['best_model'])
    label_encoder = joblib.load(config['models']['label_encoder'])

    # Convert features to DataFrame
    features_df = pd.DataFrame([features])

    # Perform prediction
    prediction_encoded = best_model.predict(features_df)

    # Decode the prediction
    prediction = label_encoder.inverse_transform(prediction_encoded)

    return prediction[0]


def feat_import(X, best_model):
    """
    Calculates and visualizes the feature importance of the best model.

    Parameters:
    X (DataFrame): The input data containing the features.
    best_model (Pipeline): The trained pipeline model.

    Returns:
    None

    Raises:
    None

    Writes:
    - A bar chart showing the feature importance to the 'ml' directory.
    """

    # Calculate the feature importance of the best model
    importances = best_model.named_steps['classifier'].feature_importances_

    # Get the preprocessor from the pipeline
    preprocessor = best_model.named_steps['preprocessor']

    # Get feature names for categorical features
    categorical_features = preprocessor.transformers_[
        0][2]  # This is the categorical features list
    categorical_transformer = preprocessor.named_transformers_['cat']
    categorical_feature_names = list(
        categorical_transformer.get_feature_names_out(categorical_features))

    # Get numeric feature names from X
    numerical_features = X.select_dtypes(include='number').columns.tolist()

    # Combine the feature names
    all_feature_names = categorical_feature_names + numerical_features

    # Create a DataFrame to map feature importance to feature names
    feature_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    })

    # Collapse the feature importance by summing the importance of one-hot
    # encoded features
    def collapse_feature_name(name):
        for original_feature in categorical_features:
            if name.startswith(original_feature):
                return original_feature
        return name

    feature_importance_df['Collapsed Feature'] = feature_importance_df[
        'Feature'].apply(collapse_feature_name)

    collapsed_importance = feature_importance_df.groupby(
        'Collapsed Feature').sum().reset_index()

    # Sort by importance
    collapsed_importance = collapsed_importance.sort_values(
        'Importance', ascending=False)

    # Create a bar chart to visualize the feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(collapsed_importance['Collapsed Feature'],
            collapsed_importance['Importance'])
    plt.xticks(rotation=60)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(config['models']['feature_importance'])


def quality_metrics(y_test, predictions):
    """
    Produces the confusion matrix, classification report, and ROC curve.

    Parameters:
    - y_test: The true labels of the test set.
    - predictions: The predicted labels for the test set.

    Returns:
    None

    Writes:
    - A confusion matrix and ROC curve to the 'ml' directory.
    """

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=False, cmap='Blues',
                     xticklabels=['Predicted  <=50K', 'Predicted  >50K'],
                     yticklabels=['True  <=50K', 'True  >50K'])

    # Annotate the numbers on the heatmap with green color
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(j + 0.5, i + 0.5, str(cm[i][j]),
                    ha='center', va='center', color='green')

    # Set title
    plt.title('Confusion Matrix')

    # Adjust layout and save/show the plot
    plt.tight_layout()
    plt.savefig(config['models']['confusion_matrix'])

    # Calculate ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    auc_score = roc_auc_score(y_test, predictions)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(config['models']['roc_curve'])


def evaluate_model_slices():
    """
    This function outputs the performance of the model on slices of the data
    and stores the results in a DataFrame.

    Parameters:
    - None

    Returns:
    - results_df: A DataFrame containing performance metrics for each slice.

    Writes:
    - A CSV file containing the results to the 'ml' directory.
    """
    # Load the cleaned_data from disk
    data = pd.read_csv(config['data']['cleaned_data_path'])

    # Change the target variable to binary
    data['salary'] = data['salary'].apply(lambda x: 0 if x == '<=50K' else 1)

    # Load the best model from disk
    best_model = joblib.load(config['models']['best_model'])

    # Get the preprocessor from the pipeline
    preprocessor = best_model.named_steps['preprocessor']

    # Get feature names for categorical features
    categorical_features = preprocessor.transformers_[
        0][2]  # This is the categorical features list

    # Initialize a list to store the results
    results = []

    # Iterate over each categorical feature
    for feature in categorical_features:
        # Get unique values of the feature
        unique_values = data[feature].unique()

        # Iterate over each unique value
        for value in unique_values:
            # Create a slice of the data with the specific value of the feature
            slice_data = data[data[feature] == value]

            # Separate features and target variable
            X_slice = slice_data.drop(columns=['salary'])
            y_slice = slice_data['salary']

            # Perform predictions on the slice
            predictions = best_model.predict(X_slice)

            # Calculate performance metrics
            report = classification_report(
                y_slice, predictions, output_dict=True)
            try:
                auc = roc_auc_score(y_slice, predictions)
            except ValueError as e:
                logging.info(
                    f"Error occurred on {feature}, {value}: \n {str(e)}")
                auc = 0

            # Append results to the list
            results.append({
                'feature': feature,
                'value': value,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1-score': report['weighted avg']['f1-score'],
                'support': report['weighted avg']['support'],
                'roc_auc_score': auc,
                'accuracy': report['accuracy']
            })

    # Convert the results list into a DataFrame
    results_df = pd.DataFrame(results)

    # Round all numeric results to 3 decimal places
    results_df = results_df.round(3)

    # Sort the DataFrame by f1-score
    results_df = results_df.sort_values(by='f1-score', ascending=False)

    # Write resuls to disk
    results_df.to_csv(config['models']['slice_results'],
                      sep='\t',
                      index=False)

    return results_df


def log_end(config, start_time):
    """
    This function logs the end of the run and updates the CSV file with the end
    time and elapsed time. Both formatted and unformatted start and end times
    are written to the CSV.

    Args:
        config (dict): Configuration dictionary containing file paths.
        start_time (datetime): The start time of the run.

    Writes:
        - A new CSV file with the start time, end time, and elapsed time in the
        'ml' directory.

    Returns:
        end_time (datetime): The end time of the run.
        elapsed_time (timedelta): The total elapsed time of the run.
    """

    # File path for the CSV file in the 'ml' directory
    file_path = config['files']['run_log']

    # Get the current time as the end time
    end_time = datetime.now()

    # Compute elapsed time as the difference between end_time and start_time
    elapsed_time = end_time - start_time

    # Store the start and end times in ISO 8601 format for future processing
    start_time = start_time.isoformat()
    end_time = end_time.isoformat()

    # Check if the file exists
    if os.path.exists(file_path):
        # Read the existing CSV file
        df = pd.read_csv(file_path)

        # Append the end time and elapsed time to the DataFrame
        new_row = {
            'start_time': start_time,
            'end_time': end_time,
            'elapsed_time': elapsed_time
        }

        # Append new row to the DataFrame
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Save the updated DataFrame back to the CSV file
        df.to_csv(file_path, index=False)

    else:
        # Create a new DataFrame with the start time and end time
        df = pd.DataFrame({
            'start_time': [start_time],
            'end_time': [end_time],
            'elapsed_time': [elapsed_time]
        })

        # Save the DataFrame to a new CSV file
        df.to_csv(file_path, index=False)

    return end_time, elapsed_time


if __name__ == "__main__":

    # Log the start of the run
    start_time = datetime.now()
    logging.info(f"SUCCESS: Run started at: {datetime.now()}")

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Set up the working environment for the model.')
    parser.add_argument('working_directory', type=str, nargs='?',
                        default='/mnt/d/GitHub/FastAPI_Heroku_CICD/',
                        help='The path to the working directory')
    args = parser.parse_args()

    # Setup the environment
    config = setup_env(args.working_directory)
    logging.info(
        f'SUCCESS: Current working directory is set to: {os.getcwd()}')

    # Read the data
    data = load_data()
    logging.info(
        f"SUCCESS: Data loaded from {config['data']['raw_data_path']}")
    logging.info(f"The shape of the data is: {data.shape}")

    # Clean the data
    data = clean_data(data)
    logging.info(
        f"SUCCESS: Cleaned data written to \
    {config['data']['cleaned_data_path']}")
    logging.info(f"The shape of the data is: {data.shape}")

    # Feature engineer the data
    df = pd.read_csv(config['data']['cleaned_data_path'])
    X, y_encoded, pipeline = feature_engineering(df)
    logging.info("SUCCESS: Feature engineering pipeline set up.")

    # Find the best model
    X_test, y_test = find_best_model(X, y_encoded, pipeline)
    logging.info(f"SUCCESS: Best model found and saved to \
    {config['models']['best_model']}")

    # Perform inference on the test data
    predictions, best_model = inference_batch(X_test, y_test)
    logging.info("SUCCESS: Inference completed on the test set.")

    # Perform inference on a single example
    dict = {'age': 39,
            'workclass': 'Local-gov',
            'education_num': 13,
            'marital_status': 'Never-married',
            'occupation': 'Adm-clerical',
            'relationship': 'Not-in-family',
            'capital_gain': 2174,
            'capital_loss': 0,
            'hours_per_week': 40
            }
    single_prediction = inference(dict)
    # Log the prediction
    logging.info(f"SUCCESS: Prediction made for single row of input: {dict} \
is: {single_prediction}")

    # Calculate feature importance
    feat_import(X, best_model)
    logging.info("SUCCESS: Feature importance calculated and visualized.")

    # Quality metrics
    quality_metrics(y_test, predictions)
    logging.info("SUCCESS: Quality metrics calculated and visualized.")

    # Evaluate model slices
    logging.info("SUCCESS: Model slices evaluated.")
    results_df = evaluate_model_slices()

    # Log the end of the run
    end_time, elapsed_time = log_end(config, start_time)
    logging.info(f"SUCCESS: Run ended at: \
{end_time} with an elapsed time of: {elapsed_time}")
