import pandas as pd
import json
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.utils import load_model, load_params

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# # Setup the MlFlow
# dagshub_url = "https://dagshub.com"
# repo_owner = "Jeet-047"
# repo_name = "MLOps-Capstone-Project"

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

def evaluate_model(x_test: pd.DataFrame, y_test: pd.Series, model: Pipeline) -> dict:
    """This function takes the following arguments and evaluate the model.
    Args:
        x_test (pd.DataFrame): The test independent feature.
        y_test (pd.Series: The test dependent feature.
        model (Pipeline): The model pickle file, contains both vectorizer and classifier integrated.
    Returns:
        dict: The evaluation metrics.
    """
    logging.info("Start evaluation the model...")
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)[:, 1]

    logging.info("  Calculating the evaluation metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }
    logging.info("  Model evaluation metrics calculated.")

    return metrics

def set_alias(model_name: str) -> None:
    """Set the latest registered model alias to "staging" """
    try:
        client = MlflowClient()
        version = client.get_latest_versions(model_name)[0].version
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=version
        )
    except Exception as e:
        logging.error(f"Error during alias setting. Error: {e}", exc_info=True)

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise


# Initialize the main function
def main():
    """
    This function execute steps one-by-one for model evaluation to logging to the MLFlow and also register it.
    """
    mlflow.set_tracking_uri('https://dagshub.com/Jeet-047/MLOps-Capstone-Project.mlflow')
    dagshub.init(repo_owner='Jeet-047', repo_name='MLOps-Capstone-Project', mlflow=True)
    mlflow.set_experiment("my-dvc-pipeline")
    with mlflow.start_run() as run: # Start MlFlow run
        try:
            logging.info("Initialize the parameters for model registering and alias.")
            params = load_params("params.yaml")
            logged_model_name = params["model_eval_register"]["logged_model_name"]
            registered_model_name = params["model_eval_register"]["registered_model_name"]

            logging.info("Fetching the trained model & test data...")
            model = load_model("./models/model.pkl")
            test_df = pd.read_csv("./data/processed/test_processed.csv")

            logging.info("Splitting the test data..")
            X_test = test_df.iloc[:, :-1].squeeze()
            y_test = test_df.iloc[:, -1]

            # Evaluate the model
            metrics = evaluate_model(X_test, y_test, model)
            
            logging.info("Saving the evaluation metrics...")
            save_metrics(metrics, "./reports/metrics.json")

            logging.info("Logging all model information to MlFlow...")
            mlflow.log_metrics(metrics) # Log metrics
            if hasattr(model, 'get_params'): # Log parameters
                params = model.get_params()
                mlflow.log_params(params)
            mlflow.sklearn.log_model( # Log and register model
                sk_model=model,
                name=logged_model_name,
                registered_model_name=registered_model_name
            )
            logging.info("Successfully, complete the model evaluation process!")

            logging.info("Start the model alias defining process.")
            set_alias(registered_model_name)    # Assign alias to the model

            logging.info("Model logged, registered, and staged successfully.")
        except Exception as e:
            logging.error(f"Error during model evaluation. With error: {e}", exc_info=True)


if __name__ == "__main__":
    main()