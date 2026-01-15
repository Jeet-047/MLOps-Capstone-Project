import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.utils import load_params, save_model
from src.logger import logging

# Initialize the train_model function
def train_model(train_data: pd.DataFrame, tfidf_params: dict, clf_params: dict)-> LogisticRegression:
    """This function takes train data, required parameters and apply TF-IDF vectorizer and 
    train the model using LogisticRegression model.
    Returns:
        LogisticRegression: an trained model pkl file.
    """
    logging.info("Start the model training...")
    # Segregate the dataset: make sure X_train is a 1D iterable of text documents
    X_train_df = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    # If multiple input columns, join them into a single text per row.
    if X_train_df.shape[1] > 1:
        X_train = X_train_df.astype(str).agg(" ".join, axis=1)
    else:
        # squeeze to a Series (ensures vectorizer sees rows, not columns)
        X_train = X_train_df.iloc[:, 0].astype(str)

    logging.info("Training samples: %d", len(X_train))

    logging.info("Setup the training pipeline")
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                max_df=tfidf_params["max_df"],
                min_df=tfidf_params["min_df"],
                ngram_range=tuple(tfidf_params["ngram_range"]),
                sublinear_tf=tfidf_params["sublinear_tf"]
            )),
            ("clf", LogisticRegression(
                C=clf_params["C"],
                class_weight=clf_params["class_weight"],
                max_iter=clf_params["max_iter"],
                solver=clf_params["solver"]
            ))
        ]
    )

    logging.info("Start training the model.")
    pipeline.fit(X_train, y_train)

    logging.info("Model training completed, returning trained pipeline.")
    return pipeline

# Initialize the main function
def main():
    try:
        logging.info("Fetch the data from data.processed")
        train_df = pd.read_csv("./data/processed/train_processed.csv")
        
        logging.info("Fetch the model building parameters")
        params = load_params("params.yaml")
        tfidf_params = params["model_building"]["tfidf"]
        clf_params = params["model_building"]["clf"]

        # Call the model training function
        trained_model = train_model(train_df, tfidf_params, clf_params)
        logging.info("Model trained successfully!!")

        # Save the trained model in the models directory
        save_model(trained_model, "models/model.pkl")
        logging.info("Model saved successfully!!")

    except Exception as e:
        logging.error(f"Error during model training. Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()