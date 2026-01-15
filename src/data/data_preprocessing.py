# data preprocessing

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging
from src.utils import load_params, download_nltk_package_if_needed

# Ensure necessary NLTK packages are available
download_nltk_package_if_needed("punkt", "tokenizers")
download_nltk_package_if_needed("stopwords", "corpora")

def preprocess_dataframe(df: pd.DataFrame, col: str)-> pd.DataFrame:
    """
    Preprocess a DataFrame by applying clean_and_normalized_text to a specific column.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        col (str): The name of the column containing text.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Initialize lemmatizer and stopwords
    stop_words = set(stopwords.words("english"))

    ## Text CLeaning function
    def clean_and_normalize_text(text: str)-> list:
        """This function clean the text by removing HTML tags, Punctuation, Numbers, 
            White space, Lower case, Tokenization, Stop word, Remove Contraction, 
            and normalize the text by Lemmatizating."""
        
        # Remove HTML tgs
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove Punctuation
        text = re.sub(r"(?!\B'\b)[%s]" % re.escape(string.punctuation.replace("'", "")), " ", text)
        # Remove Numbers and Digits
        text = re.sub(r'\d+', ' ', text)
        # Remove White space
        text = re.sub(' +', ' ', text).strip()
        # Lower case
        text = text.lower()
        # Tokenize the text
        text = text.split()
        # Remove Stop words
        text = [word for word in text if word not in stop_words]
        # Lemmatization
        normalized_tokens = [WordNetLemmatizer().lemmatize(word) for word in text]

        return " ".join(normalized_tokens)

    # Apply preprocessing to the specified column
    df[col] = df[col].apply(clean_and_normalize_text)

    # Drop rows with NaN values
    df = df.dropna(subset=[col])
    logging.info("Data pre-processing completed")
    return df


def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info('data loaded properly')

        # Fetch the preprocessing parameters
        params = load_params(params_path="params.yaml")
        preprocessing_column = params["data_preprocessing"]["preprocessing_column"]

        # Transform the data
        train_processed_data = preprocess_dataframe(train_data, preprocessing_column)
        test_processed_data = preprocess_dataframe(test_data, preprocessing_column)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "processed")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
