import os
import json
import opendatasets as od

kaggle_json_path = r"C:\Users\maury\.kaggle\kaggle.json"

def get_kaggle_credentials(json_path):
    """
    Reads Kaggle credentials from a JSON file.
    
    :param json_path: Path to the Kaggle credentials JSON file.
    :return: Tuple containing Kaggle username and key.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Kaggle credentials file not found at {json_path}")

    with open(json_path) as f:
        kaggle_creds = json.load(f)
    
    return kaggle_creds.get('username'), kaggle_creds.get('key')

def download_dataset(dataset_url):
    """
    Downloads a dataset from Kaggle using the opendatasets library.
    
    :param dataset_url: URL of the Kaggle dataset you want to download.
    :param kaggle_json_path: Path to the Kaggle credentials JSON file.
    """
    # Get Kaggle credentials
    kaggle_username, kaggle_key = get_kaggle_credentials(kaggle_json_path)
    
    # Check if the Kaggle credentials were retrieved
    if not kaggle_username or not kaggle_key:
        raise ValueError("Kaggle credentials are missing in the JSON file.")

    # Set the environment variable to point to the directory containing kaggle.json
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(kaggle_json_path)

    # Download the dataset using the opendatasets library
    try:
        od.download(dataset_url)
        print("Dataset downloaded successfully.")
    except Exception as e:
        print("Failed to download the dataset.")
        print("Error:", str(e))

if __name__ == "__main__":
    # URL of the Kaggle dataset you want to download
    dataset_url = 'https://www.kaggle.com/competitions/titanic/data'

    # Download the dataset
    download_dataset(dataset_url)
