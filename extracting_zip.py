import zipfile
import os
import argparse

def extract_zip(zip_file_path):
    """
    Extracts a zip file to a new folder named after the zip file (without extension)
    in the same directory as the zip file.

    :param zip_file_path: Path to the zip file to be extracted.
    :return: Path to the directory where the files were extracted.
    """
    # Get the base name of the zip file (without extension)
    base_name = os.path.splitext(os.path.basename(zip_file_path))[0]

    # Determine the directory to extract to
    extract_to_directory = os.path.join(os.path.dirname(zip_file_path), base_name)

    # Ensure the new folder exists
    os.makedirs(extract_to_directory, exist_ok=True)

    # Open the zip file and extract its contents into the new folder
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_directory)

    return extract_to_directory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a zip file to a folder named after the zip file.")
    parser.add_argument("zip_file_path", type=str, help="The path to the zip file.")
    
    args = parser.parse_args()
    
    extracted_dir = extract_zip(args.zip_file_path)
    print(f"Extracted to: {extracted_dir}")  #python extracting_zip.py "C:\Users\maury\Downloads\playground-series-s4e9\playground-series-s4e9.zip"