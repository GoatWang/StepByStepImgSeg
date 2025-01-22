import os
import requests
from zipfile import ZipFile

def download_and_extract_coco(dataset_type, save_dir):
    """
    Downloads and extracts the COCO dataset.
    
    Args:
        dataset_type (str): The type of dataset to download ('train2017', 'val2017', 'test2017', etc.).
        save_dir (str): Directory where the dataset will be saved.
    """
    base_url = "http://images.cocodataset.org"
    datasets = {
        "train2017": f"{base_url}/zips/train2017.zip",
        "val2017": f"{base_url}/zips/val2017.zip",
        "test2017": f"{base_url}/zips/test2017.zip",
        "annotations": f"{base_url}/annotations/annotations_trainval2017.zip",
    }
    
    if dataset_type not in datasets:
        print(f"Dataset type '{dataset_type}' is not valid. Available options: {list(datasets.keys())}")
        return
    
    url = datasets[dataset_type]
    os.makedirs(save_dir, exist_ok=True)
    zip_path = os.path.join(save_dir, f"{dataset_type}.zip")
    
    # Download the dataset
    print(f"Downloading {dataset_type} from {url}...")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    print(f"Downloaded {dataset_type} to {zip_path}")
    
    # Extract the dataset
    print(f"Extracting {zip_path}...")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    print(f"Extracted {dataset_type} to {save_dir}")
    
    # Optionally delete the zip file
    os.remove(zip_path)
    print(f"Deleted {zip_path}")

if __name__ == "__main__":
    # Set the dataset type and directory
    # dataset_type = "val2017"  # Options: 'train2017', 'val2017', 'test2017', 'annotations'
    save_dir = "../coco_dataset"
    
    # Download and extract the dataset
    # download_and_extract_coco("train2017", save_dir)
    download_and_extract_coco("val2017", save_dir)
    # download_and_extract_coco("test2017", save_dir)
    download_and_extract_coco("annotations", save_dir)
    

