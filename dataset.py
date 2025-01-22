import os
import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# TODO: implement the transform augmentation
# TODO: implement the task drawing in every batch

# Define image transformations (if required)
image_transform = transforms.Compose([
    transforms.Resize((336, 336)),  # Resize images
    transforms.ToTensor(),          # Convert to PyTorch tensor
])

# PyTorch Dataset Class
class ImageQADataset(Dataset):
    def __init__(self, metadata, tokenizer): # , transform=None # TODO: when batch transform is implemented, the
        """
        Args:
            metadata (list): A list of dictionaries with keys 'image_path', 'question', and 'answer'.
            transform (callable, optional): Transform to be applied to each image.
        """
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.transform = image_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = item["image_path"]
        question = self.tokenizer(
            item["question"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        answer = self.tokenizer(
            item["answer"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        # Load and transform the image
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

        return {"image": image, "question": question, "answer": answer}


if __name__ == "__main__":
    # Path to metadata files
    meta_fps = [
        "coco_dataset/val2017_2_task1_qapair.json",
        "coco_dataset/val2017_2_task2_qapair.json"
    ]

    # Load metadata from JSON files
    metadata = []
    for meta_fp in meta_fps:
        with open(meta_fp, "r") as f:
            metadata.extend(json.load(f))  # Assuming metadata is a list of {image_path, question, answer}

    # Split metadata into train, validation, and test sets
    train_ratio, val_ratio = 0.8, 0.1
    num_samples = len(metadata)
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))

    train_metadata = metadata[:train_end]
    val_metadata = metadata[train_end:val_end]
    test_metadata = metadata[val_end:]

    # Create Dataset objects
    train_dataset = ImageQADataset(train_metadata, transform=image_transform)
    val_dataset = ImageQADataset(val_metadata, transform=image_transform)
    test_dataset = ImageQADataset(test_metadata, transform=image_transform)

    # Create DataLoader objects
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Example usage in a training loop
    for batch in train_loader:
        images = batch["image"]      # Tensor of shape (batch_size, 3, 336, 336)
        questions = batch["question"]  # List of questions
        answers = batch["answer"]      # List of answers

        # Perform your training logic here
        print("images.shape", images.shape)
        print("questions[0]", questions[0])
        print("answers[0]", answers[0])
        break
