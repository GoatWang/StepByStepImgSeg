import torch
import json
from dataset import ImageQADataset
from transformers import AutoTokenizer, AutoModelForVision2Seq, TrainingArguments, Trainer

def load_dataset():
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
    train_dataset = ImageQADataset(train_metadata)
    val_dataset = ImageQADataset(val_metadata)
    test_dataset = ImageQADataset(test_metadata)
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    # Load the LLaVA model and tokenizer
    model_name = "LLaVA-pretrained-model-name"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name)

    # Load dataset
    train_dataset, val_dataset, test_dataset = load_dataset()
    # Preprocessing function
    def preprocess_data(example):
        inputs = tokenizer(
            example["question"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        outputs = tokenizer(
            example["answer"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs["pixel_values"] = example["image"]  # Assuming PIL Image format
        inputs["labels"] = outputs["input_ids"]
        return inputs

    # Preprocess the dataset
    tokenized_dataset = dataset.map(preprocess_data, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./llava-qa-model",
        evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        fp16=True,  # Use mixed precision for faster training if supported
        report_to="tensorboard",
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("./llava-qa-model")
    tokenizer.save_pretrained("./llava-qa-model")
