import torch
import json
from dataset import ImageQADataset
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForVision2Seq, TrainingArguments, Trainer

def load_dataset(meta_fps, tokenizer):
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
    train_dataset = ImageQADataset(train_metadata, tokenizer)
    val_dataset = ImageQADataset(val_metadata, tokenizer)
    test_dataset = ImageQADataset(test_metadata, tokenizer)
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    # Path to metadata files
    meta_fps = [
        "coco_dataset/val2017_2_task1_qapair.json",
        "coco_dataset/val2017_2_task2_qapair.json"
    ]

    # Load the LLaVA model and tokenizer
    model_name = "llava-hf/llava-1.5-7b-hf"
    cache_directory = "/notebooks/.cache/huggingface/hub"
    tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_directory)
    model = AutoModelForVision2Seq.from_pretrained(model_name, cache_dir=cache_directory)

    # Load dataset
    train_dataset, val_dataset, test_dataset = load_dataset(meta_fps, tokenizer)
    print("len(train_dataset): ", len(train_dataset))
    print("len(val_dataset): ", len(val_dataset))
    print("len(test_dataset): ", len(test_dataset))

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
        report_to="tensorboard"
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("./llava-qa-model")
    tokenizer.save_pretrained("./llava-qa-model")
