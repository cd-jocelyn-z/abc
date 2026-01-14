import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import evaluate
import numpy as np
import argparse

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
import regex as re

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# python nllb-bre-fra-lora.py --train data/train.jsonl --dev data/dev.jsonl --test data/test.jsonl --output-dir "encoder_crossattn_full_3ep" --config module_configs.json --lora-config "encoder_crossattn" --epoch 3

def get_args():
    parser = argparse.ArgumentParser(description="Train NLLB Breton-French MT model")
    parser.add_argument('--train', type=str, required=True, help='Path to training data')
    parser.add_argument('--dev', type=str, required=True, help='Path to dev data')
    parser.add_argument('--test', type=str, required=True, help='Path to test data')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--lora-config', type=str, required=True,
                        help='Which LoRA config to use from config file (all_modules, encoder_only, encoder_crossattn, q_v_only)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config file if specified)')
    args = parser.parse_args()
    return args


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_data(path: str) -> list[dict[str, str | list]]:
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} examples from: {path}")
    return data


def format_input_data(data):
    """Format data for Datasets - automatically detects structure"""
    sources = []
    targets = []

    for example in data:
        # Detect structure automatically
        if "translation" in example:
            # OFP Arbre format - nested translation with language codes
            translation = example["translation"]
            br = translation.get("bre_Latn") or translation.get("br")
            fr = translation.get("fra_Latn") or translation.get("fr")
        else:
            # Standard format
            br = example.get("br") or example.get("bre_Latn")
            fr = example.get("fr") or example.get("fra_Latn")

        # Handle expansion for multi-reference data
        if isinstance(br, list) and isinstance(fr, list):
            for bre in br:
                sources.append(bre)
                targets.append(fr)
        else:
            sources.append(br[0] if isinstance(br, list) else br)
            targets.append(fr[0] if isinstance(fr, list) else fr)

    return {"source": sources, "target": targets}

def build_datasets(train_path, dev_path, test_path):

    train_data = load_data(train_path)
    dev_data = load_data(dev_path)
    test_data = load_data(test_path)

    print("=" * 50)
    print("DEBUG - First example structure:")
    print(train_data[0])
    print("=" * 50)
    print(f"Train examples: {len(train_data)}")
    print(f"Dev examples: {len(dev_data)}")
    print(f"Test examples: {len(test_data)}")

    train_dataset = Dataset.from_dict(format_input_data(train_data))
    dev_dataset = Dataset.from_dict(format_input_data(dev_data))
    test_dataset = Dataset.from_dict(format_input_data(test_data))

    def clean_example(example):
        example["source"] = clean_text(example["source"])
        if isinstance(example["target"], list):
            example["target"] = [clean_text(t) for t in example["target"]]
        else:
            example["target"] = clean_text(example["target"])
        return example

    train_dataset = train_dataset.map(clean_example)
    dev_dataset = dev_dataset.map(clean_example)
    test_dataset = test_dataset.map(clean_example)

    return train_dataset, dev_dataset, test_dataset

ALLOWED_CHARS = r"\p{L}\p{N}'\"«»""\.!\?"
START_NON_ALLOWED = re.compile(rf"^[^{ALLOWED_CHARS}]+")
END_NON_ALLOWED = re.compile(rf"[^{ALLOWED_CHARS}]+$")

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    last_version = None
    while last_version != text:
        last_version = text
        text = START_NON_ALLOWED.sub("", text)
        text = END_NON_ALLOWED.sub("", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_datasets(train_path, dev_path, test_path, expand_eval_lists=True, ):
    train_data = load_data(train_path)
    dev_data = load_data(dev_path)
    test_data = load_data(test_path)

    print(f"Train examples: {len(train_data)}")
    print(f"Dev examples: {len(dev_data)}")
    print(f"Test examples: {len(test_data)}")

    train_dataset = Dataset.from_dict(format_input_data(train_data))
    dev_dataset = Dataset.from_dict(format_input_data(dev_data))
    test_dataset = Dataset.from_dict(format_input_data(test_data))

    def clean_example(example):
        example["source"] = clean_text(example["source"])
        if isinstance(example["target"], list):
            example["target"] = [clean_text(t) for t in example["target"]]
        else:
            example["target"] = clean_text(example["target"])
        return example

    train_dataset = train_dataset.map(clean_example)
    dev_dataset = dev_dataset.map(clean_example)
    test_dataset = test_dataset.map(clean_example)

    return train_dataset, dev_dataset, test_dataset


def tokenize_datasets(train_ds, dev_ds, test_ds, tokenizer, max_length: int = 128):
    def tokenize_function(examples):

        tokenizer.src_lang = "bre_Latn"
        tokenizer.tgt_lang = "fra_Latn"

        model_inputs = tokenizer(
            examples["source"],
            max_length=max_length,
            truncation=True,
            padding=True,
        )

        targets_to_tokenize = []
        for target in examples["target"]:
            if isinstance(target, list):
                targets_to_tokenize.append(target[0])
            else:
                targets_to_tokenize.append(target)

        labels = tokenizer(
            text_target=targets_to_tokenize,
            max_length=max_length,
            truncation=True,
            padding=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["target_text"] = examples["target"]

        return model_inputs

    tokenized_train = train_ds.map(tokenize_function, batched=True, remove_columns=["source", "target"])
    tokenized_dev = dev_ds.map(tokenize_function, batched=True, remove_columns=["source", "target"])
    tokenized_test = test_ds.map(tokenize_function, batched=True, remove_columns=["source", "target"])

    return tokenized_train, tokenized_dev, tokenized_test


bleu_metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds, tokenizer):
    """Compute BLEU score, handling multi-reference cases"""
    preds, labels = eval_preds

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]

    formatted_labels = []
    for label in decoded_labels:
        if isinstance(label, list):
            formatted_labels.append([l.strip() for l in label])
        else:
            formatted_labels.append([label.strip()])

    result = bleu_metric.compute(predictions=decoded_preds, references=formatted_labels)
    return {"bleu": result["score"]}


def apply_lora_restriction(model, restrict_to):
    """Freeze/unfreeze parameters based on restriction"""
    if restrict_to is None:
        return

    print(f"Applying restriction: {restrict_to}")

    for param in model.parameters():
        param.requires_grad = False

    if restrict_to == "encoder":
        for name, param in model.named_parameters():
            if "lora_" in name and ".encoder." in name and ".encoder_attn." not in name:
                param.requires_grad = True

    elif restrict_to == "encoder_and_crossattn":
        for name, param in model.named_parameters():
            if "lora_" in name and (".encoder." in name or ".encoder_attn." in name):
                param.requires_grad = True
    else:
        raise ValueError(f"Unknown restriction: {restrict_to}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def create_lora_model(base_model, lora_config_dict):
    """Create LoRA model from config dictionary"""
    lora_config = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["lora_alpha"],
        target_modules=lora_config_dict["target_modules"],
        lora_dropout=lora_config_dict["lora_dropout"],
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model_lora = get_peft_model(base_model, lora_config)
    restrict_to = lora_config_dict.get("restrict_to")
    if restrict_to:
        apply_lora_restriction(model_lora, restrict_to)

    model_lora.print_trainable_parameters()

    return model_lora


def main():
    args = get_args()
    output_dir = args.output_dir

    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")
    print(f"Using LoRA configuration: {args.lora_config}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Model loaded: {model.num_parameters():,} parameters")

    new_lang = "bre_Latn"
    tokenizer.add_special_tokens({"additional_special_tokens": [new_lang]})
    model.resize_token_embeddings(len(tokenizer))

    print("Added bre_Latn → vocab size now:", len(tokenizer))
    welsh_token = "cym_Latn"
    welsh_id = tokenizer.convert_tokens_to_ids(welsh_token)
    breton_id = tokenizer.convert_tokens_to_ids(new_lang)
    print("Welsh ID =", welsh_id, "| Breton ID =", breton_id)

    with torch.no_grad():
        model.model.shared.weight[breton_id] = model.model.shared.weight[welsh_id].clone()
        model.lm_head.weight[breton_id] = model.lm_head.weight[welsh_id].clone()

    print("Breton embedding initialized from Welsh.")

    train_ds, dev_ds, test_ds = build_datasets(
        args.train, args.dev, args.test,
        expand_eval_lists=True
    )

    tokenized_train, tokenized_dev, tokenized_test = tokenize_datasets(
        train_ds, dev_ds, test_ds, tokenizer, max_length=128
    )


    lora_config_dict = config[args.lora_config]
    print(f"\nLoRA configuration '{args.lora_config}':")
    for key, value in lora_config_dict.items():
        print(f"  {key}: {value}")
    model_lora = create_lora_model(model, lora_config_dict)

    if args.epochs is not None:
        num_epochs = args.epochs
        print(f"Using epochs from CLI: {num_epochs}")
    else:
        num_epochs = 1
        print(f"Using default epochs: {num_epochs}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model_lora,
        label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=200,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        learning_rate=1e-4,
        save_total_limit=3,
        predict_with_generate=False,
        generation_max_length=128,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
    )

    trainer = Seq2SeqTrainer(
        model=model_lora,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=data_collator,
        compute_metrics=None,
    )

    print("\n" + "="*60)
    print("Starting training")
    print(f"Epochs: {num_epochs}")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")

    trainer.train()

    print("\n" + "="*60)
    print("Training finished.")
    print("Evaluating with SacreBLEU")
    print("="*60 + "\n")

    trainer.compute_metrics = lambda eval_preds: compute_metrics(eval_preds, tokenizer)
    trainer.args.predict_with_generate = True
    trainer.args.per_device_eval_batch_size = 4
    trainer.args.eval_accumulation_steps = 4

    dev_results = trainer.predict(tokenized_dev, max_length=128, num_beams=4)
    print("DEV metrics:", dev_results.metrics)

    test_results = trainer.predict(tokenized_test, max_length=128, num_beams=4)
    print("TEST metrics:", test_results.metrics)
    results = {
        "lora_config": args.lora_config,
        "epochs": num_epochs,
        "dev_metrics": dev_results.metrics,
        "test_metrics": test_results.metrics,
    }

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to: {results_path}")

    save_dir = os.path.join(output_dir, "adapter")
    os.makedirs(save_dir, exist_ok=True)

    model_lora.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"Saved adapter + tokenizer to: {save_dir}")


if __name__ == "__main__":
    main()