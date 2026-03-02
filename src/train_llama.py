from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import argparse
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)



@dataclass
class TrainConfig:
    model_name: str
    output_dir: str
    train_file: str
    validation_split: float
    max_seq_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: int
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float


def load_config(path: Path) -> TrainConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainConfig(**data)


def format_example(example: dict) -> str:
    instruction = example.get("instruction", "").strip()
    context = example.get("context", "").strip()
    response = example.get("response", "").strip()
    parts = ["<|begin_of_text|>", "<|start_header_id|>system<|end_header_id|>"]
    parts.append("Você é um assistente médico de apoio à decisão clínica.")
    parts.append("<|eot_id|>")
    parts.append("<|start_header_id|>user<|end_header_id|>")
    if context:
        parts.append(f"{instruction}\n\nContexto:\n{context}")
    else:
        parts.append(instruction)
    parts.append("<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>")
    parts.append(response)
    parts.append("<|eot_id|>")
    return "\n".join(parts)


def _prepare_training_file(input_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="llama_train_"))
    output_path = temp_dir / "train.jsonl"
    with input_path.open("r", encoding="utf-8") as reader, output_path.open(
        "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            item = json.loads(line)
            payload = {
                "instruction": item.get("instruction", ""),
                "context": item.get("context", ""),
                "response": item.get("response", ""),
                "source": item.get("source", ""),
            }
            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output_path


def main(config_path: Path) -> None:
    config = load_config(config_path)
    train_path = _prepare_training_file(Path(config.train_file))
    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path)},
        split="train",
    )
    if config.validation_split > 0:
        split = dataset.train_test_split(test_size=config.validation_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict) -> dict:
        size = len(batch["instruction"])
        texts = [
            format_example(
                {
                    "instruction": batch["instruction"][index],
                    "context": batch["context"][index],
                    "response": batch["response"][index],
                }
            )
            for index in range(size)
        ]
        return tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )

    train_tokenized = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_tokenized = (
        eval_dataset.map(
            tokenize,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )
        if eval_dataset is not None
        else None
    )

    use_bnb = False
    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401
            import triton  # noqa: F401

            use_bnb = True
        except Exception:
            use_bnb = False

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    quant_config = None
    if use_bnb:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quant_config,
        torch_dtype=torch_dtype if torch.cuda.is_available() else None,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        evaluation_strategy="steps" if eval_tokenized is not None else "no",
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        optim="paged_adamw_32bit",
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    args = parser.parse_args()
    main(Path(args.config_path))
