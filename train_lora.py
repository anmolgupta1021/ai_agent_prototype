#!/usr/bin/env python3
"""train_lora.py

Example LoRA fine-tuning script using Hugging Face Transformers + PEFT.

Important:
- Set MODEL to the base model you have access to.
- This script is a starter template; tweak tokenization, max_length, and training args for your environment.
"""

import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os

MODEL = "meta-llama/Llama-2-7b-chat-hf"  # Change this to the base model you will use.

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_file', type=str, required=True, help='Path to training jsonl')
    p.add_argument('--validation_file', type=str, required=True, help='Path to validation jsonl')
    p.add_argument('--output_dir', type=str, default='lora_checkpoint')
    p.add_argument('--per_device_train_batch_size', type=int, default=1)
    p.add_argument('--num_train_epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--max_length', type=int, default=1024)
    return p.parse_args()

def build_prompt(example):
    # Compose a full text prompt used for fine-tuning.
    instruction = example.get('instruction','Instruction: Produce structured JSON note.')
    inp = example.get('input','')
    output = example.get('output','')
    prompt = f"{instruction}\n\n{inp}\n\nResponse:\n{output}"
    return prompt

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    # load model in 8-bit if possible to reduce memory (requires bitsandbytes)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map='auto', load_in_8bit=True)
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['q_proj','v_proj','k_proj','o_proj'],
        lora_dropout=0.05,
        bias='none'
    )

    model = get_peft_model(model, lora_config)

    ds = load_dataset('json', data_files={'train': args.train_file, 'validation': args.validation_file})

    def tokenize_fn(ex):
        prompt = build_prompt(ex)
        tokenized = tokenizer(prompt, truncation=True, max_length=args.max_length)
        # create labels - full sequence (causal LM)
        tokenized['input_ids'] = tokenized['input_ids']
        tokenized['attention_mask'] = tokenized['attention_mask']
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    tokenized = ds.map(tokenize_fn, remove_columns=ds['train'].column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        optim='adamw_torch',
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        tokenizer=tokenizer,
    )

    trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter + tokenizer to {args.output_dir}")

if __name__ == '__main__':
    main()
