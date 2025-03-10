import os
import json
from typing import Dict, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed
)
from datasets import Dataset
import numpy as np

MODEL_NAME = "distilgpt2"  
OUTPUT_DIR = "models/email_generator_v1"
DATA_PATH = "data/processed/train.json"
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
SEED = 42

def load_dataset(data_path: str) -> Dataset:
    """
    Загрузка датасета из JSON файла.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    return Dataset.from_list(data)

def prepare_training_data(examples: Dict, tokenizer) -> Dict:
    """
    Подготовка данных для обучения.
    """
    prompts = []
    for i in range(len(examples['text'])):
        context = examples['context'][i] if 'context' in examples else None
        
        context_str = ""
        if context:
            context_str = "Previous correspondence:\n"
            for msg in context:
                context_str += f"From: {msg['from']}\n{msg['text']}\n\n"
        
        prompt = f"""Write a professional customer service email response.

{context_str}
Incoming email:
{examples['text'][i]}

Guidelines:
1. Use professional language
2. Be specific and clear
3. Address all questions
4. Maintain appropriate tone

Response:
{examples['response'][i]}

<|endoftext|>"""
        prompts.append(prompt)
    
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    return tokenized

def data_collator(examples: List[Dict], tokenizer) -> Dict:
    """
    Подготовка батча данных.
    """
    max_length = max(len(ex['input_ids']) for ex in examples)
    
    input_ids = torch.stack([
        torch.nn.functional.pad(
            torch.tensor(ex['input_ids']),
            (0, max_length - len(ex['input_ids'])),
            value=tokenizer.pad_token_id
        )
        for ex in examples
    ])
    
    attention_mask = torch.stack([
        torch.nn.functional.pad(
            torch.tensor(ex['attention_mask']),
            (0, max_length - len(ex['attention_mask'])),
            value=0
        )
        for ex in examples
    ])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone()
    }

def main():
    set_seed(SEED)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    
    dataset = load_dataset(DATA_PATH)
    
    tokenized_dataset = dataset.map(
        lambda x: prepare_training_data(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    train_val_split = tokenized_dataset.train_test_split(
        test_size=0.1,
        seed=SEED
    )
    train_data = train_val_split['train']
    val_data = train_val_split['test']
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=100,
        warmup_ratio=0.15,
        report_to="none",
        optim="adamw_torch",
        weight_decay=0.02,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=lambda x: data_collator(x, tokenizer),
    )
    
    trainer.train()
    
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main() 