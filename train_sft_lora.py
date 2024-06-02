import torch
import os
import transformers
import evaluate
import datasets
from tokenizers import AddedToken
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel

from validate import Validator
import utils
import dataset_utils
import fire

def main(
    train_batch_size = 2,
    max_seq_length = 2048,
    train_samples=5000,
    val_samples=500,
):
    # load model, tokenizer
    model_path = "models/gemma-2b-it"
    model, tokenizer = utils.load_model_lora(model_path)

    # load dataset
    dataset_name = 'CarperAI/openai_summarize_tldr'
    train_dataset = dataset_utils.load_dataset(
        dataset_name, tokenizer, 
        max_seq_length=max_seq_length, 
        split=f"train[:{train_samples}]"
    )
    val_dataset = dataset_utils.load_dataset(
        dataset_name, 
        tokenizer, 
        max_seq_length=max_seq_length, 
        split=f"valid[:{val_samples}]"
    )

    print("train", train_dataset.num_rows)
    print("val", val_dataset.num_rows)

    # load rogue validator
    validator = Validator(metric="rouge")

    # define train configs
    collator = DataCollatorForCompletionOnlyLM(
        response_template="### Summary:\n",                     
        tokenizer=tokenizer,
        mlm=False
    )
    output_dir = "trains/sft-summary-gemma"
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=1,
        optim="adamw_8bit", # good for quantize
        bf16=True,
        learning_rate=2e-4,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        logging_steps=20,
        save_steps=500,
        save_total_limit=3,
        num_train_epochs=2,
        ddp_find_unused_parameters=False,
        group_by_length=True
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        # compute_metrics=validator.batch_eval,
    )

    trainer.train()

if __name__ == "__main__":
    fire.Fire(main)