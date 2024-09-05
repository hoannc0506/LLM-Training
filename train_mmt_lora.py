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

import model_utils
import dataset_utils
import fire

os.environ["WANDB_PROJECT"]="mmt-ft-lora"

def main(
    model_path: str = "models/gemma-2-2b-it",
    run_name: str = "ft_mmt_gemma2_lora",
    train_batch_size : int = 2,
    max_seq_length: int = 256,
    train_samples: int = 1000,
    val_samples: int = 200
):
    # load model, tokenizer
    model, tokenizer = model_utils.load_model_lora(model_path)

    # load dataset
    dataset_name = "haoranxu/ALMA-Human-Parallel"
    train_dataset = dataset_utils.load_mmt_dataset(
        dataset_name,
        max_seq_length=max_seq_length, 
        split=f"train[:{train_samples}]",
        tokenizer=tokenizer
    )
    val_dataset = dataset_utils.load_mmt_dataset(
        dataset_name,
        max_seq_length=max_seq_length, 
        split=f"validation[:{val_samples}]",
        tokenizer=tokenizer
    )
    
    print("train", train_dataset.num_rows)
    print("val", val_dataset.num_rows)

    # import pdb; pdb.set_trace()
    
    # define train configs
    collator = DataCollatorForCompletionOnlyLM(
        response_template="\n### Translation: ",                     
        tokenizer=tokenizer,
        mlm=False
    )

    output_dir = f"trains/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

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
        save_total_limit=1,
        num_train_epochs=2,
        ddp_find_unused_parameters=False,
        group_by_length=True,
        report_to="wandb",  # enable logging to W&B
        run_name=run_name,  # name of the W&B run (optional)
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

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

if __name__ == "__main__":
    fire.Fire(main)