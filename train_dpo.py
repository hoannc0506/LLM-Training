import os
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer

PatchDPOTrainer()

def load_model(model_name_or_path, max_seq_length):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        padding_side='right',
    )
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        use_cache=False,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0, # Supports any, but = 0 is optimized
        bias="none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None, # And LoftQ
    )
        
    return tokenizer, model

def load_dataset(tokenizer: AutoTokenizer, dataset_name_or_path: str):
    prompt_format = "[INST] {prompt} [/INST] "
    def convert_conversation_to_input(example):
        return dict(
            prompt=prompt_format.format(prompt=example['question_vi']),
            chosen=example['chosen_vi'],
            rejected=example['rejected_vi'],
        )
        
    dataset = datasets.load_dataset(dataset_name_or_path, split='train[:]')
    dataset = dataset.map(convert_conversation_to_input, remove_columns=list(dataset.features))
    
    return dataset

def train(
    model_name_or_path: str = 'Viet-Mistral/Vistral-7B-Chat',
    dataset_name: str = '5CD-AI/Vietnamese-Intel-orca_dpo_pairs-gg-translated',
    train_batch_size: int = 4,
    max_seq_length: int = 2048,
):
    tokenizer, model = load_model(model_name_or_path=model_name_or_path, max_seq_length=max_seq_length)
    dataset = load_dataset(tokenizer=tokenizer, dataset_name_or_path=dataset_name)
    
    dpo_trainer = DPOTrainer(
        model = model,
        ref_model = None,
        args = TrainingArguments(
            per_device_train_batch_size = train_batch_size,
            gradient_accumulation_steps = 4,
            warmup_ratio = 0.1,
            num_train_epochs = 3,
            learning_rate = 5e-7,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.0,
            lr_scheduler_type = "linear",
            seed = 42,
            output_dir = "outputs",
        ),
        beta = 0.1,
        train_dataset = dataset,
        tokenizer = tokenizer,
        max_length = max_seq_length,
        max_prompt_length = 512,
    )
    
    dpo_trainer.train()