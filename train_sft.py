import os
import torch
import evaluate
from datasets import load_dataset
from trl import ModelConfig, SFTTrainer, get_quantization_config, get_kbit_device_map
from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

# Evaluate metric setup
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
rouge = evaluate.load("rouge")
def compute_metrics(eval_preds):
    if isinstance(eval_preds, tuple):
        eval_preds = eval_preds[0]
    labels_ids = eval_preds.label_ids # list summarization ids
    pred_ids = eval_preds.predictions # list predict ids
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    result = rouge.compute(predictions=pred_str, references=label_str)
    return result



# load data
sft_ds_name = 'CarperAI/openai_summarize_tldr'
n_proc = 8
print("Loading dataset")
sft_train = load_dataset(sft_ds_name, split="train", num_proc=n_proc)
sft_valid = load_dataset(sft_ds_name, split="valid", num_proc=n_proc)
sft_test = load_dataset(sft_ds_name, split="test", num_proc=n_proc)

# define format function
###Text: document\n ### (Short) Summary: summary
formatting_func = lambda example: f"### Text: {example['prompt']}\n ### Summary: {example['label']}"

# Load pretrained
MODEL_NAME = 'facebook/opt-350m'
model_config = ModelConfig(
    model_name_or_path=MODEL_NAME
)

# check torch_dtype
torch_dtype = (
    model_config.torch_dtype
    if model_config.torch_dtype in ["auto", None]
    else getattr(torch, model_config.torch_dtype)
)

# Getting quantization config
quantization_config = get_quantization_config(model_config)

print("Initializing model")
# Creating model_kwargs dictionary with various model configuration parameters
model_kwargs = dict(
    revision=model_config.model_revision,
    trust_remote_code=model_config.trust_remote_code,
    attn_implementation=model_config.attn_implementation,
    torch_dtype=torch_dtype,
    use_cache=False,
    device_map=get_kbit_device_map() if quantization_config is not None else None,
    quantization_config=quantization_config,
)

print("Initializing tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)

# special token setting for decoder only model <pad> = </s>
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Create perf config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# define training arguments
num_epochs = 1 # 10

training_args = TrainingArguments(
    output_dir='./save_model',
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy='steps',
    save_steps=100,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    # gradient_accumulation_steps=2,
    adam_beta1=0.9,
    adam_beta2=0.95,
    num_train_epochs=num_epochs,
    # load_best_model_at_end=True,
    report_to="wandb", # wandb logging
    run_name="sum-sft-opt350m" # wandb run name
)


# define SFT trainer
max_input_length = 256

print("Build trainer")
trainer = SFTTrainer(
    model=model_config.model_name_or_path, # name
    model_init_kwargs=model_kwargs, 
    args=training_args,
    train_dataset=sft_train,
    eval_dataset=sft_valid,
    max_seq_length=max_input_length,
    tokenizer=tokenizer,
    peft_config=peft_config,
    compute_metrics=compute_metrics,
    packing=True,
    formatting_func=formatting_func # run format first to build dataset
)

trainer.train()