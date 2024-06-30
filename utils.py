import torch
import os
import transformers
from tokenizers import AddedToken
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from unsloth.unsloth import FastLanguageModel
import fire

def load_model(model_name_or_path, device_map="cuda", use_chatml_template=False):
    print("Loading tokenizer and model from:", model_name_or_path)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True, # fast load tokenizer
        padding_side='right' # custom for rotary position embedding
    )

    # set device for multi GPU
    if device_map=="all":
        local_rank = os.environ["LOCAL_RANK"]
        device_map = f"cuda:{local_rank}"
        
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2" # enable flash attention
    )

    # apply chat template
    if use_chatml_template:
        tokenizer = apply_chat_template(tokenizer)

        # resize tokenizer length
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def load_model_lora(model_name_or_path, device_map="cuda:0", use_chatml_template=False):
    print("Loading tokenizer and model, lora:", model_name_or_path)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True, # fast load tokenizer
        padding_side='right' # custom for rotary position embedding
    )

    # BitsAndBytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # set device for multi GPU
    if device_map=="all":
        local_rank = os.environ["LOCAL_RANK"]
        device_map = f"cuda:{local_rank}"
        
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        attn_implementation="flash_attention_2" # enable flash attention
    )

    model.gradient_checkpointing_enable()
    config = LoraConfig(
        r=16, 
        lora_alpha=16, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        lora_dropout=0.0, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    # model.print_trainable_parameters()

    # apply chat template
    if use_chatml_template:
        tokenizer = apply_chat_template(tokenizer)

        # resize tokenizer length
        model.resize_token_embeddings(len(tokenizer))
        
    return model, tokenizer


def load_model_lora_unsloth(model_name_or_path, max_seq_length=2048, device_map="cuda:0", use_chatml_template=False):
    print("Loading tokenizer and model with unsloth:", model_name_or_path)
    # # load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_name_or_path,
    #     use_fast=True, # fast load tokenizer
    #     padding_side='right' # custom for rotary position embedding
    # )
    
    # set device for multi GPU
    if device_map=="all":
        local_rank = os.environ["LOCAL_RANK"]
        device_map = f"cuda:{local_rank}"
    
    # load model by unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_length,
        dtype=None,
        device_map=device_map,
        load_in_4bit=True,
        use_cache=False,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0, # Supports any, but = 0 is optimized
        bias="none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None, # And LoftQ
    )
    
    model.print_trainable_parameters()

    # apply chat template
    if use_chatml_template:
        tokenizer = apply_chat_template(tokenizer)

        # resize tokenizer length
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def apply_chat_template(tokenizer, template="chatml"):
    # Apply chatml template
    if template == "chatml":
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    else:
        tokenizer.chat_template = template

    # Add additional token to ensure tokenizer don't create new sub_token
    tokenizer.add_tokens([AddedToken("<|im_start|>"), AddedToken("<|im_end|>")])

    return tokenizer


def test_chat_template(tokenizer):
    conversation = [{"role": "user", "content": "Hello!"}, 
                    {"role": "assistant", "content": "Hi there! How can I help you today?"},
                    {"role": "user", "content": "I need help with my computer."},
                    {"role": "assistant", "content": "Sure, what's the problem?"}]

    text = tokenizer.apply_chat_template(conversation, tokenize=False)
    ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    print(tokenizer.convert_ids_to_tokens(ids[0]))
    print(text)



def download_model(model_name_or_path, output_dir):
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=output_dir,
        torch_dtype=torch.bfloat16,
        device_map='cuda:0',
        attn_implementation="flash_attention_2" # enable flash attention
    )
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=output_dir,
        use_fast=True, # fast load tokenizer
        padding_side='right' # custom for rotary position embedding
    )

    return output_dir


if __name__ == "__main__":
    fire.Fire()
    # model_path = "models/gemma-2b-it"
    # model, tokenizer = load_model_lora_unsloth(model_path)
    
    # test_chat_template(tokenizer)