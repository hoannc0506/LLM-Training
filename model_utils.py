import torch
import os
import transformers
from tokenizers import AddedToken
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import fire

if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    devices = os.environ["CUDA_VISIBLE_DEVICES"]
    print(f"'CUDA_VISIBLE_DEVICES' is currently {devices} \n")


def load_tokenizer(model_name_or_path, set_eos_by_pad=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True, # fast load tokenizer
        padding_side='right' # custom for rotary position embedding
    )
    
    return tokenizer

def load_model(model_name_or_path, use_chatml_template=False):
    print("Loading tokenizer and model from:", model_name_or_path)
    
    # load tokenizer
    tokenizer = load_tokenizer(model_name_or_path)

    # set device for multi GPU
    # if device_map=="all":
    device_map = "auto"
    try:
        local_rank = os.environ["LOCAL_RANK"]
        device_map = f"cuda:{local_rank}"
    except Exception as e:
        print(e)
        
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


def load_model_lora(model_name_or_path, use_chatml_template=False):
    print("Loading tokenizer and model, lora:", model_name_or_path)
    
    # load tokenizer
    tokenizer = load_tokenizer(model_name_or_path)

    # BitsAndBytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # set device for multi GPU
    device_map = "auto"
    try:
        local_rank = os.environ["LOCAL_RANK"]
        device_map = f"cuda:{local_rank}"
    except Exception as e:
        print(e)
        pass
        
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
    # fire.Fire()
    model_path = "models/gemma-2-2b-it"
    model, tokenizer = load_model_lora(model_path)
    
    test_chat_template(tokenizer)