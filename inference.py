import torch
import model_utils
import dataset_utils
from dataset_utils import LANG_TABLE
import fire

def get_summarize_prompt(text):
    input_prompt = (
            "### Instruction: Below is infomation in a post. Write a summary of the information.\n"
            f"### Text:\n{conversation.strip()}\n"
            f"### Summary:\n"
        )

    return input_prompt


def summarize(model, tokenizer, text: str):
    inputs = tokenizer(
        get_summarize_prompt(text), 
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            top_k=40,
            temperature=0.1,
            repetition_penalty=1.05,
        )
        
    return tokenizer.batch_decode(out_ids[:, input_ids.size(1):], skip_special_tokens=True)[0].strip()


def get_translate_prompt(input_text, src_lang, tgt_lang):
    prompt = (
        f"### Instruction: Translate this from {LANG_TABLE[src_lang]} to {LANG_TABLE[tgt_lang]}, no explaination\n"
        f"### Text:\n{input_text}\n"
        f"### Translation:\n"
    )
    
    return prompt

def translate(model, tokenizer, input_text, pair='de-en'):
    src_lang = pair.split('-')[0]
    tgt_lang = pair.split('-')[1]
    
    inputs = tokenizer(
        get_translate_prompt(input_text, src_lang, tgt_lang), 
        return_tensors="pt"
    ).to(model.device)

    # import pdb; pdb.set_trace()

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            top_k=40,
            temperature=0.2,
            repetition_penalty=1.05,
        )
        
    return tokenizer.batch_decode(out_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True)[0].strip()



if __name__ == "__main__":
    model_path = "models/gemma-2-2b-it"
    model, tokenizer = model_utils.load_quantized_model(model_path)
    # fire.Fire(inference)

    # model_path = ""
    pair = "de-en"
    translate_src_text = "Die Ware hat unter 20 Euro gekostet."

    response = translate(model, tokenizer, translate_src_text, pair)
    
    print(response)
    
