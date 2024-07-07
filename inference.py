import torch
import utils
import dataset_utils
import fire

def get_inference_prompt(text):
    input_prompt = f"""
### Instruction: Below is infomation in a post. Write a summary of the information.

### Text:
{text.strip()}

### Summary:
""".strip()

    return input_prompt

def summarize(model, tokenizer, text: str):
    inputs = tokenizer(
        get_inference_prompt(text), 
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            top_k=40,
            temperature=0.0001,
            repetition_penalty=1.05,
        )
        
    return tokenizer.batch_decode(out_ids[:, input_ids.size(1):], skip_special_tokens=True)[0].strip()


def inference(
    model_name_or_path,
    input_texts=None,
    device_map="cuda:0"
):
    print("Loading model:", model_name_or_path)
    model, tokenizer = utils.load_model(model_name_or_path, device_map=device_map)

    if isinstance(input_texts, list):
        outputs = [summarize(model, tokenizer, text) for text in input_texts]

    else:
        outputs = summarize(model, tokenizer, input_texts)
    
    return outputs 


if __name__ == "__main__":
    fire.Fire(inference)

    # model_path = ""
    
    
