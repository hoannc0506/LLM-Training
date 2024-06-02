import datasets
from tokenizers import AddedToken
from transformers import (
    AutoTokenizer
)

def format_prompt(conversation, summary) -> str:
    output_prompt = f"""
### Instruction: Below is infomation in a post. Write a summary of the information.

### Text:
{conversation.strip()}

### Summary:
{summary}
""".strip()
    return output_prompt
    
def load_dataset(
    dataset_name_or_path: str,  
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    split: str = "train[:2000]"
):
    dataset = datasets.load_dataset(dataset_name_or_path, split=split)

    def generate_inputs(examples):
        examples['input_text'] = []
        examples['input_ids'] = []
        examples['attention_mask'] = []
        for idx in range(len(examples['prompt'])):
            text = format_prompt(examples['prompt'][idx], examples['label'][idx])
            tokenized_data = tokenizer(text, max_length=max_seq_length, truncation=True, padding=True)
            
            examples['input_text'].append(text)
            examples['input_ids'].append(tokenized_data['input_ids'])
            examples['attention_mask'].append(tokenized_data['attention_mask'])

        return examples
    
    # tokenize dataset
    dataset = dataset.map(generate_inputs, batched=True, remove_columns=list(dataset.features))

    return dataset