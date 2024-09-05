import datasets
from datasets import concatenate_datasets
from tokenizers import AddedToken
from transformers import AutoTokenizer
import model_utils

def load_summary_dataset(
    dataset_name_or_path: str,  
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    split: str = "train[:2000]"
):
    
    def get_summarize_prompt(conversation, summary) -> str:
        output_prompt = (
            "### Instruction: Below is infomation in a post. Write a summary of the information.\n"
            f"### Text:\n{conversation.strip()}"
            f"### Summary:\n{summary.strip()}"
        )
        return output_prompt
    
    def generate_inputs(examples):
        # examples['input_text'] = []
        examples['input_ids'] = []
        examples['attention_mask'] = []
        for idx in range(len(examples['prompt'])):
            text = get_summarize_prompt(examples['prompt'][idx], examples['label'][idx])
            tokenized_data = tokenizer(text, max_length=max_seq_length, truncation=True, padding=True)
            
            # examples['input_text'].append(text)
            examples['input_ids'].append(tokenized_data['input_ids'])
            examples['attention_mask'].append(tokenized_data['attention_mask'])

        return examples
        
    # load raw dataset
    raw_dataset = datasets.load_dataset(dataset_name_or_path, split=split)
    
    # tokenize dataset
    processed_dataset = raw_dataset.map(generate_inputs, batched=True,
                                        remove_columns=list(raw_dataset.features))

    return processed_dataset


def load_mt_dataset(
    dataset_name_or_path: str = "haoranxu/ALMA-Human-Parallel",
    pair: str = "de-en",
    split: str = "train[:2000]",
    max_seq_length: int = 256,
    tokenizer: AutoTokenizer = None
):
    LANG_TABLE = {
        "en": "English", "de": "German", "fr": "French", "cs": "Czech", "is": "Icelandic",
        "zh": "Chinese", "ja": "Japanese", "ru": "Russian", "uk": "Ukrainian",
        "ha": "Hausa", "ro": "Romanian", "gu": "Gujarati",
    }

    langs = pair.split('-')
    source_lang, target_lang = langs[0], langs[1]
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]

    def get_translate_prompt(example):
        prompt = (
            f"### Instruction: Translate this from {src_fullname} to {tgt_fullname}:"
            f"\n### Text: {example[source_lang]}"
            f"\n### Translation: {example[target_lang]}"
        )

        return prompt

    def generate_inputs(examples):
        # examples['input_text'] = []
        examples['input_ids'] = []
        examples['attention_mask'] = []
        for idx in range(len(examples['translation'])):
            text = get_translate_prompt(examples['translation'][idx])
            tokenized_data = tokenizer(text, max_length=max_seq_length, truncation=True, padding=True)
            
            # examples['input_text'].append(text)
            examples['input_ids'].append(tokenized_data['input_ids'])
            examples['attention_mask'].append(tokenized_data['attention_mask'])

        return examples

    # load raw dataset
    raw_dataset = datasets.load_dataset(dataset_name_or_path, name=pair, split=split)
    
    # tokenize dataset
    processed_dataset = raw_dataset.map(
        generate_inputs, 
        batched=True,
        remove_columns=list(raw_dataset.features)
    )

    return processed_dataset


def load_mmt_dataset(
    dataset_name_or_path: str = "haoranxu/ALMA-Human-Parallel",
    pairs: str = "de-en,cs-en,is-en,zh-en,ru-en",
    split: str = "train[:1000]",
    max_seq_length: str = 256,
    seed: int = 42,
    tokenizer: AutoTokenizer = None
):
    processed_datasets = []
    lang_pairs = pairs.split(',')

    for lang_pair in lang_pairs:
        print(f"Loading {lang_pair} dataset")
        loaded_dataset = load_mt_dataset(dataset_name_or_path, lang_pair, split,
                                         max_seq_length, tokenizer)

        processed_datasets.append(loaded_dataset)

    mmt_dataset = concatenate_datasets(processed_datasets)
    mmt_dataset = mmt_dataset.shuffle(seed=seed)

    return mmt_dataset

if __name__ == "__main__":
    model_path = "models/gemma-2-2b-it"
    tokenizer = model_utils.load_tokenizer(model_path)
    # mt_dataset = load_mt_dataset(tokenizer=tokenizer)
    mmt_dataset = load_mmt_dataset(tokenizer=tokenizer)
    
    import pdb; pdb.set_trace()
    print(here)