import torch
import utils
import fire

def compute_metrics(eval_preds, eval_metric):
    if isinstance(eval_preds, tuple):
        eval_preds = eval_preds[0]
    labels_ids = eval_preds.label_ids # list summarization ids
    pred_ids = eval_preds.predictions # list predict ids
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    result = eval_metric.compute(predictions=pred_str, references=label_str)
    
    return result