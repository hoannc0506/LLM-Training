import evaluate
from typing import List

class Validator:
    def __init__(self, metric="rouge"):
        self.metric = evaluate.load(metric)

    def get_score(self, preds: List[str], labels: List[str ]):
        return self.metric.compute(predictions=preds, references=labels)

    def batch_eval(eval_preds):
        if isinstance(eval_preds, tuple):
            eval_preds = eval_preds[0]
            
        labels_ids = eval_preds.label_ids # list summarization ids
        pred_ids = eval_preds.predictions # list predict ids
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        result = self.metric.compute(predictions=pred_str, references=label_str)

        return result

if __name__ == "__main__":
    validator = Validator(metric="rouge")

    sample_pred = ['The person is struggling with the emotional impact of a past relationship. They are considering cutting contact with former girlfriends due to the pain and discomfort it may cause.']
    sample_label = ["I still have contact with an old ex's friends but can't stand to see or talk to him. His friends are really nice ,so how do I tell them I possibly want to unfriend them on Facebook because of him?"]

    score = validator.get_score(sample_pred, sample_label)
    print(score)

