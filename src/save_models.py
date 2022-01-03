import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

nli_model = AutoModelForSequenceClassification.from_pretrained('joeddav/xlm-roberta-large-xnli')
tokenizer = AutoTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli')
if not os.path.isdir("../cache/xnli"):
    os.makedirs("../cache/xnli")
nli_model.save_pretrained("cache/xnli")
tokenizer.save_pretrained("cache/xnli")

print("done loading hugging face models")
