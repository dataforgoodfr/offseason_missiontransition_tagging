from transformers import pipeline
from src.NLP_preprocess import AidesDataset


classifier = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli')
candidate_tags = ["environnement", "nature", "forêt"]

# hypothesis_template = "Ce texte est {}."
# sequence = 'Cette aide est à propos du bois.'
# classifier(sequence, candidate_tags, hypothesis_template=hypothesis_template)

dataset = AidesDataset("data/AT_aides_full.json")
processed_data = dataset.get_unfiltered_data_words()
data_words = processed_data.values.flatten()
print("done")

def classify_description(classifier, description, candidate_tags):
    hypothesis_template = "Ce texte est {}."
    results = classifier(description, candidate_tags, hypothesis_template=hypothesis_template, multi_class=True)
    return results

results = []
for descr_words in data_words[:20]:
    descr = ' '.join(descr_words)
    result = classify_description(classifier, descr, candidate_tags)
    results.append(result)

print("done")

#TODO: loop on all samples of the dataset.
#TODO: save results on a csv file: line = description id, cols = tags.
