from transformers import pipeline
from src.aides_dataset import AidesDataset
import pandas as pd
import os
import argparse
import numpy as np

def classify_description(classifier, description, candidate_tags):
    hypothesis_template = "Ce texte est {}."
    results = classifier(description, candidate_tags, hypothesis_template=hypothesis_template, multi_class=True)
    return results

def update_results(results, result, id):
    result = dict(zip(result["labels"], result["scores"]))
    for key in result.keys():
        results[key].append(result[key])
    results["id"].append(id)
    return results

def filter_results(df_results, thr=0.5):
    cols= df_results.columns
    cols.pop("id")
    for col in cols:
        mask = df_results[col] > thr
        df_results[col] = mask * df_results[col]
    return df_results

def get_tag_descr(df_results, tag, thr=0.5):
    df_results = df_results[df_results[tag]>thr][tag]
    return list(df_results.index)

def match_tags(df_results, id, true_tags, thr=0.7):
    scores = df_results.loc[id]
    scores = scores[scores > thr]
    predicted_tags = list(scores.index)
    match = set(predicted_tags) and set(true_tags)
    return len(match)/len(true_tags)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, default='mt',
                        help='dataset selection. at=aide territoire aides. mt=mission transition aides')
    parser.add_argument("-num_samples", type=int, default=1,
                        help='limit number of samples for debugging.')
    args = parser.parse_args()
    classifier = pipeline("zero-shot-classification", model='cache/xnli')

    # load dataset
    data_path = "data/AT_aides_full.json" if args.dataset == "at" else "data/MT_aides.json"
    dataset = AidesDataset(data_path)
    processed_data = dataset.get_unfiltered_data_words(["id", "description", "categories"])

    # get categories tags
    tags_ = list(processed_data["categories"])
    tags = list(set([e for cat in tags_ for e in cat]))

    data_words = processed_data["description"].values.flatten()

    # zero shot text classification
    results = dict.fromkeys(["id"] + tags)
    for key in results.keys():
        results[key] = []
    num_samples = len(data_words) if args.num_samples is None else args.num_samples
    for id, descr_words in zip(list(processed_data.index[:num_samples]), data_words[:num_samples]):
        descr = ' '.join(descr_words)
        result = classify_description(classifier, descr, tags)
        update_results(results, result, id)
    # format results
    df_results = pd.DataFrame.from_records(results)
    df_results.set_index(keys="id", inplace=True)
    df_results = df_results.apply(lambda t: round(t, 3))
    print(df_results.head())

    # get tags per description
    accuracies = []
    for id in df_results.index:
        true_tags = processed_data.loc[id]["categories"]
        accuracies.append(match_tags(df_results=df_results, id=id, true_tags=true_tags))

    accuracy = round(np.mean(accuracies),3)
    print("ACCURACY:", accuracy)

    # save results on csv files
    out_path = 'output/zeroshot_classif'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    df_results.to_csv(os.path.join(out_path, "results_test_zeroshot.csv"))

    print("done")

# TODO: loop on all samples of the dataset.
# TODO: save results on a csv file: line = description id, cols = tags.
