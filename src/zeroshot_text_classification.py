from transformers import pipeline
from src.aides_dataset import AidesDataset
import pandas as pd
import os
import argparse

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

def get_description_tags(df_results, id, thr=0.5):
    scores = df_results.loc[id]
    scores = scores[scores > thr]
    return ';'.join(list(scores.index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, default='mt',
                        help='dataset selection. at=aide territoire aides. mt=mission transition aides')
    parser.add_argument("-num_samples", type=int,
                        help='limit number of samples for debugging.')
    args = parser.parse_args()
    classifier = pipeline("zero-shot-classification", model='cache/xnli')

    # read candidate tags
    zeroshot_tags_csv = "data/zeroshot_tags.csv"
    tags = pd.read_csv(zeroshot_tags_csv, header=None)
    tags = list(tags[0])
    tags.sort()

    # load dataset
    data_path = "data/AT_aides_full.json" if args.dataset == "at" else "data/MT_aides.json"
    dataset = AidesDataset(data_path)
    processed_data = dataset.get_unfiltered_data_words()
    data_words = processed_data.values.flatten()

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
    list_tags = []
    for id in df_results.index:
        list_tags.append(get_description_tags(df_results, id))
    tags_per_description = dict(zip(list(df_results.index), list_tags))
    tags_per_description = pd.DataFrame.from_records(tags_per_description, index=['tags']).T

    # save results on csv files
    out_path = 'output/zeroshot_classif'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    df_results.to_csv(os.path.join(out_path, "results.csv"))
    tags_per_description.to_csv(os.path.join(out_path, "tags_per_descr.csv"))

    # look at short descriptions
    short_descr = dataset.get_short_descriptions(processed_data)
    short_descr["tags"] = tags_per_description.loc[list(short_descr.index)]["tags"]
    short_descr.to_csv(os.path.join(out_path, "short_descr_tags.csv"))

    print("done")

# TODO: loop on all samples of the dataset.
# TODO: save results on a csv file: line = description id, cols = tags.
