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

def match_tags(df_results, id, true_tags, thr=0.7):
    scores = df_results.loc[id]
    scores = scores[scores > thr]
    predicted_tags = list(scores.index)
    match = set(predicted_tags) and set(true_tags)
    return len(match)/len(true_tags)

def match_tags2(id):
    tags__ = tags_per_description_results.loc[id]["tags"]
    true_tags__ = tags_per_description_results.loc[id]["categories"]
    match = set(tags__) and set(true_tags__)
    print(len(match))
    return len(match)/len(true_tags)

def get_description_tags(df_results, id, thr=0.7):
    scores = df_results.loc[id]
    scores = scores[scores > thr]
    return ';'.join(list(scores.index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, default='mt',
                        help='dataset selection. at=aide territoire aides. mt=mission transition aides')
    parser.add_argument("-num_samples", type=int,
                        help='limit number of samples for debugging.')
    parser.add_argument("-thr", type=float, default=0.7,
                        help='thresold value for tag.')
    parser.add_argument("-save_path", type=str,
                        help='save path for models. in that case, only postprocess the results.')
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

    print("TESTING ZEROSHOT WITH threshold {}".format(args.thr))

    if args.save_path is None:
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

        # get tags per description and accuracy
        accuracies = []
        list_tags = []
        list_true_tags = []
        for id in df_results.index:
            true_tags = processed_data.loc[id]["categories"]
            list_true_tags.append(true_tags)
            list_tags.append(get_description_tags(df_results, id, thr=args.thr))
            accuracies.append(match_tags(df_results=df_results, id=id, true_tags=true_tags, thr=args.thr))
        tags_per_description = dict(zip(list(df_results.index), list_tags))
        tags_per_description = pd.DataFrame.from_records(tags_per_description, index=['tags']).T
        #tags_per_description["accuracies"] = pd.Series(accuracies)
        tags_per_description["accuracies"] = accuracies
        tags_per_description["num_tags"] = tags_per_description["tags"].apply(lambda t: len(t.split(";")))
        tags_per_description["true_tags"] = list_true_tags

        accuracy = round(np.mean(accuracies),3)
        print("ACCURACY:", accuracy)

    else:
        tags_per_description = pd.read_csv(os.path.join(args.save_path, "tags_per_descr_thr{}.csv".format(args.thr)))
        tags_per_description.set_index('Unnamed: 0', inplace=True)
        tags_per_description.drop(labels=["true_tags", "accuracies"], axis=1)

    tags_per_description_results = tags_per_description.merge(processed_data, left_index=True, right_index=True, how='inner')
    tags_per_description_results.dropna(axis=0, inplace=True)
    tags_per_description_results["accuracy"] = tags_per_description_results.apply(lambda t: match_tags2(t))

    # save results on csv files
    out_path = 'output/test_zeroshot'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    df_results.to_csv(os.path.join(out_path, "results_test_zeroshot.csv"))
    tags_per_description.to_csv(os.path.join(out_path, "tags_per_descr_thr{}.csv".format(args.thr)))
    print("done")

# TODO: loop on all samples of the dataset.
# TODO: save results on a csv file: line = description id, cols = tags.
