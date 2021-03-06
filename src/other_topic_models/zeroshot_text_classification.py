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
    parser.add_argument("-thr", type=float, default=0.5,
                        help='limit number of samples for debugging.')
    parser.add_argument("-save_path", type=str,
                        help='save path for models. in that case, only postprocess the results.')
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
    processed_data = dataset.get_unfiltered_data_words(useful_features=["description", "id", "categories"])
    data_words = processed_data.values.flatten()

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

        # get tags per description
        list_tags = []
        for id in df_results.index:
            list_tags.append(get_description_tags(df_results, id, thr=args.thr))
        tags_per_description = dict(zip(list(df_results.index), list_tags))
        tags_per_description = pd.DataFrame.from_records(tags_per_description, index=['tags']).T
        tags_per_description["num_tags"] = tags_per_description["tags"].apply(lambda t: len(t.split(";")))

        # save results on csv files
        out_path = 'output/zeroshot_classif'
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        df_results.to_csv(os.path.join(out_path, "results.csv"))
        tags_per_description.to_csv(os.path.join(out_path, "tags_per_descr_thr{}.csv".format(args.thr)))

    else:
        tags_per_description = pd.read_csv(os.path.join(args.save_path, "tags_per_descr_thr{}.csv".format(args.thr)))
        tags_per_description.set_index('Unnamed: 0', inplace=True)
    # look at short descriptions
    short_descr = dataset.get_short_descriptions(processed_data)
    short_descr["description"] = short_descr["description"].apply(lambda t: ' '.join(t))
    short_descr_results = tags_per_description.merge(short_descr, left_index=True, right_index=True, how='inner')
    short_descr_results.to_csv(os.path.join(args.save_path, "short_descr_tags_thr{}.csv".format(args.thr)))

    print("done")

# TODO: loop on all samples of the dataset.
# TODO: save results on a csv file: line = description id, cols = tags.
