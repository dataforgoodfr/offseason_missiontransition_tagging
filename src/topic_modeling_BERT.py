# General imports
import argparse
import pandas as pd
from os.path import join as path_join
import os
import json
from aides_dataset import AidesDataset
from bertopic import BERTopic
from sklearn.model_selection import train_test_split


def get_topics_stats(docs):
    topics, x = topic_model.transform(docs)
    topic_info = topic_model.get_topic_info().copy()
    topic_info.set_index("Topic", inplace=True)
    topic_info["Count"] = 0
    for topic in topics:
        topic_info.at[topic, "Count"] += 1
    topic_info = topic_info[topic_info["Count"] > 0]
    return topic_info, topics

def save_tags_on_json_files(json_path, topics_per_docs, out_file, filter=True):
    f = open(json_path)
    aides = json.load(f)
    aides = pd.DataFrame.from_records(aides)
    aides.reset_index(inplace=True)
    topics_per_docs.reset_index(inplace=True)
    # set index = id to
    if filter:
        bad_tag = list(set(topics_per_docs[topics_per_docs["topic"] == -1]["list_tags"]))[0]
        topics_per_docs["list_tags"].replace(bad_tag, "", inplace=True)
    aides_with_tags = aides.merge(topics_per_docs[["id", "list_tags"]], on="id", how="outer")
    aides_with_tags.drop(columns="index", inplace=True)
    aides_with_tags.to_json(out_file, orient="records", force_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset arguments
    parser.add_argument("-aides_mt_path", type=str, required=False,
                        default="data/MT_aides.json",
                        help="Path to file containing MT aides dataset.")
    parser.add_argument("-aides_all_path", type=str, required=False,
                        default="data/AT_aides_full_.json",
                        help="Path to file containing full aides dataset.")

    # BERTopic files arguments
    parser.add_argument("-bertopic_model_path", type=str, required=False,
                        default="model/",
                        help="Path to folder to save BERTopic model to.")
    parser.add_argument("-bertopic_viz_path", type=str, required=False,
                        default="plot/bert/",
                        help="Path to folder to save BERTopic visualisation to.")
    parser.add_argument("-bertopic_res_path", type=str, required=False,
                        default="bertopics/",
                        help="Path to folder to save BERTopic results to.")

    parser.add_argument("-load_model", type=str, required=False,
                        default=None,
                        help="Path to load BERTopic model from.")

    args = parser.parse_args()

    version = "v1"

    # Load MT data and extract MT ids
    print(f"Loading MT aides from {args.aides_mt_path}.")
    MT_aides_dataset = AidesDataset(args.aides_mt_path)
    MT_id = [aide["id"] for aide in MT_aides_dataset.aides]

    # Load data and extract ids
    print(f"Loading all aides from {args.aides_all_path}.")
    aides_dataset = AidesDataset(args.aides_all_path)
    id = [aide["id"] for aide in aides_dataset.aides]

    # Process the data
    print("Processing the data.")
    # TODO: add possibly the "examples" field.
    aides_dataset.filter_features(["name", "description"])
    aides_dataset.clean_text_features(["description"],
                                      no_html_tags=True,  # Remove <li> ... </li> stuff
                                      no_escaped_characters=False,  # Keep \n, \t, etc
                                      no_punctuation=False,  # Keep the punctation (BERT handles it)
                                      no_upper_case=True,  # Keep the case (BERT handles it)
                                      no_stopwords=True)
    docs = [f'Titre : {aide["name"]}\nDescription : {aide["description"]}' for aide in aides_dataset.aides]

    # Split in train/ test
    print("Train/test split")
    docs_train, docs_test, id_train, id_test = train_test_split(docs, id, test_size=0.25, random_state=1)

    # Learn/load BERTopic model
    if args.load_model is None:
        # Building model
        print("Building BERTopic model.")
        topic_model = BERTopic(language='French')
        # Training model
        print("Training BERTopic model")
        topic_model.fit(docs_train)
        # Saving model
        save_path = path_join(args.bertopic_model_path, f"bertopic_{version}")
        print(f"Saving BERTopic model to {save_path}.")
        topic_model.save(save_path)
    else:
        print(f"Loading model from {args.load_model}.")
        topic_model = BERTopic.load(args.load_model)

    # Display, visually, topics found
    if not os.path.exists(args.bertopic_viz_path):
        os.makedirs(args.bertopic_viz_path)
    save_path = path_join(args.bertopic_viz_path, f"bertopic_{version}.html")

    print(f"Visualising topics in {save_path}.")
    fig = topic_model.visualize_topics()
    fig.write_html(save_path)

    # Save topics in csv
    if not os.path.exists(args.bertopic_res_path):
        os.makedirs(args.bertopic_res_path)
    save_path = path_join(args.bertopic_res_path, "train_topics.csv")
    print(f"Saving topics to {save_path}.")
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(save_path)

    # Write the topics
    save_path = path_join(args.bertopic_res_path, "BERT_topics")
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    print(f"Saving results to {save_path}.")
    f = open(save_path, "w")

    # get predicted tags on two datasets
    train_topic_info, topics_train = get_topics_stats(docs_train)
    test_topic_info, topics_test = get_topics_stats(docs_test)


    def get_topics_per_docs(docs, topics, ids, topic_info, MT_id, split="train"):
        xxx = [{"doc": doc, "topic": topic, "id": id} for doc, topic, id in zip(docs, topics, ids)]
        xx = pd.DataFrame.from_records(xxx)
        xx["len_doc"] = xx["doc"].apply(len)
        xx["tags"] = [topic_info[topic_info["Topic"] == t]["Name"].values[0] for t in xx["topic"]]
        xx["list_tags"] = [",".join(t.split("_")[1:]) for t in xx["tags"]]
        xx["split"] = split
        xx["dataset"] = xx["id"].apply(lambda t: "MT" if t in MT_id else "AT")
        xx = xx.sort_values(["len_doc"], axis=0)
        return xx


    # get tags results on dataframes
    topics_per_docs_train = get_topics_per_docs(docs_train, topics_train, id_train, topic_info, MT_id)
    topics_per_docs_test = get_topics_per_docs(docs_test, topics_test, id_test, topic_info, MT_id, split="test")
    topics_per_docs = pd.concat([topics_per_docs_train, topics_per_docs_test], axis=0, ignore_index=True)
    topics_per_docs.set_index("id", inplace=True)
    # filter only MT tags
    MT_topics_per_docs = topics_per_docs[topics_per_docs["dataset"] == "MT"]
    # save results (AT + MT subset)
    AT_results_path = path_join(args.bertopic_res_path, "AT_results.csv")
    MT_results_path = path_join(args.bertopic_res_path, "MT_results.csv")
    topics_per_docs.to_csv(AT_results_path)
    MT_topics_per_docs.to_csv(MT_results_path)

    # Display topics stats and example of outputs for AT and MT:
    MT_train_topic_info = MT_topics_per_docs[MT_topics_per_docs["split"] == "train"]["tags"].value_counts()
    MT_test_topic_info = MT_topics_per_docs[MT_topics_per_docs["split"] == "test"]["tags"].value_counts()

    f.write("===================== Model general information =====================")
    f.write("===================== AT full dataset =====================")
    f.write("\n\nTopic information for AT train dataset:")
    f.write("===================== Model general information =====================")
    f.write("===================== AT full dataset =====================")
    f.write("\n\nTopic information for AT train dataset:")
    f.write("\n" + str(topic_model.get_topic_info()))
    f.write("\n\nTopic information for AT test dataset:")
    f.write("\n" + str(test_topic_info))
    f.write("\n============================================================")
    f.write("\n\n===================== MT dataset =====================")
    f.write("\n\nNumber of elements per topic for MT in the train set:")
    f.write("\n" + str(MT_train_topic_info))
    f.write("\n\nNumber of elements per topic for MT in the test set:")
    f.write("\n" + str(MT_test_topic_info))
    f.write("\n\n======Examples of outputs (MT test dataset)=====")
    for i in range(30, 35):
        f.write(f"\n\n----- Document -----")
        f.write(MT_topics_per_docs[MT_topics_per_docs["split"] == "test"].iloc[i]["doc"])
        f.write("\nTopic:")
        f.write(MT_topics_per_docs[MT_topics_per_docs["split"] == "test"].iloc[i]["tags"])

    f.close()

    # save tags in the dataset json file.
    AT_out_file = os.path.basename(args.aides_all_path).split(".")[0] + "_with_tags.json"
    MT_out_file = os.path.basename(args.aides_mt_path).split(".")[0] + "_with_tags.json"
    AT_out_file = os.path.join(args.bertopic_res_path, AT_out_file)
    MT_out_file = os.path.join(args.bertopic_res_path, MT_out_file)
    save_tags_on_json_files(args.aides_all_path, topics_per_docs, AT_out_file)
    save_tags_on_json_files(args.aides_mt_path, MT_topics_per_docs, MT_out_file)

    print("done")

    """
    entetes = [
        u'id',
        u'description',
        u'tag'
    ]

    file_csv = open('Bert_csv.csv', 'w')
    ligneEntete = ";".join(entetes) + "\n"
    file_csv.write(ligneEntete)

    file_csv["id"] = MT_id_test
    for aide in MT_aides_dataset.aides:
        if aide["id"] in MT_id_test :
            file_csv["description"] = aide["description"].apply(lambda t: ' '.join(t))
    out_path = 'output/bertopic_topic_model'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    file_csv.to_csv(os.path.join(out_path, "short_descr_tags.csv"))
    """