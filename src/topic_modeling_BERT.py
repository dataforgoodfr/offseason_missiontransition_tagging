# General imports
import argparse
import pandas as pd
from os.path import join as path_join

from aides_dataset import AidesDataset
from bertopic import BERTopic
from sklearn.model_selection import train_test_split

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
        default="src/model/",
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
    aides_dataset.filter_features(["name", "description"])
    aides_dataset.clean_text_features(["description"],
        no_html_tags=True, # Remove <li> ... </li> stuff
        no_escaped_characters=False, # Keep \n, \t, etc
        no_punctuation=False, # Keep the punctation (BERT handles it)
        no_upper_case=True, # Keep the case (BERT handles it)
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
    save_path = path_join(args.bertopic_viz_path, f"bertopic_{version}.html")
    print(f"Visualising topics in {save_path}.")
    fig = topic_model.visualize_topics()
    fig.write_html(save_path)

    # Save topics in csv
    save_path = path_join(args.bertopic_res_path, "train_topics.csv")
    print(f"Saving topics to {save_path}.")
    topic_model.get_topic_info().to_csv(save_path)

    # Write the topics
    save_path = path_join(args.bertopic_res_path, "BERT_topics")
    print(f"Saving results to {save_path}.")
    f = open(save_path, "w")
    f.write("===================== Model general information =====================")
    f.write("\n\nTopic information for AT full dataset:")
    f.write("\n" + str(topic_model.get_topic_info()))

    # Stats for the topics of MT in train set
    MT_docs_train = [doc for (doc, id) in zip(docs_train, id_train) if id in MT_id]
    topics_train, _ = topic_model.transform(MT_docs_train)
    MT_train_topic_info = topic_model.get_topic_info().copy()
    MT_train_topic_info.set_index("Topic", inplace=True)
    MT_train_topic_info["Count"] = 0
    for topic in topics_train:
        MT_train_topic_info.at[topic, "Count"] += 1
    MT_train_topic_info = MT_train_topic_info[MT_train_topic_info["Count"] > 0]
    f.write("\n\nNumber of elements per topic for MT in the train set:")
    f.write("\n" + str(MT_train_topic_info))

    # Applying the model to the test elements that are in MT
    MT_docs_test = [doc for (doc, id) in zip(docs_test, id_test) if id in MT_id]
    MT_id_test = [id for (doc, id) in zip(docs_test, id_test) if id in MT_id]
    topics_test, _ = topic_model.transform(MT_docs_test)
    MT_test_topic_info = topic_model.get_topic_info().copy()
    MT_test_topic_info.set_index("Topic", inplace=True)
    MT_test_topic_info["Count"] = 0
    for topic in topics_test:
        MT_test_topic_info.at[topic, "Count"] += 1
    MT_test_topic_info = MT_test_topic_info[MT_test_topic_info["Count"] > 0]
    f.write("\n\nNumber of elements per topic for MT in the test set:")
    f.write("\n" + str(MT_test_topic_info))

    f.write("\n\n===================== Examples of outputs =====================")

    # Output for 5 smallest descriptions
    description_topic = [(doc, topic, id) for doc, topic, id in zip(MT_docs_test, topics_test, MT_id_test)]
    description_topic.sort(key = lambda x : len(x[0]))
    for i in range(30, 35):
        f.write(f"\n\n----- Document -----")
        for aide in MT_aides_dataset.aides:
            if aide["id"] == description_topic[i][2]:
                f.write(f'\n\nTitre : {aide["name"]}\nDescription : {aide["description"]}')
        f.write("\nTopic:")
        f.write(MT_test_topic_info.at[description_topic[i][1], "Name"])

    f.close()
