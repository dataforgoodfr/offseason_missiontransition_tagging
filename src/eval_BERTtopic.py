# General imports
import argparse
# Data imports
import pandas as pd

from aides_dataset import AidesDataset
# Models imports
from bertopic import BERTopic  # BERTopic for topic modeling
from transformers import pipeline  # We will load XNLI for zeroshot classification
from sklearn.model_selection import train_test_split
import re
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bertopic_path", type=str, default="model/bertopic_v3",
                        help="Path to file containing BERTopic model.")
    parser.add_argument("-n_words_per_topic", type=int, required=False, default=5,
                        help="Number of word to define a topic. Uses the most frequents from the topics.")
    parser.add_argument("-zeroshot_model", type=str, required=False,
                        default="BaptisteDoyen/camembert-base-xnli",
                        help="HuggingFace model name to perform zero-shot classification with.")
    # Dataset arguments
    parser.add_argument("-aides_mt_path", type=str, required=False,
                        default="data/MT_aides.json",
                        help="Path to file containing MT aides dataset.")
    parser.add_argument("-aides_all_path", type=str, required=False,
                        default="data/AT_aides_full.json",
                        help="Path to file containing full aides dataset.")
    parser.add_argument("-results_path", type=str, required=True,
                        help="Path to file to save results to.")

    args = parser.parse_args()

    print(f"Loading BERTopic model from {args.bertopic_path}.")
    topic_model = BERTopic.load(args.bertopic_path)
    all_topics = topic_model.get_topics()
    n_topics = len(all_topics)

    # Define labels:
    # For each topic extracted by bertopic, we create a string containing all
    # the most frequent words for this topic, giving us a label.
    print(f"Computing {args.n_words_per_topic} most frequent words per topic.")


    def get_most_frequent_words(topic):
        # Note: topic is a list of couples word*frequence.
        # Sort by frequence
        topic.sort(key=lambda x: x[1])
        # Keep N most frequents
        topic = topic[:args.n_words_per_topic]
        # Remove frequences
        topic = [word for word, _ in topic]
        # Join with comas
        topic = ", ".join(topic)
        # Return
        return topic


    all_topics = [get_most_frequent_words(topic) for topic in all_topics.values()]

    # Load 0-shot classifier
    print(f"Loading classifier {args.zeroshot_model}.")
    classifier = pipeline("zero-shot-classification", model=args.zeroshot_model)

    # Load MT data and extract MT ids
    print(f"Loading MT aides from {args.aides_mt_path}.")
    MT_aides_dataset = AidesDataset(args.aides_mt_path)
    MT_id = [aide["id"] for aide in MT_aides_dataset.aides]

    # Load data and extract ids
    print(f"Loading all aides from {args.aides_all_path}.")
    aides_dataset = AidesDataset(args.aides_all_path)
    id = [aide["id"] for aide in aides_dataset.aides]

    # Pre-process data
    print("Pre-processing data")
    aides_dataset.filter_features(["name", "description"])
    aides_dataset.clean_text_features(["description"],
                                      no_html_tags=True,
                                      no_escaped_characters=True,
                                      no_punctuation=False,
                                      no_upper_case=False,
                                      no_stopwords=False)
    docs = [f'Titre : {aide["name"]}\nDescription : {aide["description"]}' for aide in aides_dataset.aides]

    # Split in train/ test
    print("Train/test split")
    docs_train, docs_test, id_train, id_test = train_test_split(docs, id, test_size=0.25, random_state=1)

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

    # merge topic_id, topic_name and description on one dataframe
    description_topic = {"doc": MT_docs_test, "topic_id": topics_test, "doc_id": MT_id_test}
    description_topic = pd.DataFrame.from_records(description_topic)
    def get_topic_name(topic):
        topic = re.sub(r'\d+', '', topic)
        topic = topic.replace("_", " ")
        #topic = topic.replace("-", "")
        return topic
    description_topic["topic_name"] = description_topic["topic_id"].apply(lambda t: get_topic_name(MT_test_topic_info.loc[t]["Name"]))
    def clean_description(doc):
        doc = doc.replace("Titre : ", "")
        doc = doc.replace("\nDescription :", ".")
        return doc
    description_topic["doc"] = description_topic["doc"].apply(lambda t: clean_description(t))

    # use zero-shot text classification on each couple (doc, topic_name)
    description_topic["zero_shot_score"] = [0.] * len(description_topic)
    hypothesis_template = "Ce texte est {}."
    for index in list(description_topic.index):
        results = classifier(description_topic.loc[index]["doc"], description_topic.loc[index]["topic_name"], hypothesis_template=hypothesis_template)
        description_topic.at[index, "zero_shot_score"] = results["scores"][0]
    print("done")

    # assign a label for each zero-shot score
    def get_zero_shot_label(score):
        if score < 0.4:
            label = 0
        elif 0.4 <= score <= 0.6:
            label = 0.5
        else:
            label = 1
        return label
    description_topic["zero_shot_label"] = description_topic["zero_shot_score"].apply(lambda t: get_zero_shot_label(t))

    # save results
    description_topic.to_csv(os.path.join(args.results_path, "eval_BERTtopic_with_zeroshot.csv"))

