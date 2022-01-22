# General imports
import argparse

# Data imports
from aides_dataset import AidesDataset

# Models imports
from bertopic import BERTopic       # BERTopic for topic modeling
from transformers import pipeline   # We will load XNLI for zeroshot classification

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bertopic_path", type=str, required=True,
        help="Path to file containing BERTopic model.")
    parser.add_argument("-n_words_per_topic", type=int, required=False, default=5,
        help="Number of word to define a topic. Uses the most frequents from the topics.")
    parser.add_argument("-zeroshot_model", type=str, required=False,
        default="BaptisteDoyen/camembert-base-xnli",
        help="HuggingFace model name to perform zero-shot classification with.")
    parser.add_argument("-aides_path", type=str, required=True,
        help="Path to file containing MT aides dataset.")
    parser.add_argument("-results_path", type=str, required=True,
        help="Path to file to save results to.")

    # parser.add_argument("-dataset", type=str, default='mt',
    #                     help='dataset selection. at=aide territoire aides. mt=mission transition aides')
    # parser.add_argument("-num_samples", type=int,
    #                     help='limit number of samples for debugging.')
    # parser.add_argument("-thr", type=float, default=0.5,
    #                     help='limit number of samples for debugging.')
    # parser.add_argument("-save_path", type=str,
    #                     help='save path for models. in that case, only postprocess the results.')
    args = parser.parse_args()

    # Load BERTopic model
    # topic_model = BERTopic(language='French')
    # topic_model.load(args.bertopic_path)
    print(f"Loading BERTopic model from {args.bertopic_path}.")
    topic_model = BERTopic.load(args.bertopic_path)
    all_topics  = topic_model.get_topics()
    n_topics    = len(all_topics)

    # Define labels:
    # For each topic extracted by bertopic, we create a string containing all
    # the most frequent words for this topic, giving us a label.
    print(f"Computing {args.n_words_per_topic} most frequent words per topic.")
    def get_most_frequent_words(topic):
        # Note: topic is a list of couples word*frequence.
        # Sort by frequence
        topic.sort(key=lambda x:x[1])
        # Keep N most frequents
        topic = topic[:args.n_words_per_topic]
        # Remove frequences
        topic = [word for word,_ in topic]
        # Join with comas
        topic = ", ".join(topic)
        # Return
        return topic
    all_topics = [get_most_frequent_words(topic) for topic in all_topics.values()]

    # Load 0-shot classifier
    print(f"Loading classifier {args.zeroshot_model}.")
    classifier = pipeline("zero-shot-classification", model=args.zeroshot_model)

    # Load data
    print(f"Loading dataset from {args.aides_path}.")
    aides_dataset = AidesDataset(args.aides_path)

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

    # Classify
    print(f"Classifying aides & writing results in {args.results_path}.")
    with open(args.results_path, "w") as file:
        for doc in docs:
            result = classifier(doc, all_topics)
            predicted_topic = result["labels"][0]
            topic_proba = result["scores"][0]
            file.write("--------------------------------------------------------------------------------\n")
            file.write(f"--> Topic: {predicted_topic} (with probability {topic_proba})\n")
            file.write(f"--> Aide text:\n{doc}\n")
