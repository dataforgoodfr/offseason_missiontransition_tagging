
from NLP_preprocess import AidesDataset

from bertopic import BERTopic

if __name__ == '__main__':
    version = "v3"
    # Load data
    aides_dataset = AidesDataset("data/AT_aides_full.json")

    # Process the data
    aides_dataset.filter_features(["name", "description"])
    aides_dataset.clean_text_features(["description"],
        no_html_tags=True, # Remove <li> ... </li> stuff
        no_escaped_characters=False, # Keep \n, \t, etc
        no_punctuation=False, # Keep the punctation (BERT handles it)
        no_upper_case=True, # Keep the case (BERT handles it)
        no_stopwords=True)
    docs = [f'Nom : {aide["name"]}\nDescription : {aide["description"]}' for aide in aides_dataset.aides]
    print(docs[0])

    # Learn/load BERTopic model
    topic_model = BERTopic(language='French')
    topics, probabilities = topic_model.fit_transform(docs)
    # topic_model = BERTopic.load(f"src/model/bertopic_{version}")

    # Save model for future use
    topic_model.save(f"src/model/bertopic_{version}")

    # Display topics found
    print(topic_model.get_topic_info())

    # Display, visually, topics found
    fig = topic_model.visualize_topics()
    fig.write_html(f"plots/bert/bertopic_{version}.html")

    """ Ce que j'ai fait :
        1. Retirer les balises HTML car (i) elles ne sont pas du texte utile, et
        (ii) elles contiennent des mots en trop ("strong", "opportunities").
            ==> N'améliore pas les topics.
        2. Ajouter le nom de l'aide aux documents traités par BERTopic.
            ==> Améliore un peu les topics.
        3. Retirer les stop words, qui semblent polluer les topics.
        Les majuscules sont aussi retirées, pour se simplifier la vie.
            ==> Améliore un peu les topics.
    """
