
from aides_dataset import AidesDataset
from bertopic import BERTopic
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import nltk
nltk.download('stopwords')
import csv

if __name__ == '__main__':
    version = "v1"

    # Load MT data and extract MT ids
    MT_aides_dataset = AidesDataset("data/MT_aides.json")
    MT_id = [aide["id"] for aide in MT_aides_dataset.aides]

    # Load data and extract ids
    aides_dataset = AidesDataset("data/AT_aides_full_.json")
    id = [aide["id"] for aide in aides_dataset.aides]

    # Process the data
    aides_dataset.filter_features(["name", "description"])
    aides_dataset.clean_text_features(["description"],
        no_html_tags=True, # Remove <li> ... </li> stuff
        no_escaped_characters=False, # Keep \n, \t, etc
        no_punctuation=False, # Keep the punctation (BERT handles it)
        no_upper_case=True, # Keep the case (BERT handles it)
        no_stopwords=True)
    docs = [f'Titre : {aide["name"]}\nDescription : {aide["description"]}' for aide in aides_dataset.aides]

    # Split in train/ test
    docs_train, docs_test, id_train, id_test = train_test_split(docs, id, test_size=0.25, random_state=1)

    # Learn/load BERTopic model
    topic_model = BERTopic(language='French')
    topic_model.fit(docs_train)
    # topic_model = BERTopic.load(f"src/model/bertopic_{version}")

    # Save model for future use
    #topic_model.save(f"src/model/bertopic_{version}")

    # Display, visually, topics found
    fig = topic_model.visualize_topics()
    fig.write_html(f"plots/bert/bertopic_{version}.html")

    # Save topics in csv
    topic_model.get_topic_info().to_csv('bertopics/train_topics.csv')

    # Write the topics
    f = open("bertopics/BERT_topics", "w")
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

    print(MT_id_test)
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

    df = pd.DataFrame(data=MT_id_test,columns=['id'])
    list_descr = []
    for i in MT_aides_dataset :
        for aide in MT_aides_dataset :
            if aide["id"] == i:
                list_descr.append(aide["description"])
    print(df)
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