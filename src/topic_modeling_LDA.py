import json
import requests
from pprint import pprint
from NLP_preprocess import get_data_words
import gensim
import gensim.corpora as corpora
from sklearn.model_selection import train_test_split

# DEBUG
from pdb import set_trace as bp

def get_all_pages():
    all_data = []
    first_link = "https://aides-territoires.beta.gouv.fr/api/aids/"
    response = requests.get(first_link, verify=False)
    json_file = response.json()
    all_data += json_file["results"]
    while json_file["next"] is not None:
        link = json_file["next"]
        response = requests.get(link, verify=False)
        json_file = response.json()
        all_data += json_file["results"]
    with open("data/AT_aides_full.json", 'w') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    #Uncomment to download the json file
    #get_all_pages()

    # Create the panda dataframe indexed by "id" containing the selected features
    processed_data = get_data_words("data/AT_aides_full.json")
    data_train, data_test = train_test_split(processed_data, test_size=100)

    # LDA Training
    # Create Dictionary
    data_words = data_train.values.flatten()
    id2word = corpora.Dictionary(data_words)

    # Create Corpus
    train_corpus = [id2word.doc2bow(feature_words) for feature_words in data_words]

    # number of topics
    num_topics = 10

    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=train_corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())

    # Testing the model
    test_index = data_test.index[0]
    test_corpus = [id2word.doc2bow(data_test['description'][test_index])]
    print("====================")
    print("Test Results")
    print("====================")

    print("Description:")
    print(data_test['description'][test_index])
    print("Topic:")
    for x in lda_model.get_document_topics(test_corpus):
        print(x)

    print("done")
