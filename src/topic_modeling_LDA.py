import json
import requests
from pprint import pprint
from NLP_preprocess import AidesDataset
import gensim
import gensim.corpora as corpora

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

    # Create the datawords list
    aides_dataset = AidesDataset("data/AT_aides_full.json")
    data_words = aides_dataset.get_data_words()
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    corpus = [id2word.doc2bow(feature_words) for feature_words in data_words]

    # number of topics
    num_topics = 10

    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())

    print("done")
