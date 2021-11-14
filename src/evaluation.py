#code based on https://towardsdatascience.com/nlp-preprocessing-and-latent-dirichlet-allocation-lda-topic-modeling-with-gensim-713d516c6c7d

from gensim.models import CoherenceModel
import json
import re
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
import requests
import numpy as np
import requests
from pprint import pprint
import gensim
import gensim.corpora as corpora
import nltk
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from NLP_preprocess import AidesDataset
from sklearn.model_selection import train_test_split
import pickle


#nltk.download('punkt')


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
    aides_dataset = AidesDataset("data/AT_aides_full.json")
    processed_data = aides_dataset.get_data_words()
    data_train, data_test = train_test_split(processed_data, test_size=100)

    # LDA Training
    # Create the datawords list
    aides_dataset = AidesDataset("data/AT_aides_full.json")
    data_words = aides_dataset.get_data_words()
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
    print('')
    print("Description:")
    print(data_test['description'][test_index])
    print('')
    print("Topic:")
    for x in lda_model.get_document_topics(test_corpus):
        print(x)

    print('')
    print('')
    print("====================")
    print("Evaluation")
    print("====================")
    print('')
    print('')
    # Compute Perplexity (lower is better)
    print('Log-perplexity : ', round(lda_model.log_perplexity(train_corpus), 2)) #Calculate and return per-word likelihood bound, using the chunk of documents as evaluation corpus.
    print('')
    # Compute Coherence Score (higher is better)
    coherence_model_lda = CoherenceModel(model=lda_model, corpus=train_corpus, texts=processed_data, dictionary=id2word, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score: ', round(coherence_lda, 2))

    print('')  
    print('Umass for topic : ', lda_model.top_topics(train_corpus)) #Calculate the Umass topic coherence for each topic.

    print('')
    print('')
    print("====================")
    print("Visalization")
    print("====================")
    print('')

    # Visualize the topics from keywords

    LDAvis_prepared = gensimvis.prepare(lda_model, train_corpus, id2word)
    with open("plots/ldavis_prepared.html", 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
# load the pre-prepared pyLDAvis data from disk
    with open("plots/ldavis_prepared.html", 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, "plots/ldavis_prepared.html")

    print("done")




