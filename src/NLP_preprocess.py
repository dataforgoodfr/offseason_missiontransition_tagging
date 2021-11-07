#Text processing
import json
import re
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from explore_data import plot_most_common_words
import os

class AidesDataset:
    def __init__(self, json_path):
        self.aides = self.get_aides(json_path)

    def get_aides(self, json_path):
        ''' Load JSON file containing the aides '''
        f = open(json_path)
        aides = json.load(f)
        # aides_ = aides['results']
        print("number of aides:", len(aides))
        return aides

    def aide_descriptions(self):
        ''' Filter aides, keeping only descriptions '''
        descriptions = []
        for aide in self.aides:
            aide_description = {'description' : aide['description']}
            descriptions.append(aide_description)
        self.aides = descriptions

    def clean_text_features(self):
        ''' Remove part of the punctuation and change the case to lower case'''
        for aide in self.aides:
            for feature in list(aide.keys()):
                if isinstance(aide[feature], str):
                    aide[feature] = re.sub('[,\.!?:;-]|<.*?>|\n', "", aide[feature])
                    aide[feature] = aide[feature].lower()

    def sent_to_words(self):
        ''' Return a list containing tokenized lists of the features' words '''
        all_texts_list = []
        for aide in self.aides:
            for feature in list(aide.keys()):
                if isinstance(aide[feature], str):
                    all_texts_list.append(aide[feature])
        data_words = []
        for text in all_texts_list:
            data_words.append(gensim.utils.simple_preprocess(str(text)))
        return data_words

    def get_data_words(self):
        ''' Load, clean and format the aides. Return a list of list of words '''
        # Uncomment to dowload JSON file
        self.aide_descriptions()
        self.clean_text_features()
        data_words = self.sent_to_words()

        # Remove stopwords and stem
        data_words = remove_stopwords(data_words)
        data_words = french_stemmer(data_words)

        return data_words

def remove_stopwords(data_words):
    ''' Remove French stopwords for a list of list of words '''
    stop_words = stopwords.words('french')
    return [[word for word in feature_words if word not in stop_words]
                                    for feature_words in data_words]

def french_stemmer(data_words):
    ''' Apply French stemming for words in a list of list of words '''
    stemmer = FrenchStemmer()
    return [[stemmer.stem(word) for word in feature_words]
                                    for feature_words in data_words]

def flatten_list(nested_list):
    flattened_list = [s for l in nested_list for s in l]
    return flattened_list


if __name__ == '__main__':
    aides_dataset = AidesDataset("data/AT_aides_full.json")
    data_words = aides_dataset.get_data_words()
    tokens = flatten_list(data_words)
    if not os.path.isdir("plots"):
        os.makedirs("plots")
    plot_most_common_words(tokens=tokens, file_name="plots/most_common_words_LDA")