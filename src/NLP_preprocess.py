#Text processing
import json
import re
import gensim
from gensim.utils import simple_preprocess
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import pandas as pd
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

    def filter_features(self, useful_features):
        ''' Filter aides, keeping the features in the list useful_features '''
        filtered_aides = []
        for aide in self.aides:
            filtered_aide = {k: aide[k] for k in useful_features}
            filtered_aides.append(filtered_aide)
        self.aides = filtered_aides

    def clean_text_features(self, features_to_clean):
        ''' Remove part of the punctuation and change the case to lower case
            in the features features_to_clean '''
        for aide in self.aides:
            for feature in features_to_clean:
                if isinstance(aide[feature], str):
                    aide[feature] = re.sub('[,\.!?:;-]|<.*?>|\n', "", aide[feature])
                    aide[feature] = aide[feature].lower()

    def to_pandas(self):
        ''' Transform an AidesDataset to a pandas dataframe
            with column idexed by the dictionaries keys '''
        return pd.DataFrame.from_records(self.aides, index = 'id')

    def get_data_words(self):
        ''' Load, clean and format the aides. Return a list of list of words '''
        self.filter_features(["id", "description"])
        self.clean_text_features(["description"])
        data = self.to_pandas()
        data = sent_to_words(data, ["description"])
        # Remove stopwords and stem
        data = remove_stopwords(data, ["description"])
        data = french_stemmer(data, ["description"])
        return data

def sent_to_words(data, selected_features):
    ''' From a panda dataframe, transform text in features in a
        tokenized lists of the features' words '''
    for feature in selected_features:
        data[feature] = data[feature].map(gensim.utils.simple_preprocess)
    return data

def remove_stopwords(data, selected_features):
    ''' Remove French stopwords for a  list of words in selected_features '''
    stop_words = stopwords.words('french')
    for feature in selected_features:
        data[feature] = data[feature].map(
            lambda words_list : [word for word in words_list if word not in stop_words])
    return data

def french_stemmer(data, selected_features):
    ''' Apply French stemming for a  list of words in selected_features '''
    stemmer = FrenchStemmer()
    for feature in selected_features:
        data[feature] = data[feature].map(
            lambda words_list : [stemmer.stem(word) for word in words_list])
    return data

def get_data_words(link):
    ''' Load, clean and format the aides. Return a list of list of words '''
    # Uncomment to dowload JSON file
    dataset = AidesDataset(link)
    dataset.filter_features(["id", "description"])
    dataset.clean_text_features(["description"])
    data = dataset.to_pandas()
    data = sent_to_words(data, ["description"])
    # Remove stopwords and stem
    data = remove_stopwords(data, ["description"])
    data = french_stemmer(data, ["description"])
    return data


def flatten_list(nested_list):
    flattened_list = [s for l in nested_list for s in l]
    return flattened_list


if __name__ == '__main__':
    aides_dataset = AidesDataset("data/AT_aides_full.json")
    data_words = aides_dataset.get_data_words()
    tokens = data_words.values.flatten()
    tokens = flatten_list(list(tokens))
    if not os.path.isdir("plots"):
        os.makedirs("plots")
    plot_most_common_words(tokens=tokens, file_name="plots/most_common_words_LDA")