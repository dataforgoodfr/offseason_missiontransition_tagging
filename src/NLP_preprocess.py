#Text processing
import json
import re
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer

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

def get_data_words(link):
    ''' Load, clean and format the aides. Return a list of list of words '''
    # Uncomment to dowload JSON file
    dataset = AidesDataset(link)
    dataset.aide_descriptions()
    dataset.clean_text_features()

    data_words = dataset.sent_to_words()

    # Remove stopwords and stem
    data_words = remove_stopwords(data_words)
    data_words = french_stemmer(data_words)

    return data_words
