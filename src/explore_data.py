import json
import re
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
import requests
import numpy as np


def get_all_pages():
    json_files = []
    all_data = []
    first_link = "https://aides-territoires.beta.gouv.fr/api/aids/?categories=economie-circulaire&categories=circuits-courts-filieres&categories=economie-denergie&categories=recyclage-valorisation&categories=empreinte-carbone&categories=assainissement&categories=reseaux-de-chaleur&categories=limiter-les-deplacements-subis&categories=mobilite-partagee&categories=mobilite-pour-tous&categories=amenagement-de-lespace-public-et-modes-actifs&categories=transition-energetique&categories=biodiversite&categories=forets&categories=milieux-humides&categories=montagne&categories=qualite-de-lair&categories=risques-naturels&categories=sols&targeted_audiences=private_sector"
    response = requests.get(first_link, verify=False)
    json_file = response.json()
    all_data += json_file["results"]
    json_files.append(json_file)
    while json_file["next"] is not None:
        link = json_file["next"]
        response = requests.get(link, verify=False)
        json_file = response.json()
        all_data += json_file["results"]
        json_files.append(json_file)
    with open("data/MT_aides_full.json", 'w') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    return json_files, all_data

class AidesDataset:
    def __init__(self, json_path="data/MT_aides.json"):
        self.aides = self.get_aides(json_path)

    def get_aides(self, json_path):
        f = open(json_path)
        aides = json.load(f)
        # aides_ = aides['results']
        print("number of aides:", len(aides))
        return aides

    def filter_features(self, aides):
        filtered_aides = []
        # destinations, perimeter, instructors useful ? what is slub ?
        useful_features = ['categories', 'programs', 'eligibility', 'description', 'mobilization_steps',
                           'targeted_audiences', 'project_examples', 'aid_types', 'name']
        for aide in aides:
            filtered_aide = {k: aide[k] for k in useful_features}
            filtered_aides.append(filtered_aide)
        return filtered_aides

    def clean_text_features(self, aides):
        for aide in aides:
            for feature in list(aide.keys()):
                if isinstance(aide[feature], str):
                    aide[feature] = re.sub('<.*?>|\n', "", aide[feature])
        return aides

    def get_description_vocab(self, aides, vocab_out_path=None):
        tokens = []
        for aide in aides:
            aide_tokens = word_tokenize(aide["description"])
            tokens += aide_tokens
        unique_tokens = list(set(tokens))
        unique_tokens.sort()
        vocab = dict(zip(list(range(len(unique_tokens))), unique_tokens))
        if vocab_out_path is not None:
            with open(vocab_out_path, 'w') as f:
                json.dump(vocab, f, ensure_ascii=False)
        return vocab, tokens

    def get_other_features_vocab(self, aides, vocab_out_path=None):
        tokens = []
        features = list(aides[0].keys())
        features.remove("description")
        for aide in aides:
            for feature in features:
                if isinstance(aide[feature], str) and not aide[feature] == '':
                    tokens += word_tokenize(aide[feature])
                elif isinstance(aide[feature], list) and len(aide[feature]) > 0:
                    tokens += word_tokenize(" ".join(aide[feature]))
        unique_tokens = list(set(tokens))
        unique_tokens.sort()
        vocab = dict(zip(list(range(len(unique_tokens))), unique_tokens))
        if vocab_out_path is not None:
            with open(vocab_out_path, 'w') as f:
                json.dump(vocab, f, ensure_ascii=False)
        return vocab, tokens

    def preprocess_dataset(self):
        self.aides = self.filter_features(self.aides)
        self.aides = self.clean_text_features(self.aides)
        self.description_vocab, tokens_descr = self.get_description_vocab(self.aides,
                                                                          vocab_out_path='data/description_vocab.json')
        len_vocab_desc, num_rare_words_descr = self.plot_most_common_words(tokens_descr, file_name='most_common_words_descriptions', num_words_2=60)
        print("DESCRIPTION")
        print("vocab size", len_vocab_desc)
        print("rare words", num_rare_words_descr)
        self.features_vocab, tokens_features = self.get_other_features_vocab(self.aides,
                                                                             vocab_out_path='data/features_vocab.json')
        len_vocab_feat, num_rare_words_feat = self.plot_most_common_words(tokens_features, file_name='most_common_words_other_features', num_words_2=50, num_to_remove=14)
        print("FEATURES")
        print("vocab size", len_vocab_feat)
        print("rare words", num_rare_words_feat)
        print("done")

    def _truncated_frequencydist(self, freq_dist, num_to_remove=22):
        for i, key in enumerate(list(freq_dist.keys())):
            if i < num_to_remove:
               freq_dist.pop(key)
        return freq_dist

    def _plot_histogram(self, freq_dict, file_name):
        fig, ax = plt.subplots(1, 1, figsize=(60, 10))
        ax.set_title("Most common words")
        ax.bar(freq_dict.keys(), freq_dict.values())
        #ax.xticks(rotation=45)
        ax.tick_params(axis='x',labelsize=24, labelrotation=45)
        rects = ax.patches
        labels = [rect.get_height() for rect in rects]
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label,
                    ha='center', va='bottom', fontsize=20)
        ax.legend()
        plt.tight_layout()
        fig.savefig(file_name, format='pdf')



    def plot_most_common_words(self, tokens, file_name, num_words=30, num_words_2=60, num_to_remove=22):
        fdist = FreqDist(tokens)
        #plt.hist(list(fdist.values()), bins=50)
        #plt.show()
        #d = np.diff(np.unique(data)).min()
        #left_of_first_bin = data.min() - float(d) / 2
        #right_of_last_bin = data.max() + float(d) / 2
        unfreq_words = {k:v for k, v in fdist.items() if v < 10}
        fdist1 = fdist.most_common(num_words)
        fdist2 = fdist.most_common(num_words_2)
        fdist1_dict = {key: value for key, value in fdist1}
        fdist2_dict = {key: value for key, value in fdist2}
        fdist1_dict_truncated = self._truncated_frequencydist(fdist2_dict, num_to_remove)
        file_name_truncated = file_name +"_truncated"
        self._plot_histogram(fdist1_dict, file_name)
        self._plot_histogram(fdist1_dict_truncated, file_name_truncated)
        return len(fdist), len(unfreq_words)
if __name__ == '__main__':
    #json_files, all_data = get_all_pages()
    dataset = AidesDataset()
    dataset.preprocess_dataset()
    print("done")
