import json
import requests
from pprint import pprint
from NLP_preprocess import AidesDataset
import gensim
import gensim.corpora as corpora
import os
import argparse

# DEBUG
from pdb import set_trace as bp

def count_word_occurence_in_topics(topics, occ_thr=3):
    all_words = [list(v.keys())[0] for val in topics.values() for v in val]
    count_occurences = {k:all_words.count(k) for k in all_words}
    sorted_tuples = sorted(count_occurences.items(), key=lambda item: item[1])
    sorted_count_occurences = {k: v for k, v in sorted_tuples}
    frequent_words = {k:v for k,v in sorted_count_occurences.items() if v>=occ_thr}
    return sorted_count_occurences, frequent_words

def split_topic_words(words):
        words_probs = words.split('+')
        words_probs = [string.split('*') for string in words_probs]
        words_probs = [{element[1]:float(element[0])} for element in words_probs]
        return words_probs


def create_topic_dictionnary(topics):
    dict_topics = dict.fromkeys(range(len(topics)))
    for topic in topics:
        key, words = topic
        words_probs = split_topic_words(words)
        dict_topics[key] = words_probs
    return dict_topics


class LDATopicModel:
    def __init__(self, dataset, num_topics=10, out_path="output/lda_topic_model"):
        self.num_topics = num_topics
        self.dataset = dataset
        self.out_path = self.create_output_path(out_path)
        _ = self.get_corpus()

    def create_output_path(self, out_path):
        """create the output path for saving LDA topic model results"""
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        return out_path

    def get_corpus(self):
        """get the corpus given the bag-of-words for each description"""
        data_words = self.dataset.get_data_words()
        data_words = data_words.values.flatten()
        id2word = corpora.Dictionary(data_words)
        # Create Corpus
        #train_corpus = [id2word.doc2bow(feature_words) for feature_words in data_words]
        corpus = [id2word.doc2bow(feature_words) for feature_words in data_words]
        self.corpus = corpus
        self.id2word = id2word
        if self.out_path is not None:
            vocab_out_path = os.path.join(self.out_path, "id2word.json")
            with open(vocab_out_path, 'w') as f:
                json.dump(dict(id2word), f, ensure_ascii=False)
        return data_words

    def train_LDA_model(self):
        """train the LDA model and return the topics"""
        lda_model = gensim.models.LdaMulticore(corpus=self.corpus,
                                               id2word=self.id2word,
                                               num_topics=self.num_topics)
        topics = lda_model.print_topics()
        return lda_model, topics

    def postprocess_topics(self, lda_model, topics, num_descr=15):
        "look at word occurences in topics and in a txt file a sample of descriptions & their topic."
        topics = create_topic_dictionnary(topics)
        print(
            '------------------------------------------TOPICS------------------------------------------------------------------------')
        pprint(topics)
        print('-' * 60)
        words_occurences, frequent_occurences = count_word_occurence_in_topics(topics)
        print("frequent words occurrence in topics...")
        print("number of frequent words", len(frequent_occurences))
        print(frequent_occurences)
        out_file = os.path.join(self.out_path, "topic_per_description.txt")
        with open(out_file, 'a') as f:
            for descr_id in range(num_descr):
                descr, descr_topic = self.get_topic_per_description(descr_id, lda_model, topics)
                f.write(list(descr.values())[0] + '\n'+ '\n')
                for topic in descr_topic:
                    f.write(str(topic)+ '\n')
                f.write('\n' + '-' * 60 + '\n')
                # print('DESCRIPTION', descr)
                # print("------------------")
                # pprint(descr_topic)
                # print('---------------------------------------------------------------------------------------')

    def get_topic_per_description(self, description_id, lda_model, topics):
        """Get the most likely topics for the description id with their percentage."""
        topics_rate = lda_model[self.corpus[description_id]]
        topic_ids = [t[0] for t in topics_rate]
        topic_prop = [t[1] for t in topics_rate]
        description_topics = [({k: v}, topics[k]) for k, v in zip(topic_ids, topic_prop)]
        description = self.dataset.aides[description_id]
        return description, description_topics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-num_topics", type=int, default=10, help='number of topics in the lda model.')
    args = parser.parse_args()

    # Get the aides Dataset
    aides_dataset = AidesDataset("data/AT_aides_full.json")
    # Build the LDA Topic Model
    lda_TM = LDATopicModel(dataset=aides_dataset, num_topics=args.num_topics)
    # Train the LDA Model
    lda_model, topics = lda_TM.train_LDA_model()
    lda_TM.postprocess_topics(lda_model=lda_model, topics=topics)

    # for unseen document, we can use:
    #get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)

    print("done")
