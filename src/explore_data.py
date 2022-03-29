from nltk.probability import FreqDist
import os
import matplotlib.pyplot as plt
from aides_dataset import AidesDataset, flatten_list, sent_to_words, remove_stopwords, french_stemmer
import argparse
from wordcloud import WordCloud


# DEBUG
from pdb import set_trace as bp

def plot_most_common_words(tokens, file_name, num_words=30):
    fdist = FreqDist(tokens)
    fdist1 = fdist.most_common(num_words)
    fdist1_dict = {key: value for key, value in fdist1}
    plot_histogram(fdist1_dict, file_name)
    return len(fdist)

def plot_histogram(freq_dict, file_name):
    fig, ax = plt.subplots(1, 1, figsize=(60, 30))
    #ax.set_title("Most common words", fontsize=)
    ax.bar(freq_dict.keys(), freq_dict.values())
    ax.tick_params(axis='x', labelsize=36, labelrotation=45)
    rects = ax.patches
    labels = [rect.get_height() for rect in rects]
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label,
                ha='center', va='bottom', fontsize=36)
    ax.legend()
    plt.tight_layout()
    fig.savefig(file_name, format='png')

def plot_histogram_of_categories(data_words, file_name):
    colors = ["tab:gray", "tab:brown", "rosybrown", "tab:pink", "darkorange", "tab:olive", "tab:cyan", "darkcyan", "royalblue"]
    xxx = data_words.categories.apply(lambda t: list(set([e.split("|")[0] for e in t])))
    xxx = xxx.values.flatten()
    xxx = flatten_list(list(xxx))
    fdist = FreqDist(xxx)
    fdist = dict(sorted(fdist.items(), key=lambda item: item[1], reverse=True))
    fig, ax = plt.subplots(1, 1, figsize=(60, 30))
    barlist = ax.bar(fdist.keys(), fdist.values())
    ax.tick_params(axis='x', labelsize=36, labelrotation=90)
    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        labelleft=False)
    for i, col in enumerate(colors):
        barlist[i].set_color(col)
    rects = ax.patches
    labels = [rect.get_height() for rect in rects]
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label,
                ha='center', va='bottom', fontsize=36)
    ax.legend()
    plt.tight_layout()
    fig.savefig(file_name, format='png')

def generate_wordcloud(tokens, file_name):
    wordcloud = WordCloud(width=800, height=400).generate(" ".join(tokens))
    # Display the generated image
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(file_name, format='png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, default='mt',
                        help='dataset selection. at=aide territoire aides. mt=mission transition aides')
    args = parser.parse_args()
    if args.dataset == "at":
        aides_dataset = AidesDataset("data/AT_aides_full.json")
    elif args.dataset == "mt":
        aides_dataset = AidesDataset("data/MT_aides.json")

    useful_features = ['categories', 'programs', 'eligibility', 'mobilization_steps',
                       'targeted_audiences', 'project_examples', 'aid_types', 'name']
    all_features = ['id','description'] + useful_features

    # name, eligibility, project_examples
    # other are list.

    # preprocess useful future from original json file.
    aides_dataset.filter_features(['id','description'] + useful_features)
    aides_dataset.clean_text_features(useful_features, no_stopwords=True)
    data = aides_dataset.to_pandas()
    data_features = data.drop(["description"], axis=1)
    data = data[["description", "categories"]]

    #TODO: use predefined  get_data_words function instead.

    # preprocess description data
    data = sent_to_words(data, ["description"])
    data = remove_stopwords(data, ["description"])
    data = french_stemmer(data, ["description"])

    # preprocess other features data
    for col in data_features.columns:
        if isinstance(data_features[col].iloc[0], list):
            data_features[col] = data_features[col].apply(lambda t: " ".join(t) if len(t) > 0 else "")
    data_features = sent_to_words(data_features, useful_features)

    # plot words statistics on description
    tokens = data.description.values.flatten()
    tokens = flatten_list(list(tokens))

    save_path = 'plots/AT' if args.dataset == "at" else "plots/MT"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    plot_most_common_words(tokens=tokens, file_name=os.path.join(save_path, "most_common_words_description.png"), num_words=50)
    generate_wordcloud(tokens, os.path.join(save_path, "wc_description.png"))

    # plot statistics on other features
    feature_tokens = data_features.values.flatten()
    feature_tokens = flatten_list(feature_tokens)
    plot_most_common_words(tokens=feature_tokens, file_name=os.path.join(save_path, "most_common_words_features.png"), num_words=50)
    generate_wordcloud(feature_tokens, os.path.join(save_path, "wc_features.png"))

    # plot histogram of categories
    plot_histogram_of_categories(data, os.path.join(save_path, "histogram_of_categories.png"))
    print("done")

# WORDCLOUD
# Create and generate a word cloud image:
# wordcloud = WordCloud(width=800, height=400).generate(" ".join(words))
# # Display the generated image
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()