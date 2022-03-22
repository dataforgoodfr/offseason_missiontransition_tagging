import os
from src.aides_dataset import AidesDataset, flatten_list
from nltk.probability import FreqDist
import pandas as pd
import matplotlib.pyplot as plt


def filter_tags_hist_per_category(df, category):
    df["category_bool"] = df["categories"].apply(
        lambda t: category in t)
    filtered_df = df[df["category_bool"] == True]
    tags = list(filtered_df["tag"].values.flatten())
    hist_tags = FreqDist(tags)
    hist_tags = dict(sorted(hist_tags.items(), key=lambda item: item[1], reverse=True))
    return hist_tags


def plot_tag_histograms(fdist, category, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(40, 30))
    ax.bar(fdist.keys(), fdist.values())
    ax.tick_params(axis='x', labelsize=36, labelrotation=90)
    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        labelleft=False)
    fontdict = {'fontsize': 36,
                'fontweight': 'bold'}
    ax.set_title("Category: {}".format(category), fontdict=fontdict)
    rects = ax.patches
    labels = [rect.get_height() for rect in rects]
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label,
                ha='center', va='bottom', fontsize=36)
    legend_properties = {'weight': 'bold'}
    ax.legend(prop=legend_properties)
    # plt.tight_layout()
    category_ = category.replace(" ", "-")
    category_ = category_.replace("/", "")
    file_name = os.path.join(out_path, 'tags_for_{}.png'.format(category_))
    fig.savefig(file_name, format='png', bbox_inches="tight")


if __name__ == '__main__':
    aides_dataset = AidesDataset("data/MT_aides.json")
    data_words = aides_dataset.get_data_words(["id", "description", "categories"])
    unfiltered_words = aides_dataset.get_unfiltered_data_words()
    short_descr = aides_dataset.get_short_descriptions(unfiltered_words)
    tokens = data_words.values.flatten()
    tokens = flatten_list(list(tokens))
    if not os.path.isdir("plots/categories"):
        os.makedirs("plots/categories")
    # plot_most_common_words(tokens=tokens, file_name="plots/most_common_words_LDA", num_words=50)

    data_words["categories"] = data_words.categories.apply(lambda t: list(set([e.split("|")[0] for e in t])))
    global_categories = data_words["categories"].values.flatten()
    global_categories = flatten_list(list(global_categories))

    # fdist_categories = FreqDist(global_categories)
    # fdist = dict(sorted(fdist_categories.items(), key=lambda item: item[1], reverse=True))

    results = pd.read_csv("notebooks/eval_BERTtopic_human+zeroshot.csv")
    results.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
    results["label"] = results["label"].apply(lambda x: x.replace(',', '.'))
    results = results.astype({'label': 'float64'})
    results["tag"] = results["tag"] .apply(lambda t: t.replace('__', ''))
    results["tag"] = results["tag"] .apply(lambda t: t.replace('5décrits', '5_décrits'))
    results.set_index("id", inplace=True)
    print(len(results))
    results_with_categories = results.merge(data_words, left_index=True, right_index=True, how='inner')
    print(len(results_with_categories))

    for category in list(set(global_categories)):
        hist_category = filter_tags_hist_per_category(results_with_categories, category)
        plot_tag_histograms(hist_category, category, "plots/categories")
    print("done")
