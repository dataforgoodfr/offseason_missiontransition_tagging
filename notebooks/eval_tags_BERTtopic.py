import pandas as pd
import re
from transformers import pipeline


human_eval = tags_df = pd.read_csv("human_eval_BERTtopic.csv")


# In[4]:

human_eval.drop(labels=['Unnamed: 0', 'Comparaison', 'Unnamed: 7'], axis=1, inplace=True)


# In[5]:

human_eval.rename(columns={'Unnamed: 6': 'label'}, inplace=True)


# In[6]:

print(human_eval.head())


# In[7]:

human_eval["zero_shot_score"] = [0.] * len(human_eval)


def get_topic_name(topic):
    topic = re.sub(r'\d+', '', topic)
    topic = topic.replace("_", " ")
    #topic = topic.replace("-", "")
    return topic


human_eval["clean_tag"] = human_eval["tag"].apply(lambda t: get_topic_name(t))

# Load 0-shot classifier
classifier = pipeline("zero-shot-classification", model="BaptisteDoyen/camembert-base-xnli")

# use zero-shot text classification on each couple (doc, topic_name)
hypothesis_template = "Ce texte est {}."
for index in list(human_eval.index):
   results = classifier(human_eval.loc[index]["description"], human_eval.loc[index]["clean_tag"], hypothesis_template=hypothesis_template)
   human_eval.at[index, "zero_shot_score"] = results["scores"][0]
print("done")

# assign a label for each zero-shot score
def get_zero_shot_label(score):
        if score < 0.4:
            label = 0
        elif 0.4 <= score <= 0.6:
            label = 0.5
        else:
            label = 1
        return label
human_eval["zero_shot_label"] = human_eval["zero_shot_score"].apply(lambda t: get_zero_shot_label(t))

human_eval.to_csv("evals_BERTtopic.csv")

