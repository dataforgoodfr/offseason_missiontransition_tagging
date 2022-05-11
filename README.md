# offseason_missiontransition_tagging


### Requirements
The libraries needed to run the code are provided in the file requirements.txt.
* To run all the scripts from the origin repo (SMC-T-v2), run first the following command line: `export PYTHONPATH=src:${PYTHONPATH}`

### Download the data
* `python src/download_data.py -dataset "at"`
> This saves a json file in a 'data' folder named "AT_aides_full.json" with all Aides Territoires aides.
* `python src/download_data.py -dataset "mt"`
> This saves a json file in a 'data' folder named "MT_aides.json" with only Mission Transition aides.

### Exploring the data: statistics on the dataset
* To explore data on Mission Transition aides:`python src/explore_data.py - dataset "mt"`
* To explore data on Aides Territoires aides: `python src/explore_data.py - dataset "at"`
> This saves plots in a "plots" folder containing 5 png files: 
  - `histogram_of_categories.png` plots the histogram of categories for the given dataset 
  - `most_common_words_description.png` plots the histogram of most common words for the aides description. 
  - `wc_description.png` plots the wordclound of most common words for the aides description. 
  - `most_common_words_features.png` plots the histogram of most common words for the aides other textual features. 
  - `wc_features.png` plots the wordclound of most common words for the aides other textual features.

### Run a BERTopic topic model
* Run `python3 src/topic_modeling_BERT.py` to train a BERTopic model on the Aides dataset.
> This generates multiple files, including the model itself, a visualisation and results file: 
* By default, the folder `model` contains the trained model. Path to this folder can be changed with the argument `-bertopic_model_path`. 
* By default, the folder `berttopics` contains the results file (path to this folder can be changed with the argument `-bertopic_res_path`)
  1. The files `AT_results.csv` and `MT_results.csv` contain the predicted tags respectively on each description of all Aides Territoires database, and only Mission Transition database. 
  2. The files `AT_aides_full_with_tags.json` and `MT_aides_with_tags.json` are the original json files for AT and MT databases, in which the predicted tags have been tags, under the keys "list_tags". 
  3. The file `train_topics.csv` contains the full list of predicted tags. 
  4. The file `BERT_topics` is a log file containing several additional results information. 
  5. The folder `plot` contains a html visualization of the predicted tags.  
  
#### Get results from pretrained BERTTopic model
* The trained model (model used for the off-season) is available [here](https://drive.google.com/file/d/1414SWHgQVTNzIZsjdb93GX7Yc7-5blr3/view?usp=sharing)
* The associated results file for the pretrained model are available [here](https://drive.google.com/drive/folders/17kxgU4u1vdCqiEUbY3LPgs7NufNiqJbp?usp=sharing)
* To get bert topics results from a pretrained bert topic model, run `python3 src/topic_modeling_BERT.py -load_model $MODEL_PATH` with `$MODEL_PATH` a path to a pretrained topic_model.   



* Run `python3 src/topic_modeling_BERT.py -h` to learn more on the parameters.

### Run other topic models
See the instructions in this README to run other topic models contained in `src\other_topic_models`. 
