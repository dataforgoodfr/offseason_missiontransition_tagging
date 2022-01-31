# offseason_missiontransition_tagging


### Requirements
The libraries needed to run the code are provided in the file requirements.txt.
* To run all the scripts from the origin repo (SMC-T-v2), run first the following command line: `export PYTHONPATH=src:${PYTHONPATH}`

### Download the data
* `python src/download_data.py -dataset "at"`
> This saves a json file in a 'data' folder named "AT_aides_full.json" with all Aides Territoires aides.
* `python src/download_data.py -dataset "mt"`
> This saves a json file in a 'data' folder named "MT_aides.json" with only Mission Transition aides.

### Statistics on the dataset
* `python src/aides_dataset.py`
> This saves plots in a "plots" folder containing the most frequent words for the dataset descriptions.

### Run a LDA topic model
* First fill up the hyper-parameters you want in the csv file "data/csv_model_hparams.csv".
* Then run: `python src/topic_model.py`
> This saves a file "results_hparams.csv" in a folder "output/lda_topic_model".

### Run a BERTopic topic model
* Run `python3 src/topic_modeling_BERT.py` to train a BERTopic model on the Aides dataset.
> This generates multiple files, including the model itself, a visualisation and text outputs.
* Run `python3 src/topic_modeling_BERT.py -h` to learn more on the parameters.

### Run zeroshot text classification on the dataset
* `python src\zeroshot_text_classification.py`
> This saves 2 csv files in a folder "output/zeroshot_classif", "results.csv" and "tags_per_descr.csv".
