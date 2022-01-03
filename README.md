# offseason_missiontransition_tagging


### Requirements
The libraries needed to run the code are provided in the file requirements.txt.
* To run all the scripts from the origin repo (SMC-T-v2), run first the following command line: `export PYTHONPATH=src:${PYTHONPATH}`

### Download the data
* `python src/download_data.py`
> This saves a json file in a 'data' folder named "AT_aides_full.json"

### Statistics on the dataset
* `python src/aides_dataset.py`
> This saves plots in a "plots" folder.

### Run a LDA topic model
* First fill up the hyper-parameters you want in the csv file "data/csv_model_hparams.csv". 
* Then run: `python src/topic_model.py`
> This saves a file "results_hparams.csv" in a folder "output/lda_topic_model". 

### Run zeroshot text classification on the dataset
* `python src\zeroshot_text_classification.py`
> This saves 2 csv files in a folder "output/zeroshot_classif"
