#!/bin/bash
#python src/analyse_topic_model.py -iterations 25 > "output/lda_topic_model/topics_hparams1.txt"
#python src/analyse_topic_model.py -iterations 100 > "output/lda_topic_model/topics_hparams2.txt"
#python src/analyse_topic_model.py -iterations 25 -update_every 8 > "output/lda_topic_model/topics_hparams3.txt"
#python src/analyse_topic_model.py -iterations 50 -update_every 8 > "output/lda_topic_model/topics_hparams4.txt"
#python src/analyse_topic_model.py -iterations 100 -update_every 8 > "output/lda_topic_model/topics_hparams5.txt"
#python src/analyse_topic_model.py -iterations 25 -update_every 16 > "output/lda_topic_model/topics_hparams6.txt"
#python src/analyse_topic_model.py -iterations 50 -update_every 16 > "output/lda_topic_model/topics_hparams7.txt"
#python src/analyse_topic_model.py -iterations 100 -update_every 16 > "output/lda_topic_model/topics_hparams8.txt"
#python src/analyse_topic_model.py -iterations 25 -update_every 32 > "output/lda_topic_model/topics_hparams9.txt"
#python src/analyse_topic_model.py -iterations 50 -update_every 32 > "output/lda_topic_model/topics_hparams10.txt"
#python src/analyse_topic_model.py -iterations 100 -update_every 32 > "output/lda_topic_model/topics_hparams11.txt"

python src/analyse_topic_model.py > "output/lda_topic_model/topics_hparams0.txt"
#python src/analyse_topic_model.py -alpha "asymmetric"  > "output/lda_topic_model/topics_hparams1.txt"
#python src/analyse_topic_model.py -eta "auto"  > "output/lda_topic_model/topics_hparams2.txt"
#python src/analyse_topic_model.py -eta "auto" -alpha "asymmetric"  > "output/lda_topic_model/topics_hparams3.txt"
#python src/analyse_topic_model.py -decay 0.7 > "output/lda_topic_model/topics_hparams1.txt"
#python src/analyse_topic_model.py -decay 0.9 > "output/lda_topic_model/topics_hparams2.txt"
#python src/analyse_topic_model.py -offset 16 > "output/lda_topic_model/topics_hparams3.txt"
#python src/analyse_topic_model.py -offset 32 > "output/lda_topic_model/topics_hparams4.txt"
python src/analyse_topic_model.py -num_topics 1 > "output/lda_topic_model/topics_hparams1.txt"
python src/analyse_topic_model.py -num_topics 8 > "output/lda_topic_model/topics_hparams2.txt"
python src/analyse_topic_model.py -num_topics 10 > "output/lda_topic_model/topics_hparams3.txt"
python src/analyse_topic_model.py -num_topics 15 > "output/lda_topic_model/topics_hparams4.txt"