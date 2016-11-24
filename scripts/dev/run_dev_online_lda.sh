#!/bin/bash
dd=/home/srivbane/shared/caringbridge/data/dev/topic_model/
vf=vocab_25000.dict
cf=train_bow_for_cleaned_journal100000_all_with_1000_test_docs_25000_terms.mm
python -m src.topic_model.online_lda --data_dir $dd --vocab_file $vf --corpus_file $cf --num_docs 90459 --num_topics 25 --batch_size 4096 --log
