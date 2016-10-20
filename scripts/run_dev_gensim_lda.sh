#!/bin/bash
DD=/home/srivbane/shared/caringbridge/data/dev/
python -m src.topic_model.gensim_lda --journal_file ${DD}clean_journals/1M_journals_for_topic.txt --data_dir ${DD}topic_model --keep_n 25000 --num_test 5000 --num_docs 952212 --num_topics 50 --chunksizes 256 --threshold 0.1 --n_workers 16 --evals_per_pass 4 --log --rebuild
