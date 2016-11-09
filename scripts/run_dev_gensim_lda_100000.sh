#!/bin/bash
DD=/home/srivbane/shared/caringbridge/data/dev/
python -m src.topic_model.gensim_lda --journal_file ${DD}clean_journals/cleaned_journal100000_all.txt --data_dir ${DD}topic_model --keep_n 10000 --num_test 5000 --num_docs 91459 --num_topics 25 --chunksizes 128 --threshold 0.1 --n_workers 8 --evals_per_pass 3 --log --rebuild
