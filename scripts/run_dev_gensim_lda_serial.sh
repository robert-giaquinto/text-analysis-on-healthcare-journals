#!/bin/bash
DD=/home/srivbane/shared/caringbridge/data/dev/
python -m src.topic_model.gensim_lda --journal_file ${DD}clean_journals/cleaned_journal5M_all.txt --data_dir ${DD}topic_model --keep_n 10000 --num_test 5000 --num_docs 5000000 --num_topics 100 --chunksizes 512 --threshold 1.0 --n_workers 1 --evals_per_pass 2 --log
