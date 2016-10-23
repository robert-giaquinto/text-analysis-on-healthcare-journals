#!/bin/bash
DD=/home/srivbane/shared/caringbridge/data/dev/
python -m src.topic_model.gensim_lda --journal_file ${DD}clean_journals/cleaned_journals_for_topics.txt --data_dir ${DD}topic_model --keep_n 25000 --num_test 100 --num_docs 828 --num_topics 10 --chunksizes 64 --threshold 0.1 --n_workers 4 --evals_per_pass 2 --log --rebuild
