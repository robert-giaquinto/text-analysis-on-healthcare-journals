rm -f /home/srivbane/shared/caringbridge/data/dev/clean_journals/all_keys.tsv
python ~/text-analysis/src/clean_journal/journals_manager.py --sites_dir /home/srivbane/shared/caringbridge/data/dev/parsed_json/ --keys_file /home/srivbane/shared/caringbridge/data/dev/clean_journals/all_keys.tsv --n_workers 4 --outfile /home/srivbane/shared/caringbridge/data/dev/clean_journals/cleaned_journals_for_topics.txt --log --clean_method topic
echo "Remove all keys shards"
rm -rf /home/srivbane/shared/caringbridge/data/dev/clean_journals/all_keys_shards
