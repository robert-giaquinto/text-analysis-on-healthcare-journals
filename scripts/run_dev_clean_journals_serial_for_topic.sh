rm -f /home/srivbane/shared/caringbridge/data/dev/clean_journals/all_keys.tsv
python -m src.clean_journal.journals_manager --sites_dir /home/srivbane/shared/caringbridge/data/dev/parsed_json/ --keys_file /home/srivbane/shared/caringbridge/data/dev/clean_journals/all_keys.tsv --n_workers 1 --outfile /home/srivbane/shared/caringbridge/data/dev/clean_journals/dev_cleaned_journals_1worker.txt --log
echo "Remove all keys shards"
rm -rf /home/srivbane/shared/caringbridge/data/dev/clean_journals/all_keys_shards
