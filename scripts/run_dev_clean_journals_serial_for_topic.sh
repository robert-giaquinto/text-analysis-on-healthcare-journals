rm -f /home/srivbane/shared/caringbridge/data/dev/clean_journals/all_keys.tsv
python ~/text-analysis/src/clean_journal/journals_manager.py -i /home/srivbane/shared/caringbridge/data/dev/parsed_json/ -k /home/srivbane/shared/caringbridge/data/dev/clean_journals/all_keys.tsv -n 1 -o /home/srivbane/shared/caringbridge/data/dev/clean_journals/cleaned_journals.txt --log
echo "Remove all keys shards"
rm -rf /home/srivbane/shared/caringbridge/data/dev/clean_journals/all_keys_shards
