python ~/text-analysis/src/parse_journal/parse_manager.py --input_file /home/srivbane/shared/caringbridge/data/dev/journal.json --output_dir /home/srivbane/shared/caringbridge/data/dev/parsed_json/ --n_workers 1 --num_lines 10000 --log --clean
echo "Removing shard files (since this is dev mode)."
rm -rf /home/srivbane/shared/caringbridge/data/dev/journal_shards
