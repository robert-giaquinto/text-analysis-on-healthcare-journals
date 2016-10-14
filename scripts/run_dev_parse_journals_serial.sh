python -m src.parse_journal.parse_manager --input_file /home/srivbane/shared/caringbridge/data/dev/journal.json --output_dir /home/srivbane/shared/caringbridge/data/dev/parsed_json/ --n_workers 1 --num_lines 1000 --log --clean
echo "Removing shard files (since this is dev mode)."
rm -rf /home/srivbane/shared/caringbridge/data/dev/journal_shards
