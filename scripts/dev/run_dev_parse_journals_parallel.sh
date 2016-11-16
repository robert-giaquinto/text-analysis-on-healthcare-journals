dev_dir=/home/srivbane/shared/caringbridge/data/dev/
file=journal100000.json
python -m src.parse_journal.parse_manager --input_file ${dev_dir}${file} --output_dir ${dev_dir}parsed_json2/ --n_workers 8 --num_lines 100000 --log --clean

