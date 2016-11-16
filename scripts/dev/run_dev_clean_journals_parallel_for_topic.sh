dev_dir=/home/srivbane/shared/caringbridge/data/dev/
python -m src.clean_journal.clean_manager --input_dir ${dev_dir}parsed_json2/ --output_file ${dev_dir}clean_journals/cleaned_journal100000_all.txt --clean_method topic --n_workers 8 --log

