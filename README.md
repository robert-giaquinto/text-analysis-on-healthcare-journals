# Text Analysis Research #


## Introduction ##
This repository is a shared project among researchers at the University of Minnesota investigating health journeys through the scrope of a large corpus of health journals written by patients and caregivers. We explore these texts with topic modeling, sentiment analysis, and other machine learning and natural language processing techniques.   


## Getting Started ##
1. Clone the repo into your home directory:

   ```bash
   cd ~
   git clone https://github.com/robert-giaquinto/text-analysis.git
   ```

2. Setup python environment.

   We're currently using the version of python contained in "python2/2.7.12_anaconda4.1.1" on MSI (Minnesota Supercomputing Institute), because it contains some needed packages that are sometimes difficult to install. To use this as the default version on python on MSI add these two lines to your ~/.bashrc file:

   ```bash
   module unload python
   module load python2/2.7.12_anaconda4.1.1
   ```

3. Virtual environments.

   MSI allows us to install python modules into [virtual envirnments](http://docs.python-guide.org/en/latest/dev/virtualenvs/ "Background information on virtualenv"). These are also nice because it makes it easy for all to use the same python modules without issues of module dependencies caused by other projects. I prefer keeping the virtualenv in my home folder, but you can also put it in the project folder (just make sure not to push it to github). To create the virtualenv, go to your home folder and run:

   ```bash
   virtualenv venv
   ```

   To activate the virtualenv run:

   ```bash
   source ~/venv/bin/activate
   ```

   To use automatically load this virtualenv everytime you login (recommended, if you aren't working on other MSI projects) add the previous line of code to your ~/.bashrc file.


4. Installing the necessary python packages.

   A requirements.txt file, listing all packages used for this project is included in the repository. To install them first make sure your virtual environment is activated, then run the following line of code:

   ```bash
   pip install -r ~/text-analysis/requirements.txt
   ```

   If the are other packages you want to use, install them and update the requirements.txt file with this command:

   ```bash
   pip freeze > ~/text-analysis/requirements.txt
   ```

5. Installing NLTK data.

   There is a file in the misc folder for downloading all the data for NLTK. To run it use this:

   ```bash
   python ~/text-analysis/download_nltk_data.py
   ```

   If you use other files from NLTK, add them to the list of things to download in the python file mentioned above.


6. Treat the python code as a package. This is done to allow for absolute imports, which make it easy to load python files can be spread out in different folders. To do this navigate to the `text-analysis` directory and run:

   ```bash
   python setup.py develop
   ```


## Running Programs ##
It may be easiest to create bash scripts in `text-analysis/script` to run any programs you may run more than once. Programs that will be submitted to the MSI queue need to be [PBS](https://www.msi.umn.edu/content/job-submission-and-scheduling-pbs-scripts) files. In the `scripts` folder there are examples of how to run these.

Because the python files are spread out over multiple directories, the programs use absolute imports to load other modules. This means running the programs with the `-m` flag, for example  `python -m src.module_folder.python_file`.

Feel free to run any of the existing programs with the word 'dev' in the title, however the scripts to parse the giant json file and clean the text should only be run once because they may take a little time to run (i.e. `run_parse_journal.pbs`, `run_clean_journal_for_topic.pbs`, etc.).


## Testing ##
To run all the tests in the test suite `text-analysis/src/tests/` from the root `text-analysis` folder run:

```bash
python -m unittest discover -v
```




## Interacting with Journals ##
For data containing journal keys and their original text see files in:

```bash
home/srivbane/shared/.../data/parsed_json/
```

Feel free to concatenate these file shards into one file if needed (i.e. `cat parsed_journal_01_of_48.txt ... parsed_journal_48_of_48.txt > all_parsed_journals.txt`).

For all journal keys and the cleaned up text (all lowercase, lemmatized, no punctuation, etc.) see:

```bash
/.../data/clean_journals
```


## Project Structure ##
* `examples/` - A few simple python programs showing how to interact with various classes.
* `scripts/` - Bash scripts for running programs.
* `src/` - Directory containing various sub-packages of the project and any files shared across sub-packages.
   * `src/parse_journal/` - For parsing out the text from the journal.json files.
   * `src/clean_journal/` - For iterating over each journal, cleaning journal text, and creating flat files of all cleaned journal data.
   * `src/topic_model/` - For implementing various topic modeling algorithms.
   * `src/tests/` - Test suite for the all code. To run all tests navigate to ``~/text-analysis`` and run:

       ```bash
       python setup.py test
       ```
       or
       ```bash
       python -m unittest discover -v
       ```

