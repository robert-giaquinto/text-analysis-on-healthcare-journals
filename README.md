# Text Analysis Research #

## Getting Started ##
1. Clone the repo into your home directory:

   ```bash
   cd ~
   git clone https://github.com/robert-giaquinto/text-analysis.git
   ```

2. Setup python environment.

   We're currently using the version of python contained in "python2/2.7.12_anaconda4.1.1" on MSI, because it contains some needed packages that are sometimes difficult to install. To use this as the default version on python on MSI add these two lines to your ~/.bashrc file:

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


6. Treat the python code as a package. To do this navigate to the `text-analysis` directory and run:

   ```bash
   python setup.py develop
   ```


## Running Programs ##
Because of dependencies/imports between python files, the programs need to be run as modules with imports written as absolute imports. This means running the programs from the top `text-analysis` or `text-analysis/scripts` directory called as `python -m src.module_folder.python_file`.

As a result, some of the programs may not be run like scripts (e.g. `python myfile.py`). It may be easiest to create a bash script in the `text-analysis/scripts` folder for anything you may run interactively. Programs that will be submitted to the MSI queue need to be [PBS](https://www.msi.umn.edu/content/job-submission-and-scheduling-pbs-scripts) files. In the `scripts` folder there are examples of how to run these.


## Testing ##
To run all the tests in the test suite `text-analysis/src/tests/` from the root `text-analysis` folder run:

   ```bash
   python -m unittest discover -v
   ```


## Interacting with Journals ##
See example_cleaning_journals.py for how to iterate over all the journals and run the journal's ``clean_journal()`` on each journal method.

For an example of running this parallel check out the main() function in `text-analysis/src/clean_journal/journal_manager.py`.


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

