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


4. Installing the necessary packages.

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
         python ~/text-analysis/misc/download_nltk_data.py
         ```

   If you use other files from NLTK, add them to the list of things to download in the python file mentioned above.


## Interacting with Journals ##
See example_cleaning_journals.py for how to iterate over all the journals and run the journal's ``clean_journal()`` method.

For an example of running this parallel, stay tuned an example is in the works...


## Modules ##
* parse_journal - for parsing out the text from the journal.json files.