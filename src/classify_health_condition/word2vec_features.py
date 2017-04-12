from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
import argparse


def load_feats(infile):
    """
    load the features dataset and give it appropriate column labels
    """
    df = pd.read_csv(infile, sep='\t', header=None)
    num_feats = df.shape[1] - 1
    col_names = ['site_id'] + ['vec' + str(t) for t in range(num_feats)]
    df.columns = col_names
    return df

def load_hc(infile):
    col_names = ['site_id', 'health_condition']
    df = pd.read_csv(infile, sep='\t', header=None, names=col_names)
    return df

def load_hc_and_keywords(infile):
    col_names = ['site_id', 'health_condition']
    keyword_cols = ["cancer", "surgery", "injury", "breast", "stroke", "brain",
                    "transplantation", "leukemia", "lung", "lymphoma", "heart", "pancreatic",
                    "ovarian", "bone", "kidney", "myeloma", "skin", "bladder", "esophageal", "blank"]
    hc_df = pd.read_csv(infile_file, sep='\t', header=None, names=col_names + keyword_cols)

def load_data(feat_file, hc_file):
    """
    load the two data files and merge them
    """
    feats = load_feats(feat_file)
    hcs = load_hc(hc_file)
    df = pd.merge(feats, hcs, on=["site_id"], how="inner", sort=False)
    return df



def split_data(df, test_size=0.25, random_seed=None):
    """
    split the data, but set aside the site_id in case we want that later
    """
    # split data into x (data features) and y (health condition)
    feats = [c for c in list(df.columns.values) if c != 'health_condition' and c != 'site_id']
    x = df[feats].values
    y = df['health_condition'].values

    # call function to split observations (e.g. 66% of data in train, 33% in test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state=random_seed)    
    return x_train, y_train, x_test, y_test


def run_grid_search(X, y, model, param_grid, cv=5, n_jobs=1, verbose=0):
    """
    Train a model for a range of model parameters (i.e. see how model
    performs depending on how complex you allow the model to become).
    """
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose,  return_train_score=False)
    grid_search.fit(X, y)
    res = grid_search.cv_results_
    print("\nAverage percent of labels correctly classified on hold-out set during cross-validation:")
    for ps, sco, sd, t in zip(res['params'], res['mean_test_score'], res['std_test_score'], res['mean_fit_time']):
        print("Params:", ps, "\tScore:", round(sco, 3), "\t(stdev: " + str(round(sd, 3)) + "), Time to fit:\t", t, "seconds")

    print("Model that generalizes best found to use:", grid_search.best_params_)
    return grid_search.best_estimator_



def main():
    parser = argparse.ArgumentParser(description='Main script for classifying health condition.')
    parser.add_argument('--topic_file', type=str, help='Full path to the topic features csv file.',
                        default='/home/srivbane/shared/caringbridge/data/classify_health_condition/topic_features_per_site.csv')
    parser.add_argument('--keywords_and_hc_file', type=str, help='Full path to the file with health conditions and keywords.',
                        default='/home/srivbane/shared/caringbridge/data/classify_health_condition/cond_keywords.txt')
    parser.add_argument('--cv', type=int, help='Number of cross-validation folders to use.', default=3)
    parser.add_argument('--n_jobs', type=int, help='Number of cores to use', default=1)
    parser.add_argument('--verbose', type=int, help='Level of verbosity parameter to pass to sklearn', default=1)
    args = parser.parse_args()
    print('main.py: Classify Health Condition')
    print(args)

    # set a random seed so that results are reproducible
    random_seed = 2016
    
    # load all the data
    print("loading data")
    df = load_data(args.topic_file, args.hc_file)
    print("Size of dataset:", df.shape)

    # split the data into training and test
    print("splitting a portion of the data off for testing later...")
    test_size = 0.2
    x_train, y_train, x_test, y_test = split_data(df, test_size, random_seed)


    # train logisitic regression model
    # logistic regression mean we are training a classifier (i.e. a label versus a real valued output)
    print("\nFinding optimal logistic regression")
    logit_param_grid = {'penalty': ['l2'],
                        'solver': ['lbfgs'],
                        'C': [0.001, 0.01, 0.1, 1, 10, 50, 100, 500]}
    logit = run_grid_search(X=x_train, y=y_train,
                            model=LogisticRegression(random_state=random_seed),
                            param_grid=logit_param_grid,
                            cv=cv, n_jobs=n_jobs, verbose=verbose)
    print("Best logistic regression performance on test set:", logit.score(x_test, y_test))
    

    # train random forest
    print("\nFinding optimal random forest")
    rf_param_grid = {"max_features": ['sqrt', 'log2'],
                     "max_depth": [None]}
    # do cross validation to determine optimal parameters to the model
    rf = run_grid_search(X=x_train, y=y_train,
                         model=RandomForestClassifier(n_estimators=100, random_state=random_seed),
                         param_grid=rf_param_grid,
                         cv=cv, n_jobs=n_jobs, verbose=verbose)
    # train a model on the full training set using the optimal parameters
    print("Best random forest performance on test set:", rf.score(x_test, y_test))

    
if __name__ == "__main__":
    main()
