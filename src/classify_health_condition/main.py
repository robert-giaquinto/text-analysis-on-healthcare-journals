from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split


def load_topic_features(infile):
    df = pd.read_csv(infile)
    num_topics = df.shape[1] - 1
    col_names = ['site_id'] + ['topic' + str(t) for t in range(num_topics)]
    df.columns = col_names
    return df


def load_data(topic_file, keywords_and_hc_file):
    topic_df = load_topic_features(topic_file)

    hc_df = pd.read_csv(other_features_file)
    # TODO: rename the site_id column to site_id
    rval = pd.merge(rval, other, on=['site_id'], how='left', sort=False)

    hc = pd.read_csv(health_condition_file)
    # TODO: make sure a column is named site_id
    rval = pd.merge(rval, hc, on=['site_id'], how='left', sort=False)
    return rval



def split_data(df, test_size=0.25, random_seed=None):
    # split data into x (data features) and y (health condition)
    feats = [c for c in list(df.columns.values) if c != 'health_condition']
    x = df[feats]
    y = df['health_condition']

    # call function to split observations (e.g. 66% of data in train, 33% in test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state=random_seed)

    # save site id as an numpy array and remove it from the dataset
    site_id_train = x_train['site_id'].values
    site_id_test = x_test['site_id'].values
    x_train.drop('site_id', axis=1, inplace=True)
    x_test.drop('site_id', axis=1, inplace=True)
    # return all the partitions, convert to numpy arrays if they aren't already
    return site_id_train, x_train.values, y_train.values, site_id_test, x_test.values, y_test.values




def main():
    topic_file = '/home/srivbane/shared/caringbridge/data/dev/topic_model/topic_features_per_site.csv'
    other_features_file = None # this will point to mark's file
    health_condition_file = None # need to create this too, if it doesn't already exist. should have a column for site id and a column for health condition
    custom_conditions = ['custom', 'Other']

    random_seed = 2016
    
    # load all the data
    df = load_data(topic_file, other_features_file, health_condition_file)

    # for now we'll just work with sites that filled in health condition
    df = df[df['health_condition'].notnull()].reset_index()


    # save the custom answers for prediction later
    custom_df = df[df.health_condition.isin(custom_conditions),].reset_index()
    df = df[-df.health_condition.isin(custom_conditions),].reset_index()

    # split the data into training and test
    test_size = 0.25
    site_id_train, x_train, y_train, site_id_test, x_test, y_test = split_data(df, test_size, random_seed))
    
    # train lasso model
    lasso = LassoCV(n_alphas=100, cv=10, n_jobs=1, normalize=True)
    lasso.fit(x_train, y_train)
    lasso.score(x_test, y_test)


    # train random forest


    # train k-nearest neighbors



    # which is best?
