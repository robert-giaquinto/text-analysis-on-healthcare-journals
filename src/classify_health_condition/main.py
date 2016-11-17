from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def load_topic_features(infile):
    """
    load the topic features dataset and give it appropriate column labels
    """
    df = pd.read_csv(infile)
    num_topics = df.shape[1] - 1
    col_names = ['site_id'] + ['topic' + str(t) for t in range(num_topics)]
    df.columns = col_names
    return df


def load_data(topic_file, keywords_and_hc_file):
    """
    load the two data files and merge them
    """
    topic_df = load_topic_features(topic_file)
    print(topic_df.head())

    col_names = ['site_id', 'health_condition',
                 "cancer", "surgery", "injury", "breast", "stroke", "brain",
                 "transplantation", "leukemia", "lung", "lymphoma", "heart", "pancreatic",
                 "ovarian", "bone", "kidney", "myeloma", "skin", "bladder", "esophageal", "blank"]
    hc_df = pd.read_csv(keywords_and_hc_file, sep='\t', header=None, names=col_names)
    hc_df.drop('blank', axis=1, inplace=True)
    print(hc_df.head())

    rval = pd.merge(topic_df, hc_df, on=['site_id'], how='inner', sort=False)
    return rval



def split_data(df, test_size=0.25, random_seed=None):
    """
    split the data, but set aside the site_id in case we want that later
    """
    # split data into x (data features) and y (health condition)
    feats = [c for c in list(df.columns.values) if c != 'health_condition']
    x = df[feats]
    y = df['health_condition']

    # convert y from labels to a binary matrix (each column shows True/False that column is a specific label)
    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer
    #lb = preprocessing.LabelBinarizer()
    #y_mat = lb.fit_transform(y)

    
    # call function to split observations (e.g. 66% of data in train, 33% in test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state=random_seed)
    
    # save site id as an numpy array and remove it from the dataset
    site_id_train = x_train['site_id'].values
    site_id_test = x_test['site_id'].values
    x_train.drop('site_id', axis=1, inplace=True)
    x_test.drop('site_id', axis=1, inplace=True)
    # return all the partitions, convert to numpy arrays if they aren't already
    return site_id_train, x_train.values, y_train, site_id_test, x_test.values, y_test




def main():
    topic_file = '/home/srivbane/shared/caringbridge/data/classify_health_condition/topic_features_per_site.csv'
    keywords_and_hc_file = '/home/srivbane/shared/caringbridge/data/classify_health_condition/cond_keywords.txt'

    # set a random seed so that results are reproducible
    random_seed = 2016
    
    # load all the data
    print("loading data")
    df = load_data(topic_file, keywords_and_hc_file)
    print("Size of dataset:", df.shape)

    # save the custom answers for prediction later
    custom_df = df.loc[df.health_condition == 'custom',].reset_index()
    print("size of custom df:", custom_df.shape)
    df = df.loc[df.health_condition != 'custom',].reset_index()
    print("Size of data available for training:", df.shape)

    # split the data into training and test
    print("splitting a portion of the data off for testing later...")
    test_size = 0.25
    site_id_train, x_train, y_train, site_id_test, x_test, y_test = split_data(df, test_size, random_seed)
    
    # train lasso model
    # logistic regression mean we are training a classifier (i.e. a label versus a real valued output)
    # using penalty = 'L1' means this will do the lasso algorithm we talked about
    print("Training lasso")
    lasso = LogisticRegressionCV(Cs=25, cv=5, penalty = 'l1', solver='liblinear', n_jobs=2)
    lasso.fit(x_train, y_train)
    lasso.score(x_test, y_test)

    # todo visualize performance

    

    # train random forest


    # train k-nearest neighbors



    # which is best?
if __name__ == "__main__":
    main()
