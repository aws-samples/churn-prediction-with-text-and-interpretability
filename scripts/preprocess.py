""" Module for preparing the data for the churn prediction model with text. 

Run in CLI example:
    'python preprocess.py --test-size 0.33'

"""


import os
import sys
import json
import yaml
import joblib
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer


with open("../model/params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)

model_dir = params['model_dir']


class BertEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-nli-mean-tokens'):
        self.model = SentenceTransformer(model_name)
        self.model.parallel_tokenization = False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = []
        for sample in X:
            encodings = self.model.encode(sample)
            output.append(encodings)
        return output


def extract_labels(
    data,
    label_name
):
    labels = []
    for sample in data:
        value = sample[label_name]
        labels.append(value)
    labels = np.array(labels).astype('int')

    return labels.reshape(labels.shape[0],-1)


def convert_label(
    df
):
    df.churn = df.churn.replace("no", 0)
    df.churn = df.churn.replace("yes", 1)
    return df

def extract_numerical_features(
    sample,
    numerical_feature_names
):
    output = []
    for feature_name in numerical_feature_names:
        if feature_name in sample.keys():
            value = sample[feature_name]
            if value is None:
                value = np.nan
        else:
            value = np.nan
        output.append(value)
    return output


def extract_categorical_features(
    sample,
    categorical_feature_names
):
    output = []
    for feature_name in categorical_feature_names:
        if feature_name in sample.keys():
            value = sample[feature_name]
            if value is None:
                value = ""
        else:
            value = ""
        output.append(value)
    return output


def extract_textual_features(
    sample,
    textual_feature_names
):
    output = []
    for feature_name in textual_feature_names:
        if feature_name in sample.keys():
            value = sample[feature_name]
            if value is None:
                value = ""
        else:
            value = ""
        output.append(value)
    return output


def split_data(
    df,
    label_name,
    test_size
):
    """Splits data and creates json format.
    """
    X = df.drop(columns=[label_name], axis=1)
    y = df[label_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=123, stratify=y)
    
    train = pd.DataFrame(X_train, columns = X.columns)
    train[label_name] = y_train
    test = pd.DataFrame(X_test, columns = X.columns)
    test[label_name] = y_test
    
    # create list of dicts
    train = train.to_json(orient="records")
    train = json.loads(train)
    test = test.to_json(orient="records")
    test = json.loads(test)
    
    return train, test


def extract_features(
    data,
    numerical_feature_names,
    categorical_feature_names,
    textual_feature_names
):
    """extract features by given feature names.
    """
    numerical_features = []
    categorical_features = []
    textual_features = []
    for sample in data:
        num_feat = extract_numerical_features(sample, numerical_feature_names)
        numerical_features.append(num_feat)
        cat_feat = extract_categorical_features(sample, categorical_feature_names)
        categorical_features.append(cat_feat)
        text_feat = extract_textual_features(sample, textual_feature_names)
        textual_features.append(text_feat)
        
    textual_features = [i if isinstance(i[0], str) else ["nan"] for i in textual_features]
    textual_features = [i[0] for i in textual_features]

    return numerical_features, categorical_features, textual_features


def save_feature_names(
    numerical_feature_names,
    categorical_feature_names,
    textual_feature_names,
    filepath
):
    feature_names = {
        'numerical': numerical_feature_names,
        'categorical': categorical_feature_names,
        'textual': textual_feature_names
    }
    with open(filepath, 'w') as f:
        json.dump(feature_names, f)


def load_feature_names(filepath):
    with open(filepath, 'r') as f:
        feature_names = json.load(f)
    numerical_feature_names = feature_names['numerical']
    categorical_feature_names = feature_names['categorical']
    textual_feature_names = feature_names['textual']
    return numerical_feature_names, categorical_feature_names, textual_feature_names


def get_feature_names(
    df
):
    num_columns = df.select_dtypes(include=np.number).columns.tolist()
    numerical_feature_names = [i for i in num_columns if i not in ['churn']]

    cat_columns = df.select_dtypes(include='object').columns.tolist()
    categorical_feature_names = [i for i in cat_columns if i not in ['chat_log']]

    textual_feature_names = ['chat_log']
    label_name = 'churn'

    return numerical_feature_names, categorical_feature_names, textual_feature_names, label_name
    

def prep_data(
    df, use_existing=False, test_size=0.33
):  
    """
    Args:
        df: Pandas dataframe with raw data
        use_existing: Set to True if you want to use locally stored, 
        already prepared train/test data. Set to False if you want 
        to rerun the data preparation pipeline.
    Returns:
        Train and test data as well as train labels and test labels.
    """
    # if prepared data exists, don't prepare again if use_existing set to True
    train_file = Path(model_dir, 'train.csv')
    labels_file = Path(model_dir, 'labels.csv')
    test_file = Path(model_dir, 'test.csv')
    labels_test_file = Path(model_dir, 'labels_test.csv')
    feature_names_file = Path(model_dir, "feature_names.json")
    oh_feature_names_file = Path(model_dir, "one_hot_feature_names.json")
    all_file_paths = [train_file, labels_file, test_file, labels_test_file,
                     feature_names_file, oh_feature_names_file]
    
    if use_existing == True and all(file.exists() for file in all_file_paths):
        features = np.array(pd.read_csv('../model/train.csv'))
        labels = np.array(pd.read_csv('../model/labels.csv'))
        features_test = np.array(pd.read_csv('../model/test.csv'))
        labels_test = np.array(pd.read_csv('../model/labels_test.csv'))
        print("Using already prepared data.")

    else:
        print("Running data preparation pipeline...")
        # convert label to binary numeric
        df = convert_label(df)

        # extract feature names
        numerical_feature_names, categorical_feature_names, textual_feature_names, label_name = get_feature_names(
            df
        )
        # train/test split and convert to json format (list of dicts)
        train, test = split_data(
            df,
            label_name,
            test_size
        )
        # extract features & label
        print('extracting features')
        numerical_features, categorical_features, textual_features = extract_features(
            train,
            numerical_feature_names,
            categorical_feature_names,
            textual_feature_names
        )
        labels = extract_labels(
            train,
            label_name
        )
        # extract features & label (for test data)
        numerical_features_test, categorical_features_test, textual_features_test = extract_features(
            test,
            numerical_feature_names,
            categorical_feature_names,
            textual_feature_names
        )
        labels_test = extract_labels(
            test,
            label_name
        )
        # define preprocessors
        print('defining preprocessors')
        numerical_transformer = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=True)
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        textual_transformer = BertEncoder()

        # fit preprocessors
        print('fitting numerical_transformer')
        numerical_transformer.fit(numerical_features)
        print('saving numerical_transformer')
        joblib.dump(numerical_transformer, Path(model_dir, "numerical_transformer.joblib"))
        print('fitting categorical_transformer')
        categorical_transformer.fit(categorical_features)
        print('saving categorical_transformer')
        joblib.dump(categorical_transformer, Path(model_dir, "categorical_transformer.joblib"))

        # transform features
        print('transforming numerical_features')
        numerical_features = numerical_transformer.transform(numerical_features)
        print('transforming categorical_features')
        categorical_features = categorical_transformer.transform(categorical_features)
        print('transforming textual_features')
        textual_features = textual_transformer.transform(textual_features)

        # transform features (for test data)
        print('transforming numerical_features_test')
        numerical_features_test = numerical_transformer.transform(numerical_features_test)
        print('transforming categorical_features_test')
        categorical_features_test = categorical_transformer.transform(categorical_features_test)
        print('transforming textual_features_test')
        textual_features_test = textual_transformer.transform(textual_features_test)

        # concat features
        print('concatenating features')
        categorical_features = categorical_features.toarray()
        textual_features = np.array(textual_features)
        textual_features = textual_features.reshape(textual_features.shape[0], -1)
        features = np.concatenate([
            numerical_features,
            categorical_features,
            textual_features
        ], axis=1)

        # concat features (test data)
        print('concatenating features of test data')
        categorical_features_test = categorical_features_test.toarray()
        textual_features_test = np.array(textual_features_test)
        textual_features_test = textual_features_test.reshape(textual_features_test.shape[0], -1)
        features_test = np.concatenate([
            numerical_features_test,
            categorical_features_test,
            textual_features_test
        ], axis=1)

        # save to disk
        pd.DataFrame(features).to_csv(Path(model_dir, "train.csv"), index=False)
        pd.DataFrame(labels).to_csv(Path(model_dir, "labels.csv"), index=False)
        pd.DataFrame(features_test).to_csv(Path(model_dir, "test.csv"), index=False)
        pd.DataFrame(labels_test).to_csv(Path(model_dir, "labels_test.csv"), index=False)

        save_feature_names(
            numerical_feature_names,
            categorical_feature_names,
            textual_feature_names,
            Path(model_dir, "feature_names.json")
        )
        # one-hot encoded feature names (for feat_imp)
        save_feature_names(
            numerical_feature_names,
            categorical_transformer.get_feature_names().tolist(),
            textual_feature_names,
            Path(model_dir, "one_hot_feature_names.json")
        )

    return features, features_test, labels, labels_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-existing", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.33)
    args = parser.parse_args()
    
    data_dir = params['data_dir']
    df = pd.read_csv(Path(data_dir, "churn_dataset.csv"))
    
    if args.use_existing:
        use_existing = True
    else:
        use_existing = False
    test_size = args.test_size
    
    prep_data(df, use_existing, test_size)
