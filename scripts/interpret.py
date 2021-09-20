""" Module for interpreting the trained churn prediction model with text. 

The parameters for model interpretation are stored in 'params.yaml'.

Run in CLI example:
    'python interpret.py --churn 1 --speaker Customer'

"""


import json
import yaml
import spacy
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn import preprocessing
from xgboost import XGBClassifier
from collections import defaultdict
from matplotlib import pyplot as plt
import preprocess
from preprocess import BertEncoder
import train

nlp = spacy.load('en_core_web_sm')

with open("../model/params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)

model_dir = params['model_dir']


def get_chats(
    df,
    churn=None,
    speaker=None
):
    """
    Args:
        df: dataframe with all data
        churn (int): 1 for churn, 0 for no churn, None (default) for all chats
        customer (str): 'Customer' for only customer chats,
                        'Agent' for only agent chats, 
                         None (default) for all chats
    
    Returns:
        list of chats (strings)
    """
    # convert labels to binary numeric
    df = preprocess.convert_label(df)
    # keep only churn/no churn chat logs
    if churn is not None:
        df = df[df.churn == churn]
    # drop short chat logs
    df = df[df.chat_log.apply(lambda x: len(str(x))>=5)]
    
    # select chats
    chat_logs = list(df['chat_log'])
    chat_logs = [i if isinstance(i, str) else "nan" for i in chat_logs]
    
    if speaker is not None:
        chats = []
        for chat in chat_logs:
            sents = chat.split('\n')
            cchat = []
            for sent in sents:
                if str(sent).split(':')[0] == speaker:
                    cchat.append(sent[10:])
            chats.append(' '.join(cchat))
    else:
        chats = chat_logs

    return chats, df


def get_keywords(texts):
    """Returns candidate keywords based on POS as well as original form of keyword.
    """
    candidate_pos = ['ADJ', 'VERB', 'NOUN', 'PROPN']
    keywords = []
    tokens = [] # for referencing keywords in original text later on
    for text in texts:
        text_keywords = []
        text_tokens = []
        doc = nlp(text)
        for token in doc:
            if token.pos_ in candidate_pos and token.is_stop is False:
                text_tokens.append(str(token))
                text_keywords.append(token.lemma_.lower())

        keywords.extend(text_keywords)
        tokens.extend(text_tokens)
        
    return keywords, tokens


def map_to_orig_tok(keywords, tokens):
    """Create dictionary mapping keywords to original tokens.
    """
    keywords_dict = defaultdict(list)
    for kw, t in zip(keywords, tokens):
        keywords_dict[kw].append(t)
    for kw, l in keywords_dict.items():
        keywords_dict[kw] = list(set(l))

    keywords_dict = dict(keywords_dict)
    
    return keywords_dict


def get_relevant_keywords(
    text,
    keywords_dict
):
    """Returns relevant keywords based on semantic similarity to class embedding.
    """
    # obtain class embedding
    textual_transformer = BertEncoder()
    textual_features = textual_transformer.transform(text)
    class_embedding = np.mean(np.array(textual_features), axis=0)
    
    # obtain relevant keywords
    unique_keywords = list(keywords_dict.keys())
    topn_relevant_keywords = params['topn_relevant_keywords']
    relevant_keywords, simMat = relevant_keywords_helper(unique_keywords,
                                                         class_embedding,
                                                         topn_relevant_keywords)
    return relevant_keywords, simMat


def relevant_keywords_helper(
    keywords,
    class_embedding,
    topn_relevant_keywords
):
    """Helper function for obtaining embedding similarity.
    """
    textual_transformer = BertEncoder()
    keyword_embeddings = textual_transformer.transform(keywords)

    simMatrix = np.dot(keyword_embeddings, class_embedding.T)
    d = {"keyword" : keywords, "sim" : list(simMatrix)}
    df_simMatrix = pd.DataFrame(d).sort_values(by="sim", ascending=False).reset_index(drop=True)
    relevant_keywords = list(df_simMatrix['keyword'])[:topn_relevant_keywords]

    return relevant_keywords, df_simMatrix


def prep_ablation(df):
    """Prepare data for making predictions."""
    
    # convert df to list of dicts
    data = df.to_json(orient="records")
    data = json.loads(data)
    
    # load model assets from training job
    model_assets = train.get_train_assets()
    #print('extracting features')
    numerical_features, categorical_features, textual_features = preprocess.extract_features(
        data,
        model_assets['numerical_feature_names'],
        model_assets['categorical_feature_names'],
        model_assets['textual_feature_names']
    )

    # extract labels
    _, _, _, label_name = preprocess.get_feature_names(df)
    labels = preprocess.extract_labels(
        data,
        label_name
    )

    # preprocess the data
    #print('transforming numerical_features')
    numerical_features = model_assets['numerical_transformer'].transform(numerical_features)
    #print('transforming categorical_features')
    categorical_features = model_assets['categorical_transformer'].transform(categorical_features)
    #print('transforming textual_features')
    textual_features = model_assets['textual_transformer'].transform(textual_features)

    #print('concatenating features')
    categorical_features = categorical_features.toarray()
    textual_features = np.array(textual_features)
    textual_features = textual_features.reshape(textual_features.shape[0], -1)
    features = np.concatenate([
        numerical_features,
        categorical_features,
        textual_features
    ], axis=1)

    return features, labels


def perform_ablation(
    df,
    keywords,
    keywords_dict
):
    """Predict w/ and w/o keywords ablated.
    """
    # select subset of relevant keywords
    frac_relevant_keywords = params['frac_relevant_keywords']
    topn = int(params['topn_relevant_keywords'] * frac_relevant_keywords)
    keywords_select = keywords[:topn]
    keywords_select_dict = {}
    for kw in keywords_select:
        keywords_select_dict[kw] = keywords_dict[kw]

    # loop through keywords and perform ablation analysis
    results_dict = {}
    for keyword, keywords_list in tqdm(keywords_select_dict.items()):
        # get portion of df where keywords occur
        df_incl = pd.DataFrame()
        df_excl = pd.DataFrame()
        for i, row in df.iterrows():
            for kw in keywords_list:
                if kw in row['chat_log']:
                    # df including keyword in chat
                    df_incl = df_incl.append(row)
                    # df excluding keyword in chat
                    chat_wkw = row['chat_log'].replace(kw, ' ')
                    row['chat_log'] = chat_wkw
                    df_excl = df_excl.append(row)

        # prep data incl/excl keyword in text (loads trained preprocessors)
        features_incl, labels_incl = prep_ablation(
            df_incl
        )

        features_excl, labels_excl = prep_ablation(
            df_excl
        )

        # predict using previously trained model
        pred_incl = train.predict(features_incl, labels_incl)
        pred_excl = train.predict(features_excl, labels_excl)

        # save results
        results_dict[keyword] = {'incl' : np.average(pred_incl),
                                 'excl' : np.average(pred_excl),
                                 'chg' : np.average(pred_excl) - np.average(pred_incl),
                                 'count' : len(pred_incl)}

    # store results
    results_df = pd.DataFrame.from_dict(results_dict, orient='index')
    results_df = results_df.sort_values(by=["chg", "count"], ascending=(True, False))
    results_df.to_csv(Path(model_dir, "ablation_results.csv"))
    
    return results_df


def get_important_keywords(
    simMat_df,
    marg_contr_df
):
    # merge dataframes
    marg_contr_df.insert(0, 'keyword', marg_contr_df.index)
    marg_contr_df = marg_contr_df.drop(['incl', 'excl'], axis=1)
    results_df = marg_contr_df.merge(simMat_df, how='left', on='keyword')

    # rescale metrics
    temp_df = results_df[['chg','count','sim']].copy()
    temp_df['chg'] = temp_df['chg'] * (-1)
    temp_df['count'] = np.log(temp_df['count']) # taking log transformation on keyword counts due to extreme outliers
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    temp_df = min_max_scaler.fit_transform(temp_df)
    temp_df = pd.DataFrame(temp_df, columns=['chg','count','sim'])
    
    # calculate weighted average
    w_marg_contr = params['w_marg_contr']
    w_count = params['w_count']
    w_sim = params['w_sim']
    temp_df['joint'] = temp_df['chg'] * w_marg_contr + temp_df['count'] * w_count + temp_df['sim'] * w_sim
    
    results_df['joint'] = temp_df['joint']
    results_df = results_df.sort_values(by="joint", ascending=False).reset_index(drop=True)
    results_df = results_df[['keyword','sim','chg','count','joint']]
    
    # store results
    results_df.to_csv(Path(model_dir, "important_keywords.csv"))
    
    return results_df


def obtain_context(
    chats_list,
    keyword,
    limit=3
):
    """Prints out limited number of chats where keyword occurs.
    """
    nlp = spacy.load('en_core_web_sm')
    counter = 0
    for chat in chats_list:
        doc = nlp(chat)
        for sent in doc.sents:
            if keyword in sent.text and counter < limit:
                print(sent.text)
                print('\n')
                counter+=1
        if counter >= limit:
            break

    return None


def train_xgb():
    """Train XGBoost model to explain categorical and numerical feature importance.
    """
    # load one-hot feature names
    filepath = Path(model_dir, "one_hot_feature_names.json")
    numerical_feature_names, categorical_feature_names, _ = preprocess.load_feature_names(filepath)
    one_hot_feature_names = numerical_feature_names + categorical_feature_names
    
    # load train data and exclude text embeddings
    train = pd.read_csv(Path(model_dir, "train.csv"))
    train_notext = train.iloc[:, :len(one_hot_feature_names)]
    labels = pd.read_csv(Path(model_dir, "labels.csv"))
    test = pd.read_csv(Path(model_dir, "test.csv"))
    test_notext = test.iloc[:, :len(one_hot_feature_names)]
    labels_test = pd.read_csv(Path(model_dir, "labels_test.csv"))

    # train XGBoost model
    xgb = XGBClassifier()
    xgb.fit(train_notext, labels)
    
    y_pred = xgb.predict_proba(test_notext)
    y_pred = [p[1] for p in y_pred]

    # plot important features
    topn = 10
    sorted_idx = xgb.feature_importances_.argsort()
    plt.barh(np.array(one_hot_feature_names)[sorted_idx[-topn:]],
             xgb.feature_importances_[sorted_idx[-topn:]])
    plt.title("Xgboost Feature Importance")
    plt.show()

    return y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--churn", type=int, default=1)
    parser.add_argument("--speaker", type=str, default='Customer')
    args = parser.parse_args()

    data_dir = params['data_dir'] 
    df = pd.read_csv(Path(data_dir, "churn_dataset.csv"))

    churn = args.churn
    speaker = args.speaker
    chats, df_sub = get_chats(df, churn, speaker)
    keywords, tokens = get_keywords(chats)
    keywords_dict = map_to_orig_tok(keywords, tokens)

    relevant_keywords, simMat_df = get_relevant_keywords(chats, keywords_dict)
    marg_contr_df = perform_ablation(df_sub, relevant_keywords, keywords_dict)

    get_important_keywords(simMat_df, marg_contr_df)
