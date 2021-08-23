""" Module for training the churn prediction model with text. 

Training parameters are stored in 'params.yaml'.

Run in CLI example:
    'python train.py'

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
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import preprocess
from preprocess import BertEncoder


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


with open("../model/params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)

model_dir = params['model_dir']

    
class Net(nn.Module):
    def __init__(self, x1_size, x2_size):
        super(Net, self).__init__()
        self.batch_norm = nn.BatchNorm1d(x1_size)
        self.fc1 = nn.Linear(x1_size, 10)
        self.fc2 = nn.Linear(10 + x2_size, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x1, x2):
        x1 = self.batch_norm(x1)
        x1 = F.relu(self.fc1(x1))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x12 = torch.cat((x1.view(x1.size(0), -1),
                        x2.view(x2.size(0), -1)), dim=1)
        x12 = F.dropout(x12, p=0.1, training=self.training)
        x12 = self.fc2(x12)
        out = self.fc3(x12)
        return out


def train(
    X,
    y,
    X_test,
    y_test
):
    # get parameters
    batch_size = params['batch_size']
    batch_size_test = params['batch_size_test']
    epochs = params['epochs']
    pos_weight = params['pos_weight']
    lr = params['lr']
    momentum = params['momentum']
    
    # prepare training job
    X = np.array(X)
    y = np.array(y)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    training_data = TensorDataset( Tensor(X), Tensor(y) )
    train_loader = DataLoader(training_data, batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4)
    test_data = TensorDataset( Tensor(X_test), Tensor(y_test) )
    test_loader = DataLoader(test_data, batch_size=batch_size_test,
                                              shuffle=True,
                                              num_workers=4)
    
    # get size of num/cat & text data
    numerical_feature_names, categorical_feature_names, _ = preprocess.load_feature_names(Path(model_dir, "one_hot_feature_names.json"))
    number_cat_num_features = len(numerical_feature_names) + len(categorical_feature_names)
    x1_size = X[:, :number_cat_num_features].shape[1]
    x2_size = X[:, number_cat_num_features:].shape[1]

    model = Net(x1_size, x2_size)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float))
    #criterion = nn.BCELoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    # train NN model
    scores_df = pd.DataFrame()
    train_scores = []
    test_scores = []
    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        model.train()
        print("starting epoch: ", epoch)
        
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # zero the parameter gradients
            optimizer.zero_grad()

            # split data inputs into cat/num features and text features
            x1 = data[:, :number_cat_num_features]
            x2 = data[:, number_cat_num_features:]
            
            # forward + backward + optimize
            output = model(x1, x2)
            loss = criterion(output, target) #target.type_as(output)) #labels_batch.long())
            loss.backward()
            optimizer.step()

        # training performance after each epoch
        with torch.no_grad():
            output = model(Tensor(X[:, :number_cat_num_features]), Tensor(X[:, number_cat_num_features:])).reshape(-1)
            preds = torch.sigmoid(output)
            train_score = roc_auc_score(Tensor(y), preds)
            train_scores.append(train_score)
            logger.info('Train Epoch: {}, train-auc-score: {:.4f}'.format(epoch, train_score))

        # test performance after each epoch
        test_score = test(model, test_loader)
        test_scores.append(test_score)
    
    # save scores
    print('saving scores')
    scores_df['train_scores'] = train_scores
    scores_df['test_scores'] = test_scores
    scores_df.to_csv(Path(model_dir, 'training_scores.csv'), index=False)
    
    # save model
    print('saving model')
    torch.save(model.state_dict(), Path(model_dir, 'model.pth'))

    return None


def test(
    model,
    test_loader
):
    model.eval()
    correct = 0
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for data, targets in test_loader:
            
            # split data inputs
            numerical_feature_names, categorical_feature_names, _ = preprocess.load_feature_names(Path(model_dir, "one_hot_feature_names.json"))
            number_cat_num_features = len(numerical_feature_names) + len(categorical_feature_names)
            x1 = data[:, :number_cat_num_features]
            x2 = data[:, number_cat_num_features:]
            
            output = model(x1, x2).reshape(-1)
            preds = torch.sigmoid(output)
            preds_all.extend(preds)
            targets_all.extend(targets)
        test_score = roc_auc_score(Tensor(targets_all), Tensor(preds_all))

    logger.info('test_auc_score: {:.4f}'.format(test_score))

    return test_score


def get_train_assets(
):
    #print('loading feature_names')
    numerical_feature_names, categorical_feature_names, textual_feature_names = preprocess.load_feature_names(Path(model_dir, "feature_names.json"))
    #print('loading numerical_transformer')
    numerical_transformer = joblib.load(Path(model_dir, "numerical_transformer.joblib"))
    #print('loading categorical_transformer')
    categorical_transformer = joblib.load(Path(model_dir, "categorical_transformer.joblib"))
    #print('loading textual_transformer')
    textual_transformer = BertEncoder()

    model_assets = {
        'numerical_feature_names': numerical_feature_names,
        'numerical_transformer': numerical_transformer,
        'categorical_feature_names': categorical_feature_names,
        'categorical_transformer': categorical_transformer,
        'textual_feature_names': textual_feature_names,
        'textual_transformer': textual_transformer
    }
    return model_assets


def predict(
    features,
    labels
):
    # get size of num/cat & text data to specify neural net
    numerical_feature_names, categorical_feature_names, _ = preprocess.load_feature_names(Path(model_dir, "one_hot_feature_names.json"))
    number_cat_num_features = len(numerical_feature_names) + len(categorical_feature_names)
    x1_size = features[:, :number_cat_num_features].shape[1]
    x2_size = features[:, number_cat_num_features:].shape[1]
    
    model = Net(x1_size, x2_size)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.eval()
    with torch.no_grad():
        output = model(Tensor(features[:, :number_cat_num_features]), Tensor(features[:, number_cat_num_features:])).reshape(-1)
        preds = torch.sigmoid(output)

    return preds


def plot_train_stats(
):
    scores_df = pd.read_csv(Path(model_dir, 'scores.csv'))
    scores_df.plot(xlabel='Epochs', ylabel='AUC Score', title='AUC Score')
    return None


def plot_pr_curve(
    features,
    labels
):
    preds = predict(features, labels)
    precisions, recalls, thresholds = precision_recall_curve(labels, preds)
    roc_auc = roc_auc_score(labels, preds)

    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(recalls, precisions, label='Model w/ text: AUC = %0.2f' % roc_auc)
    ax.plot([1, 0], [0, 1],'--')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left')
    plt.title('Precision-Recall Curve')
    plt.show()
    return None


def plot_roc_curve(
    features,
    labels
):
    preds = predict(features, labels)
    fpr, tpr, threshold = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return None


if __name__ == "__main__":

    model_dir = params['model_dir']

    X_train = pd.read_csv(Path(model_dir, "train.csv"))
    y_train = pd.read_csv(Path(model_dir, "labels.csv"))
    X_test = pd.read_csv(Path(model_dir, "test.csv"))
    y_test = pd.read_csv(Path(model_dir, "labels_test.csv"))

    train(X=X_train, y=y_train, X_test=X_test, y_test=y_test)
