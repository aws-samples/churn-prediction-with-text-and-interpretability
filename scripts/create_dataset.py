""" Module for creating the dataset by combining categorical/numerical and text csv. files.

First, download categorical/numerical data into data folder as described in README.md, then:

Run in CLI example:
    'python create_dataset.py'

"""

import yaml
import pandas as pd
from pathlib import Path

with open("../model/params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)


def create_joint_dataset(
    df_categorical, 
    df_text
):
    df_text_no = df_text[df_text.churn == 'no'].reset_index(drop=True)
    df_text_yes = df_text[df_text.churn == 'yes'].reset_index(drop=True)

    df_cat_no = df_categorical[df_categorical.churn == 'no'].reset_index(drop=True)[:len(df_text_no)]
    df_cat_yes = df_categorical[df_categorical.churn == 'yes'].reset_index(drop=True)[:len(df_text_yes)]

    df_no = pd.concat([df_text_no, df_cat_no.iloc[:,:-1]], axis=1)
    df_yes = pd.concat([df_text_yes, df_cat_yes.iloc[:,:-1]], axis=1)

    df = pd.concat([df_no, df_yes], axis=0)

    # shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    return df


if __name__ == "__main__":
    data_dir = params['data_dir']
    
    # load data
    df_categorical = pd.read_csv(Path(data_dir, "train.csv"))
    df_text = pd.read_csv(Path(data_dir, "text.csv"))
    
    # create joint dataset
    df = create_joint_dataset(df_categorical, df_text)
    
    # save data
    df.to_csv(Path(data_dir, "churn_dataset.csv"), index=False)
