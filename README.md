## Churn Prediction with Text and Interpretability

Customer churn, the loss of current customers, is a problem faced by a wide range of companies. When trying to retain customers, it is in a company’s best interest to focus their efforts on customers who are more likely to leave, but companies need a way to detect customers who are likely to leave before they have decided to leave. Users prone to churn often leave clues to their disposition in user behavior and customer support chat logs which can be detected and understood using Natural Language Processing (NLP) tools.

Here, we demonstrate how to build a churn prediction model that leverages both text and structured data (numerical and categorical) which we call a bi-modal model architecture. We use Amazon SageMaker to prepare, build, and train the model. Detecting customers who are likely to churn is only part of the battle, finding the root cause is an essential part of actually solving the issue. Since we are not only interested in the likelihood of a customer churning but also in the driving factors, we complement the prediction model with an analysis into feature importance for both text and non-text inputs.

The categorical and numerical data is from Kaggle: Customer Churn Prediction 2020 and was combined with a synthetic text dataset we created using GPT-2.

## Blog Post

[Medium blog post](https://medium.com/@daniel.herkert1/customer-churn-prediction-with-text-and-interpretability-bd3d57af34b1)

## Installation

```
git clone https://github.com/aws-samples/churn-prediction-with-text-and-interpretability.git
conda create -n py39 python=3.9
conda activate py39
cd churn-prediction-with-text-and-interpretability
pip install -r requirements.txt
```

## Download categorical/numerical data and combine with synthetic text data

1. Download categorical/numerical data - [Customer Churn Prediction 2020](https://www.kaggle.com/c/customer-churn-prediction-2020/data) 
    May require Kaggle account.
    Download train.csv and store in data folder.

2. Run script to combine categorical data with synthetic text data (../scripts)
    ```
    python create_dataset.py
    ```

## Run in Notebook

An example notebook to run the entire pipeline and print/visualize the results in included in ../notebook.

## Run in Terminal

The python scripts to prepare the data, train and evaluate the model, as well as interpret the model, are stored in ../scripts.
The parameters used for training and interpreting the model are stored in ../model/params.yaml.


1. Prepare the data:
    ```
    python preprocess.py
    ```
2. Train and evaluate the model:
    ```
    python train.py
    ```
3. Interpret the trained model (text):
    ```
    python interpret.py --churn 1 --speaker Customer
    ```

## Credits

* Packages:
    * [Spacy](https://spacy.io/usage/linguistic-features/) 
    * [PyTorch](https://pytorch.org/)
    * [XGBoost](https://xgboost.readthedocs.io/en/latest/)
    * [Hugging Face Sentence Transformers](https://huggingface.co/sentence-transformers)

* Datasets:
    * [Customer Churn Prediction 2020](https://www.kaggle.com/c/customer-churn-prediction-2020/data) (with synthetic text dataset)
    
* Models:
    * GPT2, Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
    * BERT, Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
    * Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks, Reimers, Nils and Gurevych, Iryna

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

