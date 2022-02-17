"""
Testing the functions in model.py file

Author: Ferruh Unlu
Date: 12/12/2021

Test 1 : Testing to see if pickle file can predict and returns rows

"""


import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import pytest
import pickle as pkl

from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

logging.basicConfig(
    filename='test_model1.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

'''
Loading the same data used in training the model so that the same data can be used in testing
'''


@pytest.fixture(scope='session')
def load_data():
    path_to_data = os.path.join(os.getcwd(), "data", "census.csv")
    data = pd.read_csv(path_to_data)

    return data


@pytest.fixture(scope='session')
def load_model_and_encoder():
    model = pkl.load(open(os.path.join(os.getcwd(), "starter", "model", "rf_model.pkl"), 'rb'))
    encoder = pkl.load(open(os.path.join(os.getcwd(), "starter", "model", "encoder.pkl"), 'rb'))
    lb = pkl.load(open(os.path.join(os.getcwd(), "starter", "model", "label_binarizer.pkl"), 'rb'))
    return model, encoder, lb


def test_train_model(
        load_data,
        n_estimators=100):

    # test train_models
    data = load_data

    train, test = train_test_split(data, test_size=0.20)
    x_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    try:
        model = train_model(x_train, y_train, n_estimators)
        logging.info("train models successfully ran: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing train_models. Issue occurred while testing the model")
        raise err

    logging.info(
        "train_models function test ended. Please review the log for details.")

    assert model is not None


def test_compute_model_metrics(load_data):

    data = load_data

    train, test = train_test_split(load_data, test_size=0.20)

    x_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    x_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    model = train_model(x_train, y_train, n_estimators=100)

    preds = inference(model, x_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    pr = precision * 100
    re = recall * 100

    try:
        assert pr > 70
        logging.info ("Precision is in desired range. Its value is {0}".format(pr))
    except AssertionError as err:
        logging.error(f"Precision is too low. Check your model and retrain as needed: {0}".format(pr))

    try:
        assert re > 70
        logging.info ("Recall is in desired range. Its value is {0}".format(re))
    except AssertionError as err:
        logging.error("Error is: {0}".format(err))
        logging.error(f"Recall is too low. Check your model and retrain as needed: {0}".format(re))


def test_inference(load_data):

    data = load_data

    train, test = train_test_split(load_data, test_size=0.20)

    x_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    x_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    model = train_model(x_train, y_train, n_estimators=100)

    preds = inference(model, x_test)
    try:
        assert len(preds) > 0
        logging.info("Model returned predictions as expected. No error detected.")
    except AssertionError as err:
        logging.error(f"Error is: {0}".format(err))

