# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is based on Random Forest Classifier algorithm and it is trained to predict the salary whether would be greater or less than 50K annually based on the census data provided.
The only parameter used in this model is the number of the trees, which is n_estimators.

## Intended Use
The intended use of this model is to predict the salary for the people living in different countries. The salary is simplified to be greater or less than 50K only in a general rule. No other income is considered.

## Training Data

The training data is obtained by splitting the census data (80%) given all the details in the link below.

https://archive.ics.uci.edu/ml/datasets/census+income

The original data set has 32,561 rows, and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Evaluation Data
The same data whose description given above is used to evaluate the model.

## Metrics
The metrics obtained upon training the model to show the model's performance are given as follows:_

Precision:  0.724031007751938
Recall:  0.613263296126067
fbeta:  0.6640597227159616



## Ethical Considerations

Data used in this study may contain data that is obtained from the protected classes which may lead to bias. Also, the data is collected from not all the countries, hence may not work well in real production environment depending on the data coming.

## Caveats and Recommendations
The data should be collected from more countries for possibly better results. Also, different algorithms should be tried to rule out whether Random Forest Classifier would be the best to use for Production deployments.
