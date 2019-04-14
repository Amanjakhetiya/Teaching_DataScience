import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.stats import skew
from tensorflow.contrib import learn
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.getLogger().setLevel(logging.INFO)
from sklearn.metrics import confusion_matrix, classification_report

import random
random.seed(42) # to sample data the same way

def prepData(df):
    # Let's fill in the "holes" with the means on numerical attributes
    df.fillna(df.mean(), inplace=True)

    # Building a Predictive Model in Python
    # sklearn requires all inputs to be numeric,
    # convert all our categorical variables into numeric by encoding the categories.
    nonnumeric_columns = df.loc[:, df.dtypes == object]
    le = preprocessing.LabelEncoder()
    for i in nonnumeric_columns:
        df[i] = le.fit_transform(df[i])

    df.fillna(0.0, inplace=True)

    return df

def prepModels(models):

    model = LogisticRegression(penalty='l1',C=0.001)
    models.append(('LogisticRegression', model))

    model = SVC(gamma='auto', class_weight='balanced')
    models.append(('SVC',model))

    # n_estimators=120,300,500,800,1200
    # max_depth=5,8,15,25,30,None
    # min_samples_split=1,2,5,10,15,100
    # min_samples_leaf=1,2,5,10
    # max_features='log2','sqrt
    model = RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_split=2, min_samples_leaf=1,
                                   max_features='log2')
    models.append(('RandomForestClassifier', model))

    # feat_cols = learn.infer_real_valued_columns_from_input(train_df)
    # opti = tf.train.GradientDescentOptimizer(learning_rate=0.05)
    # model = learn.DNNClassifier(n_classes=2,feature_columns=feat_cols,optimizer=opti,hidden_units=[10, 20, 10])
    # models.append(('DNNClassifier', model))

    return models

def classification_model(model, data, predictors, outcome):
    kf = KFold(data.shape[0], n_folds=5)
    scores = []
    for train_rows, test_rows in kf:
        train_predictors = (data[predictors].iloc[train_rows, :])
        train_target = data[outcome].iloc[train_rows]

        model.fit(train_predictors, train_target)

        test_predictors = (data[predictors].iloc[test_rows, :])
        test_target = data[outcome].iloc[test_rows]

        # # Print accuracy
        # predictions = model.predict(test_predictors)
        # accuracy = accuracy_score(predictions, test_target)
        # print("Accuracy : %s" % "{0:.3%}".format(accuracy))

        score = model.score(test_predictors,test_target)
        scores.append(score)

    mean_score = np.mean(scores)
    print("Accuracy : {0:4.4f}".format(mean_score))
    return mean_score

if __name__ == "__main__":
    train_df = pd.read_csv('../data/bar_train.csv')
    train_df = prepData(train_df)

    test_df = pd.read_csv('../data/bar_test.csv')
    test_df = prepData(test_df)

    predictor_all_variables = list(train_df.columns.values)
    outcome_variable = predictor_all_variables[-1]
    predictor_variables = [col for col in predictor_all_variables if col != outcome_variable] # and col.startswith("bar")]

    models = []
    models = prepModels(models)
    max_score = -1
    best_model = None
    best_model_name = None
    for name, model in models:
        score = classification_model(model,train_df,predictor_variables,outcome_variable)
        if score > max_score:
            max_score = score
            best_model = model
            best_model_name = name

    print("Best Model {0} with Score {1:4.4f}".format(best_model_name, max_score))
    best_model.fit(train_df[predictor_variables], train_df[outcome_variable])

    score = best_model.score(test_df[predictor_variables], test_df[outcome_variable])
    print("Accuracy for test csv: {}".format(score))

    sample_test_df = train_df.sample(500)
    prediction_values = best_model.predict(sample_test_df[predictor_variables])
    actual_values = sample_test_df[outcome_variable].values
    print("Prediction for test csv: {}".format(prediction_values))
    print("Actual for test csv: {}".format(actual_values))
    target_names = ["SL", "TP"]
    cm = classification_report(actual_values, prediction_values, target_names=target_names)
    print(cm)
    # test_df[outcome_variable] = best_model.predict(test_df[predictor_variables])
    # test_df.to_csv('../data/test_predicted.csv',index=False)


