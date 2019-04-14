import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.getLogger().setLevel(logging.INFO)

def from_one_hot(data):
    new_data = np.zeros(len(data))
    for i, item in enumerate(data):
        if np.array_equal(item, np.array([0., 1.])) :
            new_data[i] = int(1)
        else:
            new_data[i] = int(0)
    return np.array(new_data)

def prepData(df, features,label):
    # Let's fill in the "holes" with the means on numerical attributes
    df.fillna(df.mean(), inplace=True)

    # convert all our categorical variables into numeric by encoding the categories.
    nonnumeric_columns = df.loc[:, df.dtypes == object]
    le = preprocessing.LabelEncoder()
    for i in nonnumeric_columns:
        df[i] = le.fit_transform(df[i])

    df[features] = df[features].apply(lambda x: (x - x.min()) / (x.max() - x.min())) # normalization

    unique_label_values = list(df[label].unique())
    df[label] = df[label].map(lambda x: np.eye(len(unique_label_values))[unique_label_values.index(x)])

    # Let's fill in the "holes" with the means on numerical attributes
    df.fillna(0, inplace=True)
    return df

if __name__ == "__main__":
    train_df = pd.read_csv('../data/bar_train.csv')
    test_df = pd.read_csv('../data/bar_test.csv')

    predictor_all_variables = list(train_df.columns.values)
    outcome_variable = predictor_all_variables[-1]
    predictor_variables = [col for col in predictor_all_variables if col != outcome_variable]# and col.startswith("b")]
    print("Features considered {} \nOutcome {}".format(predictor_variables,outcome_variable))

    train_df = prepData(train_df, predictor_variables,outcome_variable)
    test_df = prepData(test_df, predictor_variables,outcome_variable)

    #  Number of features: all those bar1, bar2...High low, date, EXCCEPT the last 'result'
    #  Number of outputs = 2  possibilities TP and SL, stored as One-hot, [0,1], or [1,0], so '2'
    n_features = len(predictor_variables)
    n_outputs = 2

    # define our x placeholder as the variable to feed our x_train ie train_df[predictor_variables] data into
    # 'None' means the placeholder can be fed as many examples as you want to give it.
    x = tf.placeholder(tf.float32, [None, n_features])
    # define y_, which will be used to feed y_train into
    y_ = tf.placeholder(tf.float32, [None, n_outputs])

    # define the weights W and bias b. These two values are the grunt workers of the classifier
    # they will be the only values we will need to calculate our prediction after the classifier is trained.

    # Either u can initialize with 0 or random numbers, better w random
    # w = tf.Variable(tf.zeros([n_features, n_outputs]))
    # b = tf.Variable(tf.zeros([n_outputs]))
    w = tf.Variable(tf.random_normal([n_features, n_outputs], mean=0.0, stddev=1.0, dtype=tf.float32))
    b = tf.Variable(tf.random_normal([n_outputs], mean=0.0, stddev=1.0, dtype=tf.float32))

    # define y, which is our classifier function, also known as multinomial logistic regression.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized)
    logits = tf.matmul(x, w) + b
    y = tf.nn.softmax(logits) # Predicted label

    # taking the log of all our predictions y (whose values range from 0 to 1) and
    # element wise multiplying by the examples true value y_. If the log function for each value is close to zero,
    # it will make the value a large negative number (i.e., -np.log(0.01) = 4.6), and if it is close to 1,
    # it will make the value a small negative number (i.e., -np.log(0.99) = 0.1).
    # penalizing the classifier with a very large number if the prediction is confidently incorrect and
    # a very small number if the prediction is confidendently correct.

    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y + 1e-5), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))

    # Train using tf.train.GradientDescentOptimizer
    # train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

    # Add accuracy checking nodes
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # # This is for saving all our work
    # saver = tf.train.Saver([w, b])

    print("...training....")

    # Run the training
    shuffled = train_df.sample(frac=1)
    training_rows = int(0.2 * len(shuffled))
    cutoff_point = len(shuffled) - training_rows
    trainingSet = shuffled[0:cutoff_point]
    testSet = shuffled[cutoff_point:]

    for i in range(10000):
        train = trainingSet.sample(n=100)
        x_train = [x for x in train[predictor_variables].values]
        y_train = [x for x in train[outcome_variable].as_matrix()]
        sess.run(train_step, feed_dict={x: x_train,y_: y_train})
        if i % 100 == 0:
            #print(sess.run(y, feed_dict={x: x_feed}))
            test = testSet.sample(n=5)
            x_test = [x for x in test[predictor_variables].values]
            y_test = [x for x in test[outcome_variable].as_matrix()]
            print('Training Step:' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})) +
                  '  Loss = ' + str(sess.run(cross_entropy, {x: x_train, y_: y_train})))

            # iter_accuracy = sess.run(accuracy, feed_dict={x:x_test,y_: y_test })
            # print("Accuracy {} iteration: {}".format(i,iter_accuracy))

    # Test trained model
    accuracy_value = sess.run(accuracy, feed_dict={x: [x for x in test_df[predictor_variables].values],
                                  y_: [x for x in test_df[outcome_variable].as_matrix()]})
    print("Accuracy on Test data set: {}".format(accuracy_value))

    # y has probailities, use the argmax function to return the position of the highest value
    predictor = tf.argmax(y, axis=1)
    # prediction_values = prediction.eval(feed_dict={x: [x for x in test_df[predictor_variables].values]}, session=sess)
    sample_entries = test_df#.sample(n=10, replace=True)
    x_sample = [x for x in sample_entries[predictor_variables].values]
    prediction_values = sess.run(predictor, feed_dict={x:x_sample })
    prediction_values = [np.eye(n_outputs)[k] for k in prediction_values]
    actual_values = list(sample_entries[outcome_variable].values)
    # print('Prediction {}\nActual {}'.format(prediction_values,actual_values))

    prediction_values = from_one_hot(prediction_values)
    actual_values = from_one_hot(actual_values)

    print('Prediction {}\nActual {}'.format(prediction_values,actual_values))
    cm = confusion_matrix(actual_values, prediction_values)
    print(cm)
