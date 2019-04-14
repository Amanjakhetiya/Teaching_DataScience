import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.getLogger().setLevel(logging.INFO)

training_file = "../data/iris_train_label.csv"
testing_file = "../data/iris_test_label.csv"

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename = training_file,
    target_dtype = np.int,
    features_dtype = np.float32,
    target_column = 4
)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=testing_file,
    target_dtype=np.int,
    features_dtype=np.float32,
    target_column = 4)


feature_columns = [tf.contrib.layers.real_valued_column("",dimension=4)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10,20,10],
                                            n_classes=3,
                                            model_dir="/logs")

classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)['accuracy']
print("Accuracy: {0:f}".format(accuracy_score))


new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable = True))
print('Predictions:{}'.format(str(y)))