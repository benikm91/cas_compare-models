"""
Run multiple models and compare them to # samples needed
"""
import pickle
import time
from pathlib import Path

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import numpy as np
import pandas as pd
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from scipy.stats import loguniform, uniform, randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
enable_halving_search_cv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, HalvingRandomSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from joblib import Memory
from house.subset_train_cv import SubsetTrainCV

from timeit import default_timer as timer
from datetime import timedelta

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)

enable_gpu = False  # Note: feed-forward-NN is slower with GPU here


if not enable_gpu:
    gpus = []
    tf.config.set_visible_devices([], 'GPU')
else:
    gpus = tf.config.list_physical_devices('GPU')
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

def fix_randomness(random_state = 42):
    """
    Fixiere den Zufall, dass runs von cifar10_main.py immer das gleiche Ergebnis ergeben.
    """
    tf.random.set_seed(random_state)
    np.random.seed(random_state)


def load_dataset():
    data = pd.read_csv(Path.cwd() / 'data/houses_train.csv', index_col=0)
    X = data.drop(columns='object_type_name')
    y = LabelEncoder().fit_transform(data[['object_type_name']])
    return X, y


class SampleSubsetInTraining(BaseEstimator, TransformerMixin):

    def __init__(self, num_samples_percentage):
        self.num_samples_percentage = num_samples_percentage
        self.n_used_train_samples = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if y is not None:
            self.n_used_train_samples = int(X.shape[0] * self.num_samples_percentage)
            mask = np.full(X.shape[0], False)
            mask[:self.n_used_train_samples] = True
            np.random.shuffle(mask)
            return X[mask], y[mask]
        else:
            return X

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X, y)


# @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def create_nn(meta, dropout_rate=0.0):
    nn = tf.keras.Sequential()
    nn.add(tf.keras.Input(shape=meta["X_shape_"][1:]))
    nn.add(tf.keras.layers.Dense(64, activation='relu'))
    nn.add(tf.keras.layers.Dropout(dropout_rate))
    nn.add(tf.keras.layers.Dense(32, activation='relu'))
    nn.add(tf.keras.layers.Dropout(dropout_rate))
    nn.add(tf.keras.layers.Dense(meta["n_classes_"], activation='softmax'))
    return nn


if __name__ == "__main__":

    # config
    random_state = 42
    fix_randomness(random_state)
    n_jobs = -1
    tf.compat.v1.disable_eager_execution()

    X, y = load_dataset()

    print(X.shape, y.shape)

    X = X.drop(columns=['municipality_name'])

    preprocess = make_pipeline(
        make_column_transformer(
            (OneHotEncoder(handle_unknown='ignore'), ['zipcode']),
            remainder=StandardScaler(),
            sparse_threshold=0
        ),
    )

    cache_memory = None
    # cache_memory = mkdtemp()
    # location = './cache'
    # cache_memory = Memory(location, verbose=0)

    models = {
        'LR': make_pipeline(preprocess, LogisticRegression(max_iter=10000), memory=cache_memory),
        'KNN': make_pipeline(preprocess, KNeighborsClassifier(), memory=cache_memory),
        'DT': make_pipeline(preprocess, DecisionTreeClassifier(), memory=cache_memory),
        'RF': make_pipeline(preprocess, RandomForestClassifier(), memory=cache_memory),
        'NN': make_pipeline(preprocess, KerasClassifier(
            model=create_nn,
            epochs=50,
            loss="sparse_categorical_crossentropy",
            batch_size=128,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]
        ), memory=cache_memory),
        'PCA+LR': make_pipeline(preprocess, PCA(n_components=0.9), LogisticRegression(max_iter=10000), memory=cache_memory),
        'PCA+SVM(linear)': make_pipeline(preprocess, PCA(n_components=0.9), LinearSVC(max_iter=10000), memory=cache_memory),
        'PCA+SVM(rbf)': make_pipeline(preprocess, PCA(n_components=0.9), SVC(kernel='rbf'), memory=cache_memory),
        'PCA+KNN': make_pipeline(preprocess, PCA(n_components=0.9), KNeighborsClassifier(), memory=cache_memory),
        'PCA+DT': make_pipeline(preprocess, PCA(n_components=0.9), DecisionTreeClassifier(), memory=cache_memory),
        'PCA+RF': make_pipeline(preprocess, PCA(n_components=0.9), RandomForestClassifier(), memory=cache_memory),
        'PCA+NN': make_pipeline(preprocess, PCA(n_components=0.9), KerasClassifier(
            model=create_nn,
            epochs=50,
            loss="sparse_categorical_crossentropy",
            batch_size=128,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]
        ), memory=cache_memory)
    }

    hps = dict()
    hps['LR'] = dict(
        logisticregression__penalty=['l2'],
        logisticregression__C=loguniform(1e-5, 100),
    )
    hps['PCA'] = dict(
        pca__n_components=loguniform(0.5, 1)
    )
    hps['PCA+LR'] = { **hps['PCA'], **hps['LR'] }
    hps['SVM(linear)'] = dict(
        linearsvc__C=loguniform(1e-5, 100),
    )
    hps['PCA+SVM(linear)'] = { **hps['PCA'], **hps['SVM(linear)'] }
    hps['SVM(rbf)'] = dict(
        svc__gamma=loguniform(1e-5, 100),
        svc__C=loguniform(1e-5, 100),
    )
    hps['PCA+SVM(rbf)'] = { **hps['PCA'], **hps['SVM(rbf)'] }
    hps['KNN'] = dict(
        kneighborsclassifier__n_neighbors=[3, 5, 7, 9, 11],
        kneighborsclassifier__weights=['uniform', 'distance']
    )
    hps['PCA+KNN'] = { **hps['PCA'], **hps['KNN'] }
    hps['DT'] = dict(
        # TODO check parameters
        # decisiontreeclassifier__criterion=['gini', 'entropy'],
        decisiontreeclassifier__max_depth = [2, 4, 6, 8, 10, 12, None]
    )
    hps['PCA+DT'] = { **hps['PCA'], **hps['DT'] }
    hps['RF'] = dict(
        # TODO check parameters
        randomforestclassifier__n_estimators=randint(10, 1000),
        randomforestclassifier__max_depth=[2, 4, 6, 8, 10, 12, None],
    )
    hps['PCA+RF'] = { **hps['PCA'], **hps['RF'] }
    hps['NN'] = dict(
        # kerasclassifier__model__dropout_rate=uniform(0.0, 1.0)
    )
    hps['PCA+NN'] = { **hps['PCA'], **hps['NN'] }

    total_start = timer()
    start_datetime_str = time.strftime("%Y%m%d-%H%M%S")

    for subset_percentage in np.logspace(np.log2(0.005), np.log2(1), num=10, base=2)[9:]:
        n_samples = int(X.shape[0] * subset_percentage)
        result = dict()
        iter_start = timer()
        print(f"Start {n_samples}")
        for model_name, model in models.items():
            hyperparameter_search_space = hps[model_name]
            model_print = f"{model_name} for {n_samples} samples ({subset_percentage * 100:.2f}%)"
            print(model_print, end="")
            if len(hyperparameter_search_space) > 0:
                # Do nested-cross-validation for hyperparameters, if declared
                nested_model = HalvingRandomSearchCV(estimator=model, param_distributions=hyperparameter_search_space, cv=3, random_state=random_state, n_jobs=n_jobs)
            else:
                # No hyperparameters to fit
                nested_model = model
            subset_cv = SubsetTrainCV(KFold(n_splits=3, shuffle=True, random_state=random_state), random_start_state=random_state, subset_percentage=subset_percentage)
            sample_result = cross_validate(nested_model, X, y, cv=subset_cv, return_train_score=True, return_estimator=True, scoring='accuracy', error_score='raise', n_jobs=n_jobs)
            result[model_name] = {
                'test_score': sample_result['test_score'],
                'train_score': sample_result['train_score'],
                'fit_time': sample_result['fit_time'],  # Note: fit_time is with hyper-parameter-search if specified
                'score_time': sample_result['score_time'],
                'n_samples_percentage': subset_percentage,
                'n_used_train_samples': subset_cv.train_sample_sizes,
                'hyper_parameters': list(map(lambda x: getattr(x, 'best_params_', None), sample_result['estimator'])),
                'hyper_parameter_search_space': hyperparameter_search_space,
            }
            print(f"\r{model_print}: {np.mean(result[model_name]['test_score']) * 100:.2f}%")

        if cache_memory is not None:
            cache_memory.clear()

        with open(f'out/house_{n_samples}_{start_datetime_str}.pickle', 'wb') as file:
            pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)

        iter_end = timer()
        print(f"{n_samples} took {timedelta(seconds=iter_end-iter_start)}")

    total_end = timer()
    print(f"Total took {timedelta(seconds=total_end-total_start)}")
