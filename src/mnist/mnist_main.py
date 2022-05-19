"""
Run multiple models and compare them to # samples needed
"""
import pickle

from sklearn.experimental import enable_halving_search_cv

enable_halving_search_cv

import numpy as np
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from keras import Sequential, Input
from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Conv2D, Reshape
from scipy.stats import loguniform, randint

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, KFold, HalvingRandomSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from house.subset_train_cv import SubsetTrainCV

from timeit import default_timer as timer
from datetime import timedelta
import time

enable_gpu = True

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


def load_mnist_dataset():
    """
    Lade die MNIST Bilder und Labels
    :return: Bilder als numpy matrix mit shape (70000, 784), Labels als numpy vector mit shape (70000,)
    """
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    y = y.astype(np.int32)  # Cast string like '1' to integer like 1.
    return X, y


def create_nn(meta):
    nn = Sequential()
    nn.add(Input(shape=meta["X_shape_"][1:]))
    nn.add(Dense(64, activation='relu'))
    nn.add(Dense(32, activation='relu'))
    nn.add(Dense(meta["n_classes_"], activation='softmax'))
    return nn


def create_cnn(meta, image_shape):
    cnn = Sequential()
    cnn.add(Input(shape=meta["X_shape_"][1:]))
    cnn.add(Reshape(image_shape))  # Reshape 1D-pixel-vector to 3D image
    cnn.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    cnn.add(GlobalAveragePooling2D())
    cnn.add(Dense(meta["n_classes_"], activation='softmax'))
    return cnn


def create_big_cnn(meta, image_shape):
    cnn = Sequential()
    cnn.add(Input(shape=meta["X_shape_"][1:]))
    cnn.add(Reshape(image_shape))  # Reshape 1D-pixel-vector to 3D image
    cnn.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    cnn.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    cnn.add(GlobalAveragePooling2D())
    cnn.add(Dense(meta["n_classes_"], activation='softmax'))
    print(cnn.summary())
    return cnn


if __name__ == "__main__":
    # config
    random_state = 42
    fix_randomness(random_state)
    n_jobs = -1
    tf.compat.v1.disable_eager_execution()

    X, y = load_mnist_dataset()

    cache_memory = None

    preprocess = StandardScaler()

    models = {
        'LR': make_pipeline(preprocess, LogisticRegression(max_iter=10000), memory=cache_memory),
        'KNN': make_pipeline(preprocess, KNeighborsClassifier(), memory=cache_memory),
        'DT': make_pipeline(preprocess, DecisionTreeClassifier(), memory=cache_memory),
        'RF': make_pipeline(preprocess, RandomForestClassifier(), memory=cache_memory),
        'CNN': make_pipeline(preprocess, KerasClassifier(
            model=create_cnn,
            image_shape=(28, 28, 1),
            epochs=50,
            loss="sparse_categorical_crossentropy",
            batch_size=128,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]
        ), memory=cache_memory),
        'CNN_BIG': make_pipeline(preprocess, KerasClassifier(
            model=create_big_cnn,
            image_shape=(28, 28, 1),
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
    hps['CNN'] = dict()
    hps['CNN_BIG'] = dict()
    hps['PCA+CNN'] = { **hps['PCA'], **hps['CNN'] }

    total_start = timer()
    start_datetime_str = time.strftime("%Y%m%d-%H%M%S")

    subset_model = {
        'CNN_BIG': models['CNN_BIG']
    }

    for subset_percentage in np.logspace(np.log2(0.005), np.log2(1), num=10, base=2):
        n_samples = int(X.shape[0] * subset_percentage)
        result = dict()
        iter_start = timer()
        print(f"Start {n_samples}")
        for model_name, model in subset_model.items():
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

        with open(f'out/mnist_{n_samples}_{start_datetime_str}.pickle', 'wb') as file:
            pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)

        iter_end = timer()
        print(f"{n_samples} took {timedelta(seconds=iter_end-iter_start)}")

    total_end = timer()
    print(f"Total took {timedelta(seconds=total_end-total_start)}")
