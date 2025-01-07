import pandas as pd

import sklearn.datasets as datasets


def generate_data_circles(conf):
    """
    Generates artificial data for testing purposes.

    Args:
        conf (dict): Configuration dictionary for the data generation process.

    Returns:
        df: Artificial dataset (Pandas DataFrame).
    """

    X, y = datasets.make_circles(**conf)
    df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    df['class'] = y

    return df


def generate_data_moons(conf):
    """
    Generates artificial data for testing purposes.

    Args:
        conf (dict): Configuration dictionary for the data generation process.

    Returns:
        df: Artificial dataset (Pandas DataFrame).
    """
    X, y = datasets.make_moons(**conf)
    df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    df['class'] = y

    return df


def generate_data_blobs(conf):
    """
    Generates artificial data for testing purposes.

    Args:
        conf (dict): Configuration dictionary for the data generation process.

    Returns:
        df: Artificial dataset (Pandas DataFrame).
    """

    X, y = datasets.make_blobs(**conf)
    feature_columns = [f'feature_{i+1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_columns)
    df['class'] = y

    return df


def generate_data_classification(conf):
    """
    Generates artificial data for testing purposes.

    Args:
        conf (dict): Configuration dictionary for the data generation process.

    Returns:
        df: Artificial dataset (Pandas DataFrame).
    """

    X, y = datasets.make_classification(**conf)
    feature_columns = [f'feature_{i+1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_columns)
    df['class'] = y

    return df


def generate_gaussian_quantiles(conf):
    """
    Generates artificial data for testing purposes.

    Args:
        conf (dict): Configuration dictionary for the data generation process.

    Returns:
        df: Artificial dataset (Pandas DataFrame).
    """

    X, y = datasets.make_gaussian_quantiles(**conf)
    feature_columns = [f'feature_{i+1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_columns)
    df['class'] = y

    return df