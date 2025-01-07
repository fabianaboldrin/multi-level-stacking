import numpy as np
import pandas as pd
import time
import warnings
from multi_level_stacking.constants import MODEL_DICTIONARY
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score

# Suppress the specific warning
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _multi_level_stacking(X_train, y_train, P, A, M):
    """
    Implements a stacked learning algorithm.

    Args:
        D0_x: Input features (NumPy array).
        D0_y: Input labels (NumPy array).
        P: Number of stacking levels.
        A: A list of L learning algorithms (functions that take X, y and return a trained model).
        M: Meta-learning algorithm (function that takes X, y and returns a trained model).

    Returns:
        A trained meta-classifier (hP).
    """


    # Reset index on D0_x and D0_y
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True
                            )
    n_samples = X_train.shape[0]
    Dp_x = X_train
    Dp_y = y_train
    L = len(A)
    trained_model_dict = {}

    for p in range(P):
        # Stage 1: Induction of level-p classifiers
        hp = []
        for j in range(L):
            model = A[j](Dp_x, Dp_y)  # Train base classifier
            hp.append(model)
        
        trained_model_dict[p] = hp  # Store trained models for each level

        # Stage 2: Create training set for level (p + 1)
        Dp1_x = []
        # Dp1_y = []

        for i in range(n_samples):
            # Create Dpi by removing the i-th sample.  Handle it more efficiently using boolean indexing.
            mask = np.ones(n_samples, dtype=bool)
            mask[i] = False
            Dpi_x = Dp_x[mask]
            Dpi_y = Dp_y[mask]

            hpi_predictions = []
            for j in range(L):
                model = A[j](Dpi_x, Dpi_y)
                prediction = model.predict_proba([Dp_x.iloc[i]])[:,1]  # Predict the probability of class 1 on the held-out sample
                hpi_predictions.append(round(prediction[0],3))

            Dp1_x.append(hpi_predictions)  # Store predictions for the next level
        
        Dp_x = pd.DataFrame(np.array(Dp1_x))  # Convert to pandas dataframe for next level.
        Dp_y = Dp_y  # Keep the same labels for the next level

    # Stage 3: Induce the level-P classifier
    hP = M(Dp_x, Dp_y)  # Train meta-classifier

    trained_model_dict[P] = [hP]  # Store the final model

    return trained_model_dict  # Return both hP and the trained models for each level


def _multi_level_stacking_prediction(x, trained_models):
    """
    Predicts the output using a stack of trained models.

    Args:
        x: The input data point.  Should be a NumPy array.
        trained_models: A dictionary where keys are level numbers (0, 1, 2,...P-1,P)
                        and values are lists/arrays of trained models for that level. 
                        Each model in the list should have a 'predict' method. 
                        The dictionary should also contain the final model as entry 'P'.

    Returns:
        The predicted output (y_hat).
    """


    x_current = np.array([x]) # Make sure it's a numpy array. Could be a single value or a vector
    num_levels = max(trained_models.keys()) # Assumes levels are numbered from 0,1,2...P 
    
    for level in range(num_levels):
        models_at_level = trained_models[level]
        x_next = []

        for model in models_at_level:
            prediction = model.predict_proba(x_current)[:,1]  # Predict the probability of class 1 on the held-out sample
            x_next.append(prediction)
        
        x_current = np.array(x_next).reshape(1,-1)  # Reshape for next level input

    # Final prediction using level P model (which has just one entry):
    y_hat = trained_models[num_levels][0].predict(x_current)
    y_hat = np.squeeze(y_hat)
    
    return y_hat


def _run_k_fold_validation(D0_x, D0_y, p, A, M, validation_params):

    # Initialize KFold
    kf = KFold(n_splits=validation_params['n_splits'], 
               shuffle=validation_params['shuffle'],
               random_state=validation_params['random_state'])

    # Initialize lists to store performance metrics
    errors = []
    auc = []

    # Iterate over each fold
    for train_index, test_index in kf.split(D0_x):

        X_train, X_test = D0_x.iloc[train_index], D0_x.iloc[test_index]
        y_train, y_test = D0_y.iloc[train_index], D0_y.iloc[test_index]

        # Run multi-level stacking on the training data
        trained_model_dict = _multi_level_stacking(X_train, y_train, p, A, M)

        # Efficiently predict on the test data
        y_pred = np.array([_multi_level_stacking_prediction(X_test.iloc[i], trained_model_dict) for i in range(X_test.shape[0])])

        # Calculate performance metrics
        errors.append(1 - accuracy_score(y_test, y_pred))
        auc.append(roc_auc_score(y_test, y_pred, average='weighted'))
        

    # Print average performance metrics
    print("Average Error:", np.mean(errors)) # 1- accuracy = error rate
    print("Average AUC:", np.mean(auc)) # AUC = Area Under ROC Curve
    
    dict_metrics = {
        'error': np.mean(errors),
        'auc': np.mean(auc)
    }

    return dict_metrics




def run_multi_level_stacking(D0, 
                             dataset_name,
                             modeling_params,
                             validation_params,
                             meta_algorithm):
    """
    Wrapper function to run multi-level stacking.

    Args:
        D0: Input dataframe.
        dataset_name: Name of the dataset.
        P: Number of stacking levels.
        L: Number of classifiers per level.
        A: A list of L learning algorithms (functions that take X, y and return a trained model).
        M: Meta-learning algorithm (function that takes X, y and returns a trained model.

    Returns:
        A trained meta-classifier (hP).
    """
    list_of_models = modeling_params['list_of_algorithms']

    # Extract algorithms from the model dictionary
    A = [MODEL_DICTIONARY[model_name] for model_name in list_of_models]
    M = MODEL_DICTIONARY[meta_algorithm]
    P = modeling_params['list_of_levels']

    # Extract input features and labels
    D0_x = D0.drop(columns=['class'])
    D0_y = D0['class']

    # Measure the time taken for k-fold validation for each level
    start_time = time.time()

    for p in P:
        dict_metrics_p = _run_k_fold_validation(D0_x, D0_y, p, A, M, validation_params)

        # Append metrics to CSV file
        df_metrics = pd.DataFrame([{
            'dataset_name': dataset_name,
            'meta_algorithm': meta_algorithm,
            'level': p,
            'error': dict_metrics_p['error'],
            'auc': dict_metrics_p['auc'],
        }])
        
        # Append to CSV file
        df_metrics.to_csv('/home/fabiana_boldrin_bunge_com/multi-level-stacking/data/08_reporting/multi_level_stacking_metrics.csv',
                          mode='a', 
                          header=not pd.io.common.file_exists('multi_level_stacking_metrics.csv'), 
                          index=False)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken for multi-level stacking: {elapsed_time:.2f} seconds")
    
    return True