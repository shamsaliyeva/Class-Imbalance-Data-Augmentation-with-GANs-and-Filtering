# scripts/evaluation.py

from sklearn.metrics import accuracy_score, f1_score, recall_score, cohen_kappa_score, confusion_matrix
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

def calculate_metrics(y_test, y_pred):
    """
    Calculate various classification metrics.
    
    Parameters:
    - y_test: array-like, true labels
    - y_pred: array-like, predicted labels
    
    Returns:
    - metrics: dict, dictionary containing the calculated metrics
    """
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate F1-score with macro averaging
    f1_macro = f1_score(y_test, y_pred, average='macro')

    # Calculate recall with macro averaging
    recall_macro = recall_score(y_test, y_pred, average='macro')

    # Calculate geometric mean
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    geometric_mean = np.sqrt(sensitivity * specificity)

    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(y_test, y_pred)

    # Store metrics in a dictionary
    metrics = {
        "Accuracy": accuracy,
        "F1-score (macro)": f1_macro,
        "Recall (macro)": recall_macro,
        "Geometric Mean": geometric_mean,
        "Cohen's Kappa": kappa
    }
    
    # Convert the dictionary into a DataFrame
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])
    
    return metrics_df



def calculate_error_rate_per_class(y_true, y_pred):
    """
    Calculate the error rate per class.
    
    Parameters:
    - y_true: true labels
    - y_pred: predicted labels
    
    Returns:
    - error_rate_per_class: array-like, error rate for each class
    """
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Calculate error rate per class
    error_rate_per_class = 1 - np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    
    return error_rate_per_class


def f1_score_per_class(y_true, y_pred):
    """
    Calculate F1 score per class.

    Parameters:
    - y_true: true labels
    - y_pred: predicted labels

    Returns:
    - f1_scores: array-like, F1 score per class
    """
    # Calculate F1 score per class
    f1_scores = f1_score(y_true, y_pred, average=None)
    
    
    return f1_scores


def ks_test(synth_data, minority_train_data, continuous_columns, target_column):
    """
    Perform Kolmogorov-Smirnov test for each continuous column.
    
    Parameters:
    - synth_data: pandas DataFrame, the synthetic data
    - minority_train_data: pandas DataFrame, the minority class training data
    - continuous_columns: list, names of continuous columns in the dataset
    - target_column: str, name of the target column
    
    Returns:
    - ks_results_df: pandas DataFrame, the results of the KS test for each continuous column
    - average_ks_score: float, the average KS test score
    """
    ks_results_df = pd.DataFrame(columns=['Column', 'KS Statistic', 'P-value', 'Result'])

    for column in continuous_columns:
        ks_statistic, p_value = stats.ks_2samp(synth_data[column], minority_train_data[column])
        result = "False" if p_value < 0.05 else "True"
        ks_results_df = pd.concat([ks_results_df, pd.DataFrame({'Column': [column],
                                                                'KS Statistic': [ks_statistic],
                                                                'P-value': [p_value],
                                                                'Result': [result]})],
                                  ignore_index=True)

    filtered_df = ks_results_df[ks_results_df['Column'] != target_column]
    average_ks_score = (1 - filtered_df['KS Statistic']).mean()
    return ks_results_df, average_ks_score


def chi2_test(synth_data, minority_train_data, discrete_columns, target_column):
    """
    Perform Chi-Square test for each discrete column.
    
    Parameters:
    - synth_data: pandas DataFrame, the synthetic data
    - minority_train_data: pandas DataFrame, the minority class training data
    - discrete_columns: list, names of discrete columns in the dataset
    - target_column: str, name of the target column
    
    Returns:
    - chi2_test_results_df: pandas DataFrame, the results of the Chi-Square test for each discrete column
    - average_p_value: float, the average p-value from the Chi-Square tests
    """
    chi2_test_results = []

    for column in discrete_columns:
        contingency_table = pd.crosstab(minority_train_data[column], synth_data[column])
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
        is_significant = p_value > 0.05
        chi2_test_results.append({'Column': column, 'Chi-Square Statistic': chi2_stat, 'P-value': p_value, 'P>0.05': is_significant})

    chi2_test_results_df = pd.DataFrame(chi2_test_results)
    average_p_value = chi2_test_results_df.loc[chi2_test_results_df['Column'] != target_column, 'P-value'].mean()
    return chi2_test_results_df, average_p_value

