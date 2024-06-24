# utils.py

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

def calculate_imbalance_ratio(train_data, target_column):
    """
    Calculate the imbalance ratio of the target classes in the training data.

    Parameters:
    - train_data: pandas DataFrame, the training dataset
    - target_column: str, the name of the target column

    Returns:
    - imbalance_ratio: float, the imbalance ratio of the target classes
    - majority_class: str, the label of the majority class
    - majority_class_count: int, the count of instances in the majority class
    - minority_class: str, the label of the minority class
    - minority_class_count: int, the count of instances in the minority class
    """
    # Initialize a dictionary to store the count of each class
    class_counts = {}

    # Loop over each unique class in the target column
    for c in train_data[target_column].unique():
        # Count the number of instances for the current class
        class_count = (train_data[target_column] == c).sum()
        # Store the class count in the dictionary
        class_counts[c] = class_count

    # Find the majority class and its count
    majority_class = max(class_counts, key=class_counts.get)
    majority_class_count = class_counts[majority_class]

    # Find the minority class and its count
    minority_class = min(class_counts, key=class_counts.get)
    minority_class_count = class_counts[minority_class]

    # Calculate the imbalance ratio
    imbalance_ratio = majority_class_count / minority_class_count

    return imbalance_ratio, majority_class, majority_class_count, minority_class, minority_class_count


def filter_noisy_samples(generated_data, real_training_data, X_test, y_test, target_column, model_params, error_rate_threshold):
    filtered_data = []

    # Copy the input dataframes to avoid modifying them directly
    generated_data_copy = generated_data.copy()
    real_training_data_copy = real_training_data.copy()

    # Train a decision tree classifier on the real training data        
    tree_model = DecisionTreeClassifier(**model_params)
    tree_model.fit(real_training_data_copy.drop(target_column, axis=1), real_training_data_copy[target_column])

    # Make predictions on the test set using the decision tree classifier
    y_pred = tree_model.predict(X_test)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate error rate per class
    error_rate_per_class = 1 - np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    print('Initial Error rate: ', error_rate_per_class)

    for index, generated_instance in generated_data_copy.iterrows():
        # Append the generated instance to the real training data
        augmented_data = pd.concat([real_training_data_copy, generated_instance.to_frame().T], axis=0)
        augmented_labels = augmented_data[target_column]
        augmented_features = augmented_data.drop(target_column, axis=1)

        # Retrain the decision tree classifier on the augmented data
        tree_model_updated = DecisionTreeClassifier(**model_params)
        tree_model_updated.fit(augmented_features, augmented_labels)

        # Make predictions on the test set using the retrained model
        y_pred_updated = tree_model_updated.predict(X_test)

        # Calculate updated confusion matrix and error rate per class
        conf_matrix_updated = confusion_matrix(y_test, y_pred_updated)
        error_rate_per_class_updated = 1 - np.diag(conf_matrix_updated) / np.sum(conf_matrix_updated, axis=1)
        print('Sample ', index)
        # Check if the error rate decreases for the class corresponding to the generated instance's label
        if (error_rate_per_class_updated[1] <= error_rate_per_class[1]) and (error_rate_per_class_updated[0] < error_rate_threshold):
            filtered_data.append(generated_instance)
            # Update the error rate per class
            error_rate_per_class = error_rate_per_class_updated
            # Update the real training data for the next iteration
            real_training_data_copy = augmented_data
            print('Error rate: ', error_rate_per_class)
            print('Sample Added\n')

    return pd.DataFrame(filtered_data)
