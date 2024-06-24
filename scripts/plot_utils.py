# plot_utils.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_class_distribution(df, column, title=None, colors=['#77A1D3', '#A8D5BA']):
    """
    Plot the distribution of classes in a specified column.

    Parameters:
    - df: pandas DataFrame, the dataset
    - column: str, the column name for which the distribution needs to be plotted
    - title: str, optional, title of the plot
    - colors: list of str, colors to be used in the plot

    Returns:
    - None
    """
    class_distribution = df[column].value_counts()

    # Plot the class distribution
    plt.figure(figsize=(5, 5))
    bars = class_distribution.plot(kind='bar', color=colors)
    plt.title(title if title else f'{column} Class Distribution')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=0)

    # Add total numbers on top of each bar
    for i, count in enumerate(class_distribution):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot the confusion matrix as a heatmap.

    Parameters:
    - y_true: true labels
    - y_pred: predicted labels
    - labels: list of str, optional, list of class labels

    Returns:
    - None
    """
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(4, 3))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

