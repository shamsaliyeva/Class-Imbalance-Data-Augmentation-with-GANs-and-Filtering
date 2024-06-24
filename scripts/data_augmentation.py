#scripts/data_augmentation.py
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
import pandas as pd
import os
import joblib

def generate_synthetic_samples_ctgan(train_data, minority_class, majority_class_count, minority_class_count, ctgan_args, train_args, continuous_columns, discrete_columns, target_column):
    """
    Generate synthetic samples for the minority class to balance the dataset.

    Parameters:
    - train_data: pandas DataFrame, the original training data
    - minority_class: str, label of the minority class
    - majority_class_count: int, count of instances in the majority class
    - minority_class_count: int, count of instances in the minority class
    - ctgan_args: ModelParameters, parameters for CTGAN model
    - train_args: TrainParameters, training parameters
    - continuous_columns: list, names of continuous columns in the dataset
    - discrete_columns: list, names of discrete columns in the dataset
    - target_column: str, name of the target column
    - model_name: str, optional, name of the trained CTGAN model

    Returns:
    - augmented_train_data: pandas DataFrame, the augmented training data with synthetic samples
    - synth_data: pandas DataFrame, the synthetic samples generated for the minority class
    """
    num_minority_samples_needed = majority_class_count - minority_class_count

    # Create a copy of the original training data
    augmented_train_data = train_data.copy()

    # If target_column is provided, remove it from the data
    if target_column:
        augmented_train_data.drop(columns=[target_column], inplace=True)

    # Synthesize new samples for the minority class if needed
    if num_minority_samples_needed > 0:
        # Subset the minority class data
        minority_train_data = train_data[train_data[target_column] == minority_class].copy()

        # Train the synthesizer model
        synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
        synth.fit(data=minority_train_data, train_arguments=train_args, num_cols=continuous_columns, cat_cols=discrete_columns)

        # Generate synthetic samples to balance the classes
        synth_data = synth.sample(num_minority_samples_needed)

        # Append the synthetic data to the augmented training data
        augmented_train_data = pd.concat([augmented_train_data, synth_data], ignore_index=True)

    return augmented_train_data, synth_data



def generate_synthetic_samples_wgangp(train_data, minority_class, majority_class_count, minority_class_count, wgan_args, train_args, continuous_columns, discrete_columns, target_column, n_critic):
    """
    Generate synthetic samples for the minority class to balance the dataset using GAN.

    Parameters:
    - train_data: pandas DataFrame, the original training data
    - minority_class: str, label of the minority class
    - majority_class_count: int, count of instances in the majority class
    - minority_class_count: int, count of instances in the minority class
    - wgan_args: ModelParameters, parameters for WGAN-GP model
    - train_args: TrainParameters, training parameters
    - continuous_columns: list, names of continuous columns in the dataset
    - discrete_columns: list, names of discrete columns in the dataset
    - target_column: str, name of the target column
    - n_critic: int, number of discriminator updates per generator update

    Returns:
    - augmented_train_data: pandas DataFrame, the augmented training data with synthetic samples
    - synth_data: pandas DataFrame, the synthetic samples generated for the minority class
    """
    num_minority_samples_needed = majority_class_count - minority_class_count

    # Create a copy of the original training data
    augmented_train_data = train_data.copy()

    # If target_column is provided, remove it from the data
    if target_column:
        augmented_train_data.drop(columns=[target_column], inplace=True)

    # Synthesize new samples for the minority class if needed
    if num_minority_samples_needed > 0:
        # Subset the minority class data
        minority_train_data = train_data[train_data[target_column] == minority_class].copy()

        # Train the generator model
        synth = RegularSynthesizer(modelname='wgangp', model_parameters=wgan_args, n_critic=n_critic)
        synth.fit(minority_train_data, train_args, continuous_columns, discrete_columns)

        # Generate synthetic samples to balance the classes
        synth_data = synth.sample(int(num_minority_samples_needed))

        # If the number of generated samples exceeds the required number, randomly select a subset
        if len(synth_data) > num_minority_samples_needed:
            synth_data = synth_data.sample(n=num_minority_samples_needed, replace=False, random_state=42)

        # Append the synthetic data to the augmented training data
        augmented_train_data = pd.concat([augmented_train_data, synth_data], ignore_index=True)

    return augmented_train_data, synth_data

