import pandas as pd
import joblib


def predict_overall_rating(model, input_data, feature_columns=None):
    """
    Predicts the overall rating using the trained model.

    Args:
        model (sklearn model): The trained model.
        input_data (pd.DataFrame or dict): A row of input data (can be a dictionary or DataFrame).
        feature_columns (list): The columns that were used for training the model.
                                This ensures the new data matches the original training data.

    Returns:
        float: The predicted overall rating.
    """
    # If input_data is a dictionary, convert it to a DataFrame
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        input_data = pd.DataFrame([input_data])

    # Align the new data's columns with the trained model's features
    if feature_columns is not None:
        input_data = input_data[
            feature_columns
        ]  # Select only the features used for training

    # Predict the rating
    prediction = model.predict(input_data)
    return round(prediction[0], 2)


def load_model(model_file):
    """
    Loads a stored model using joblib.

    Args:
        model_file (str): The path to the stored model.

    Returns:
        model: The loaded machine learning model.
        list: The feature columns used for training the model.
    """
    model = joblib.load(model_file)

    # Assuming that the model was trained using specific features,
    # load these features from the model object (or store them separately during training)
    try:
        feature_columns = (model.feature_names_in_)  # Some sklearn models store feature names
    except AttributeError:
        feature_columns = None  # In case feature names aren't available in the model

    return model, feature_columns
