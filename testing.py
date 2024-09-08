import pandas as pd
import processing, filtering, predict, review, output, analytics


def testing(num_samples, test_features, original_data, to_html=False, html_filename="results.html"):
    """
    Tests the trained model using the specified number of samples from the test set.

    Args:
        num_samples (int): Number of samples to test from the test set.
        test_features (pd.DataFrame): DataFrame containing the feature columns for the test set.
        original_data (pd.DataFrame): The original data DataFrame used for extracting full sample data.
        to_html (bool): If True, save the results to an HTML file. Otherwise, print to the console.
        html_filename (str): The name of the file to save the results.
    """

    # Load the model and feature columns from file
    model, feature_columns = predict.load_model("linerrg")

    results = []

    for i in range(num_samples):
        # Get the index of the sample data
        sample_index = test_features.index[i]  # Index of the row in test_features

        # Extract the full row from the original data DataFrame
        full_sample_data = original_data.loc[sample_index]

        # Extract features from the test_features DataFrame
        sample_features = test_features.iloc[i]

        # Predict the overall rating using the stored feature columns
        predicted_rating = predict.predict_overall_rating(
            model, sample_features, feature_columns
        )

        # calcualting the Accuracy
        accuracy = analytics.accuracy(test_features, predicted_rating)

        if to_html:
            # Generate a review
            long_review = review.long_review(full_sample_data.to_dict(),True)
        else:
            # Generate a review
            long_review = review.long_review(full_sample_data.to_dict(),False)

        # Collect the results in a list of dictionaries
        results.append(
            {
                "Index": i + 1,
                "Hotel Name": full_sample_data.get("Hotel Name", "N/A").title(),
                "Predicted Rating": predicted_rating,
                "Review": long_review,
                "Accuracy": accuracy[0],
                "Original Rating": accuracy[1],
            }
        )

    # Save or print the results
    output.save_or_print_results(results, to_html=to_html, html_filename=html_filename)


if __name__ == "__main__":
    print("\n\033[1mTesting the model\033[0m\n")

    # Load the testing data
    data = pd.read_csv("testing.csv", encoding="ISO-8859-1")

    # Load the trained model
    model, feature_columns = predict.load_model("linerrg")

    # Step 1: Filter the data
    data = filtering.filter_data(data)

    # Step 2: Map text ratings to numerical values
    data = processing.map_rating(data)

    # Call the testing function with appropriate parameters
    # testing(num_samples=len(data), test_features=data, original_data=data, to_html=True)
    testing(num_samples=10, test_features=data, original_data=data)
    # testing(num_samples=len(data), test_features=data, original_data=data)
    # testing(num_samples=12, test_features=data, original_data=data, to_html=True)
