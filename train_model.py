import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import processing, filtering

# ANSI escape codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RED = "\033[31m"


# Model Name
# MODEL_NAME = "linear_regressor_v02"
MODEL_NAME = "render_forest_v02"

# Step 1: Load the dataset
print(f"{CYAN}{BOLD}Loading dataset from csv file{RESET}")
data = pd.read_csv("./data/dataset.csv", encoding="ISO-8859-1")

# Step 2: Filtering the data
print(f"{YELLOW}{BOLD}Filtering data{RESET}")
data = filtering.filter_data(data)

# Step 3: Processing the data to convert text rating into their respective numerical values
print(f"{GREEN}{BOLD}Processing data{RESET}")
data = processing.map_rating(data)

# Step 4: Calculate an average rating as the overall rating
print(f"{MAGENTA}{BOLD}Calculating overall rating{RESET}")
# data["Overall Rating"] = data.iloc[:, 3:17].mean(axis=1)

# Learn those columns that doesn't have an int value
numeric_columns = data.select_dtypes(include=[int]).columns
# Create another column Overall Rating in the data
data["Overall Rating"] = data[numeric_columns].apply(
    lambda x: x.mean(skipna=True), axis=1
)

# Step 5: Creating Features and Target Variable
print(f"{CYAN}{BOLD}Creating Features and Target Variable{RESET}")
X = data.drop(
    columns=[
        "Hotel Name",
        "Location",
        "Country",
        "Overall Rating",
        "Q14. You will recommend this hotel facility to your friends.",
    ]
)  # Features
y = data["Overall Rating"]  # Target (calculated overall rating)

# Step 6: Train-Test Split
print(f"{YELLOW}{BOLD}Splitting data into training and testing sets{RESET}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Train the model
print(f"{GREEN}{BOLD}Training the model{RESET}")
# model = LinearRegression()
model = RandomForestRegressor(n_estimators=20000, random_state=100)
model.fit(X_train, y_train)

# Step 8: Storing the model for future use, and saving the feature names used for training
print(f"{MAGENTA}{BOLD}Saving the model and feature names{RESET}")
joblib.dump(model, "./models/" + MODEL_NAME)
joblib.dump(
    X_train.columns, "./models/" + MODEL_NAME + "_features.pkl"
)  # Save the feature columns used during training

# Step 9: Model Evaluation
# print(f"{RED}{BOLD}Evaluating the model{RESET}")
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f"{RED}{BOLD}Mean Squared Error: {mse}{RESET}")
