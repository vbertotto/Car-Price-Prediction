# Car Price Prediction with XGBoost

This repository contains code for predicting car prices using machine learning techniques, specifically the XGBoost algorithm. The main focus is on preparing the data, encoding categorical variables, scaling features, and applying hyperparameter tuning to optimize the model.

## Dataset
The dataset used contains various features such as:
- **brand**: The brand of the car.
- **model**: The model of the car.
- **fuel_type**: Type of fuel (e.g., gasoline, diesel, etc.).
- **engine**: Information on the engine type.
- **transmission**: The type of transmission (manual/automatic).
- **ext_col** and **int_col**: Exterior and interior colors of the car.
- **accident**: Whether the car has been in an accident.
- **clean_title**: If the car has a clean title.
- **model_year**: Year the car was manufactured.
- **price**: The target variable, which is the price of the car.

## Data Preprocessing

1. **Handling Missing Values**: Missing values in both training and test datasets are filled with the placeholder `"unknown"`.
2. **Feature Engineering**: A new feature `car_age` is derived by subtracting the car's `model_year` from the current year (2024).
3. **Log Transformation**: The target variable `price` is transformed using the log transformation (`log1p`) to stabilize variance and normalize the distribution.
4. **Outlier Removal**: Outliers in the `price_log` column are identified and removed using the Interquartile Range (IQR) method.
5. **Encoding Categorical Variables**: Target encoding is applied to the categorical columns using the average target variable for each category.
6. **Scaling**: RobustScaler is used to scale the features, making the model more robust to outliers.

## Model Training

We use **XGBoost**, a powerful gradient boosting model, with hyperparameter tuning to improve performance. The hyperparameters are optimized using **RandomizedSearchCV** with 5-fold cross-validation.

### Hyperparameters Tuning
The model is optimized for:
- `n_estimators`: Number of trees.
- `max_depth`: Maximum depth of the trees.
- `learning_rate`: Step size shrinkage.
- `subsample`: Fraction of samples used per tree.
- `colsample_bytree`: Fraction of features used per tree.
- `reg_alpha` and `reg_lambda`: L1 and L2 regularization terms.

### Evaluation
The model's performance is evaluated using **Mean Absolute Error (MAE)** on the validation set.

## Predictions

After training, the model makes predictions on the test set, and the log-transformed predictions are reverted to the original scale using `expm1` (inverse of `log1p`). The predictions are saved to a CSV file for submission.

## Files

- **train.csv**: Training dataset.
- **test.csv**: Test dataset.
- **submission.csv**: Predicted car prices for the test dataset.

## How to Run the Code

1. Clone this repository.
2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the script to train the model and generate predictions:
    ```bash
    python car_price_prediction.py
    ```

The resulting predictions will be saved in a file named `submission.csv`.

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- xgboost
- category_encoders
- numpy

## Conclusion

This project demonstrates how to build a car price prediction model using XGBoost, incorporating data preprocessing techniques such as feature engineering, log transformation, encoding, and scaling. Hyperparameter tuning further improves the model's performance for better accuracy in predicting car prices.

Feel free to fork and modify the code!
