# Neural Network for Data Prediction

This Python script demonstrates how to create and train a neural network using the scikit-learn library to predict data values. The script reads training and test data from Excel files, normalizes the data, trains a neural network, and then predicts and visualizes the test data.

## Prerequisites

- Python
- pandas
- numpy
- matplotlib
- scikit-learn (`MLPRegressor` from `sklearn.neural_network`)

## Usage

1. Clone this repository or download the script.
2. Install the required Python packages using `pip install pandas numpy matplotlib scikit-learn`.
3. Modify the paths to your training and test data Excel files (`SourceFile` and `Test_Data`).
4. Customize the neural network parameters and training options as needed.
5. Run the script using `python script_name.py`, where `script_name.py` is the name of your script.

## Description

This script performs the following tasks:

- Reads training and test data from Excel files.
- Normalizes the training and test data to ensure consistent scaling.
- Creates a neural network using scikit-learn's `MLPRegressor`.
- Trains the neural network on the training data.
- Predicts values for the test data and de-normalizes the predictions.
- Visualizes the predicted and actual values using matplotlib.

You can customize the script by adjusting neural network parameters, data paths, or plotting settings to suit your specific use case.

## License

This project is licensed under the [MIT License](LICENSE).
