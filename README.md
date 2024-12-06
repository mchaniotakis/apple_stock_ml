# Apple Stock ML Project

A machine learning project for predicting extreme events in Apple stock prices using Random Forest and Temporal CNN models.

## Installation

### Prerequisites

- Python 3.10 or higher
- Poetry (Python package manager)

### Setup

1. [Install Poetry](https://python-poetry.org/docs/#installation) (if not already installed)

2. Clone the repository
```
git clone https://github.com/mchaniotakis/apple_stock_ml

cd apple_stock_ml

```

3. Install dependencies using Poetry
```
poetry lock
poetry install
```
## Project Structure

```
src/
├── data_processing.py # Data loading and preprocessing functions
├── random_forest.py # Random Forest model implementation
├── temporal_cnn.py # Temporal CNN model implementation
├── model_evaluation.py # Model evaluation and metrics calculation
└── improvement.py # Model improvements and enhancements

```

## Data
 - You dont have to specifically download any data, all models use data_processing to import the data requested.
 - For every model you can specify the `--ticker stock_name` option to use that stock for training. Every stock is cashed so that you dont have to download it twice. 
 - You can download multiple stocks using `SP500` for the S&P500 stocks, or if you would like the top X stocks, you can use this ticker : `SP500TOP10` which will download the top 10 stocks and use them for training.
 - Make sure to change the `--start-date` `--end-date` if you need to use different dates.
 - Use `--threshold` to specify the threshold for the extreme event point, currently at 2.0
 - `--model-output` specifies the output location of the model saved after its trained, currently at models/


## Evaluation
 - Once the models are trained, they would all be located at models/.. at that point feel free to run the evaluation script to compare their performance. Please take a look at `evaluation_results`
 - Run with ```python apple_stock_ml/src/model_validation.py --ticker AAPL --rf_idx_to_keep [-1]``` which makes sure that the RF model is only using the daily return feature.
 - The performance metrics for both models are stored in the `evaluation_results` directory, including:
        - Confusion Matrix
        - Accuracy
        - Precision, Recall, F1-Score
        - ROC curves

## Train models

### Random Forest Model

Run the Random Forest model with default settings:
```
python apple_stock_ml/src/random_forest.py --ticker AAPL
```
Optional flags:

``--use-smote``: Enable SMOTE for handling class imbalance

``-o``: Enable optimization mode for hyperparameter tuning


### Temporal CNN model
1. run the Temporal CNN model with the default settings:
```
python apple_stock_ml/src/temporal_cnn.py --ticker AAPL --learning-rate 0.1 --epochs 300 --patience 15 --batch-size 256
```
2. Optional flags:

``--pca``: Enable PCA dimensionality reduction

``--sequence-length``: Set custom sequence length (default: 10)

``--augs``: Enable data augmentation

``--use-smote``: Enable SMOTE for handling class imbalance

### Improved Model

1. Run the improved model with the default settings:
```
python apple_stock_ml/src/improvement.py --ticker AAPL --sequence-length 10 --learning-rate 0.01 --batch-size 256 --patience 15
```
2. Optional flags:

``--pca``: Enable PCA dimensionality reduction

``--sequence-length``: Set custom sequence length (default: 10)

``--augs``: Enable data augmentation

``--use-smote``: Enable SMOTE for handling class imbalance

## Model Validation
1.Run the model validation (we are using only the dialy return for RF):
```
python apple_stock_ml/src/model_validation.py --ticker AAPL --rf_idx_to_keep [-1]
```
The validation script will:
 - Load the trained models (from models/...pt)
 - Generate performance metrics
 - Create confusion matrices
 - Compare model performances

## Reports and Documentation

- Detailed analysis and results can be found in `report.pdf`
- Model architecture details and hyperparameters are documented in the respective source files
- Performance comparisons and improvement strategies are detailed in the report


### Notes
 - All models will save their outputs in the appropriate directories under the project folder
 - Results and metrics will be stored in the evaluation_results directory
 - Models can be run with different stock tickers by changing the --ticker parameter
 - Use the --help flag with any script to see all available options
  - For more details about the implementation and metrics, refer to the project documentation and the original assignment requirements.