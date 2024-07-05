# Stroke prediction

This project is a machine learning model for predicting strokes. The project uses ensemble models to make predictions
and is implemented in Python. 

## Installing / Getting started

1. CD into the folder where requirements.txt can be found and run the following command in your terminal to install the required packages
   
```shell
pip install -r requirements.txt
```

2. Run the script itself

```shell
python main.py
```

## Data

The data used in this project is a healthcare dataset for stroke prediction. 
It includes various features such as gender, age, heart disease, work type, bmi and other.
The dataset is sourced from Kaggle and can be found [here](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

The data analysis for this project is conducted in a Jupyter notebook: [data_analysis.ipynb](./data_analysis.ipynb).
This notebook contains exploratory data analysis and visualizations to understand 
the data better.

## Model

The model is implemented in the [main.py](./main.py) file. In this file, you can find the code for loading the data,
preprocessing it, training the model and evaluating its performance.

Many different models have been used in this project, but the best performing model is the Voting Classifier with Random Forest classifier.
You can check out the feature importance of the Random Forest model with AdaBoost classifier, which is saved in the data folder.

## Author

Hristina Adamović SV32/2020


