# Wind Power Prediction with XGBoost

<h4 align="center">
    <p>
        <a href="README.md">한국어</a> |
        <b>English</b>
    <p>
</h4>

<h3 align="center">
    <p>The MLOps platform to Let your AI run</p>
</h3>

## Introduction

This tutorial trains and saves an XGBoost model to predict wind power generation using sensor log data collected from wind turbines. ([Wind Power Forecasting](https://www.kaggle.com/datasets/theforcecoder/wind-power-forecasting), provided by Kaggle)  
To enable retraining with the developed model training code, a pipeline is configured and saved.

> 📘 For quick execution, you can utilize the following Jupyter Notebook.  
> If you download and execute the Jupyter Notebook below, a model named ""my-xgboost-regressor" will be created and saved in Runway.
>
> **[wind_power_prediction_with_xgboost](https://drive.google.com/uc?export=download&id=16ruQV9Q4sJuxvN7IxrPjTqHSv5gNducc)**

![link pipeline](../../assets/wind_power_prediction_with_xgboost/link_pipeline.png)

### Install package

1. Install the required packages for the tutorial.
    ```python
    !pip install xgboost
    ```

## Data
Load the CSV file included in the **Runway tutorial folder** to create and preprocess the dataset.

> 📘 This tutorial uses the data included in [Wind Power Forecasting](https://www.kaggle.com/datasets/theforcecoder/wind-power-forecasting), provided by Kaggle.  
> The dataset for wind power generation is located in the `./dataset` directory, and can be downloaded from the link below if needed. 
> **[Wind power forecasting dataset](https://drive.google.com/uc?export=download&id=16iE44jF7J6rCa01EGcUP1wuMrKJUdN7J)**

### Load data

1. Check the path of the dataset file in the file explorer.
2. Assign the dataset file path to the RUNWAY_DATA_PATH parameter.
 
    ```python
    import os
    import pandas as pd

    RUNWAY_DATA_PATH = "/home/jovyan/workspace/examples/tutorial/wind_power_prediction_with_xgboost/dataset"
    dfs = []
    for dirname, _, filenames in os.walk(RUNWAY_DATA_PATH):
        for filename in filenames:
            if filename.endswith(".csv"):
                d = pd.read_csv(os.path.join(dirname, filename))
            elif filename.endswith(".parquet"):
                d = pd.read_parquet(os.path.join(dirname, filename))
            else:
                raise ValueError("Not valid file type")
            dfs += [d]

    df = pd.concat(dfs)
    df.columns = df.columns.map(lambda x: x.lower())
    ```

### Preprocess data

1. Split data to X, y.
    ```python
    X_columns = [
        "ambienttemperatue",
        "bearingshafttemperature",
        "blade1pitchangle",
        "blade2pitchangle",
        "blade3pitchangle",
        "controlboxtemperature",
        "gearboxbearingtemperature",
        "gearboxoiltemperature",
        "generatorrpm",
        "generatorwinding1temperature",
        "generatorwinding2temperature",
        "hubtemperature",
        "mainboxtemperature",
        "nacelleposition",
        "reactivepower",
        "rotorrpm",
        "turbinestatus",
        "winddirection",
        "windspeed",
    ]
    y_column = "activepower"


    X_df = df[X_columns]
    y_df = df[y_column]
    ```

2. Split data to train and valid.
    ```python
    from sklearn.model_selection import train_test_split

    ## Split data into training and testing sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_df, y_df, test_size=0.2, random_state=2024)
    ```

## Model
### Train model

> 📘 You can find guidance on registering Link parameters in the **[Set Pipeline Parameters](https://docs.live.mrxrunway.ai/en/guide/core-features/dev-instances/set-pipeline-parameters/)**.

1. To specify the number of components to use in XGBRegressor, you register the following items with the Link parameter.

    - `LEARNING_RATE`: 0.1
    - `MAX_DEPTH`: 5
    - `ALPHA`: 10
    - `N_ESTIMATORS`: 10

    ![link parameter](../../assets/wind_power_prediction_with_xgboost/link_parameter.png)

2. Load the model using the `XGBRegressor` module of XGBoost.
    ```python
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error


    params = {
       "objective": "reg:squarederror",
       "learning_rate": LEARNING_RATE,
       "max_depth": MAX_DEPTH,
       "alpha": ALPHA,
       "n_estimators": N_ESTIMATORS,
       }

    regr = xgb.XGBRegressor(
       objective=params["objective"],
       learning_rate=params["learning_rate"],
       max_depth=params["max_depth"],
       alpha=params["alpha"],
       n_estimators=params["n_estimators"],
    )
    ```

3. Use the loaded model and the training dataset to perform model training and evaluate it with the evaluation data.
    ```python
    regr.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

    y_pred = regr.predict(X_valid)
    mae = mean_absolute_error(y_pred, y_valid)
    mse = mean_squared_error(y_pred, y_valid)
    ```


### Model wrapping class

1. Write the `RunwayModel` class to be used for API serving.
    ```python
    import mlflow
    import pandas as pd


    class RunwayModel(mlflow.pyfunc.PythonModel):
        def __init__(self, xgb_regressor):
            self._regr = xgb_regressor

        def predict(self, context, X):
            pred = self._regr.predict(X)
            activepower_pred = {"activepower": pred}
            pred_df = pd.DataFrame(activepower_pred)
            return pred_df
    ```


### Model Registration

Register the trained regression model in Runway so that it can be used for inference services.

1. Use the Runway platform’s model registration code snippet to register (`log_model`) the trained model and record the related information.
    ```python
    import mlflow
    import runway

    with mlflow.start_run():
        runway_model = RunwayModel(regr)
        input_df = df[X_columns]
        input_sample = input_df.sample(1)

        mlflow.log_params(params)
        mlflow.log_metric("valid_mae", mae)
        mlflow.log_metric("valid_mse", mse)

        runway.log_model(
            model=runway_model,
            model_name="my-xgboost-regressor",
            input_samples={"predict": input_sample},
        )
    ```

## Pipeline Configuration and Saving

> 📘 For specific guidance on creating a pipeline, refer to the [Create a pipeline](https://docs.live.mrxrunway.ai/en/guide/core-features/dev-instances/create-a-pipeline/).

1.  Write and verify the pipeline in **Link** to ensure it runs smoothly.
2.  After verifying successful execution, click the **Upload pipeline** button in the Link pipeline panel.
3.  Click the **New Pipeline** button.
4.  Enter the name for the pipeline to be saved in Runway in the **Pipeline** field.
5.  The **Pipeline version** field will automatically select version 1.
6.  Click the **Upload** button.
7.  Once the upload is complete, the uploaded pipeline item will appear on the Pipeline page within the project.
