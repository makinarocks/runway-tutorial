# Auto MPG Regression

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

In this tutorial, we will train and save a regression model using the AutoMPG dataset provided by UC Irvine, which contains various information about automobiles.

> 📘 For quick execution, you can use the Jupyter Notebook provided below.
> If you download and run the Jupyter Notebook, a model named "auto-mpg-reg-model-sklearn" will be created and saved in Runway.
>
> **[auto mpg model notebook](https://drive.google.com/uc?export=download&id=1v2L3OeycGqgqcc8w2ost9SPX730sVcwg)**

![link pipeline](../../assets/auto_mpg_regression/link_pipeline.png)

### Package Preparation

1. (Optional) Install the required packages for the tutorial.
    ```python
    !pip install sklearn pandas numpy
    ```

## Data
Load the CSV file included in the Runway tutorial folder to create and preprocess the dataset.

> 📘 The dataset used in this tutorial is the AutoMPG dataset, which contains information about automobiles released in the late 1970s and early 1980s.
> The AutoMPG dataset is located in the `./dataset` directory, and you can also download it from the link below if needed.
>  **[auto-mpg.csv](https://runway-tutorial.s3.ap-northeast-2.amazonaws.com/auto-mpg.csv)**

### Load Data

1. Check the path of the dataset file in the file explorer.
2. Assign the dataset file path to the RUNWAY_DATA_PATH parameter.
    
    ```python
    import os
    import pandas as pd

    RUNWAY_DATA_PATH = "/home/jovyan/workspace/examples/tutorial/auto_mpg_regression/dataset"

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
    ```

### Data Preprocessing

1. Remove any missing values in the dataset and separate the predictor and target columns.
    ```python
    # Drop NA data in dataset
    data_clean = df.dropna()

    # Select Predictor columns
    X = df[["cylinders", "displacement", "weight", "acceleration", "origin"]]

    # Select target column
    y = df["mpg"]
    ```

2. Split the dataset into training and testing sets.

    ```python
    from sklearn.model_selection import train_test_split

    #Split data into training and testing sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=2024)
    ```

## Model

### Model Class

1. Define a model class for training the regression model.

    ```python
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler


    class RunwayRegressor:
        def __init__(self):
            """Initialize."""
            self.preprocessing = StandardScaler()
            self.model = LinearRegression()

        def fit(self, X, y):
            """fit model."""
            X_scaled = self.preprocessing.fit_transform(X)
            self.model.fit(X_scaled, y)

        def predict(self, X):
            X_scaled = self.preprocessing.transform(X)
            pred = self.model.predict(X_scaled)
            pred_df = pd.DataFrame({"mpg_pred": pred})
            return pred_df
    ```

### Model Training

1. Use the declared model class and the training dataset to train the model, and log the information related to train.
    ```python
    from sklearn.metrics import mean_squared_error


    runway_regressor = RunwayRegressor()
    runway_regressor.fit(X_train, y_train)

    #Test model on held out test set
    valid_pred = runway_regressor.predict(X_valid)

    #Mean Squared error on the testing set
    mse = mean_squared_error(valid_pred, y_valid)

    #Print evaluate model score
    print("Mean Squared Error: {}".format(mse))
    ```

### Model Registration

Register the trained regression model in Runway so that it can be used for inference services.

1. Use the model registration code snippet in the Runway platform to register (`log_model`) the trained model and record related information.

    ``` python
    import mlflow
    import runway

    with mlflow.start_run():
        mlflow.log_metric("mse", mse)

        runway.log_model(
            model=runway_regressor,
            input_samples={"predict": X_train.sample(1)},
            model_name="auto-mpg-reg-model-sklearn",
        )
    ```

## Pipeline Configuration and Saving

> 📘 For specific guidance on creating a pipeline, refer to the [Upload Pipeline](https://docs.live.mrxrunway.ai/en/guide/core-features/dev-instances/create-a-pipeline/).

1.  Write and verify the pipeline in **Link** to ensure it runs smoothly.
2.  After verifying successful execution, click the **Upload pipeline** button in the Link pipeline panel.
3.  Click the **New Pipeline** button.
4.  Enter the name for the pipeline to be saved in Runway in the **Pipeline** field.
5.  The **Pipeline version** field will automatically select version 1.
6.  Click the **Upload** button.
7.  Once the upload is complete, the uploaded pipeline item will appear on the Pipeline page within the project.

