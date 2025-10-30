# Robotarm Anomaly Detection

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

This tutorial trains and saves an anomaly detection model using data that simulates the movements of a four-axis robotic arm.
The trained model code is then organized into a pipeline to facilitate retraining and reuse.

> 📘 For quick execution, you can utilize the following Jupyter Notebook.  
> If you download and execute the Jupyter Notebook below, a model named "pca-model" will be created and saved in Runway.
>
> **[robotarm anomaly detection notebook](https://drive.google.com/uc?export=download&id=10d2Hc4lYx0WOuEvLOkqNTQMpDezbzVzw)**

![link pipeline](../../assets/robotarm_anomaly_detection/link_pipeline.png)


### Package Preparation

1. (Optional) Install the required packages for the tutorial.
    ```python
    !pip install pandas scikit-learn
    ```

## Data
Load the CSV file included in the Runway tutorial folder to create and preprocess the dataset.

> 📘 In this tutorial, we use sample data that simulates the movements of a 4-axis robotic arm.
The robotic arm dataset is located in the `./dataset` directory, and can be downloaded from the link below if needed.
> **[robotarm-train.csv](https://drive.google.com/uc?export=download&id=1Ks8SUVBQawiKW0q0zQT1sc9um618cdEE)**

### Load Data

1. Check the path of the dataset file in the file explorer.
2. Assign the dataset file path to the RUNWAY_DATA_PATH parameter.
    ```python
    import os
    import pandas as pd

    RUNWAY_DATA_PATH = "/home/jovyan/workspace/examples/tutorial/robotarm_anomaly_detection/dataset"
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

1. Set the index in the dataset and remove the "id" values, then use only a total of 1000 data points.

    ```python
    proc_df = df.set_index("datetime").drop(columns=["id"]).tail(1000)
    ```

2. Split the dataset into training and testing sets.

    ```python
    from sklearn.model_selection import train_test_split

    train, valid = train_test_split(proc_df, test_size=0.2, random_state=2024)
    ```

## Model

### Model Class

1. Write a model class for model training.

    ```python
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler


    class PCADetector:
        def __init__(self, n_components):
            self._use_columns = ...
            self._scaler = StandardScaler()
            self._pca = PCA(n_components=n_components)

        def fit(self, X):
            self._use_columns = X.columns
            X_scaled = self._scaler.fit_transform(X)
            self._pca.fit(X_scaled)

        def predict(self, X):
            X = X[self._use_columns]
            X_scaled = self._scaler.transform(X)
            recon = self._recon(X_scaled)
            recon_err = ((X_scaled - recon) ** 2).mean(1)
            recon_err_df = pd.DataFrame(recon_err, columns=["anomaly_score"], index=X.index)
            return recon_err_df

        def _recon(self, X):
            z = self._pca.transform(X)
            recon = self._pca.inverse_transform(z)
            return recon

        def reconstruct(self, X):
            X_scaled = self._scaler.transform(X)
            recon_scaled = self._recon(X_scaled)
            recon = self._scaler.inverse_transform(recon_scaled)
            recon_df = pd.DataFrame(recon, index=X.index, columns=X.columns)
            return recon_df
    ```

### Model Training

> 📘 You can find guidance on registering Link parameters in the **[Set Pipeline Parameter](https://docs.live.mrxrunway.ai/en/guide/core-features/dev-instances/set-pipeline-parameters/)**.

1. To specify the number of components to use in PCA, register 2 in the N_COMPONENTS Link parameter.

    - `N_COMPONENTS`: 2

    ![link parameter](../../assets/robotarm_anomaly_detection/link_parameter.png)

2. Use the declared model class and the training dataset to train the model, and log the information related to train.

    ```python
    parameters = {"n_components": N_COMPONENTS}
    detector = PCADetector(n_components=parameters["n_components"])
    detector.fit(train)

    train_pred = detector.predict(train)
    valid_pred = detector.predict(valid)

    mean_train_recon_err = train_pred.mean()
    mean_valid_recon_err = valid_pred.mean()
    ```


### Model Registration

Register the trained model in Runway so that it can be used for inference services.

1. Use the model registration code snippet in the Runway platform to register (`log_model`) the trained model and record related information.
    ```python
    import mlflow
    import runway

    with mlflow.start_run():
        mlflow.log_params(parameters)

        mlflow.log_metric("mean_train_recon_err", mean_train_recon_err)
        mlflow.log_metric("mean_valid_recon_err", mean_valid_recon_err)

        runway.log_model(
            model=detector,
            input_samples={"predict": proc_df.sample(1)},
            model_name="pca-model",
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
