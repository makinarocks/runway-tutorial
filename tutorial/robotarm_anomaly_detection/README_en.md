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

We use the Link included in Runway to load a table-formatted dataset and train a regression model, which will then be saved. We also set up and save a pipeline to reuse the written model training code for future retraining.

> 📘 For quick execution, you can utilize the following Jupyter Notebook.  
> If you download and execute the Jupyter Notebook below, a model named "pca-model" will be created and saved in Runway.
>
> **[robotarm anomaly detection notebook](https://drive.google.com/uc?export=download&id=10d2Hc4lYx0WOuEvLOkqNTQMpDezbzVzw)**

![link pipeline](../../assets/robotarm_anomaly_detection/link_pipeline.png)

## Runway

> 📘 This tutorial creates a model for anomaly detection using data simulated to mimic a 4-axis robot arm.
>
> You can download the robot arm dataset by clicking the link below.
> **[robotarm-train.csv](https://drive.google.com/uc?export=download&id=1Ks8SUVBQawiKW0q0zQT1sc9um618cdEE)**

### Create a dataset

> 📘 For detailed information on dataset creation, please refer to the [official documentation](https://docs.live.mrxrunway.ai/en/Guide/ml_development/datasets/dataset-runway/).

1. Navigate to the dataset page from the Runway project menu.
2. Access the dataset creation menu in the dataset menu.
    - Click the `+` button at the top of the left dataset list.
    - Click the `Create` button on the initial screen.
3. In the dialog, enter the name of the dataset to create and click the `Create` button.

### Creating Dataset Version

1.  Click the `Create version` button in the `Versions` section.
2.  Select `Local file` in the dialog.
3.  Enter the name and description of the dataset to be saved.
4.  Select the file to be created as a dataset using the file explorer or Drag&Drop.
5.  Click `Create`.

## Link

### Package Preparation

1. (Optional) Install the required packages for the tutorial.
    ```python
    !pip install pandas scikit-learn
    ```

### Data

#### Load Data

> 📘 You can find detailed instructions on how to load the dataset in the [Import Dataset](https://docs.live.mrxrunway.ai/en/Guide/ml_development/dev_instances/%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%84%B8%ED%8A%B8_%EA%B0%80%EC%A0%B8%EC%98%A4%EA%B8%B0/).

1. Click the **Add Runway Snippet** button at the top of the notebook cell.
2. Select **Import Dataset**.
3. Choose the version of the dataset you want to use and click **Save**.
4. Upon clicking the button, a snippet will be generated in the notebook cell allowing you to browse the files within the selected dataset. Additionally, a dataset parameter with the dataset path as its value will be added.
5. Utilize the name of the registered dataset parameter in the notebook cell where you want to load the dataset.
    ```python
    import os
    import pandas as pd

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

#### Data Preprocessing

1. Set the index in the dataset and remove the "id" values, then use only a total of 1000 data points.

    ```python
    proc_df = raw_df.set_index("datetime").drop(columns=["id"]).tail(1000)
    ```

2. Split the dataset into training and testing sets.

    ```python
    from sklearn.model_selection import train_test_split

    train, valid = train_test_split(proc_df, test_size=0.2, random_state=2024)
    ```

### Model

#### Model Class

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

> 📘 You can find guidance on registering Link parameters in the **[Set Pipeline Parameter](https://docs.live.mrxrunway.ai/en/Guide/ml_development/dev_instances/%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8_%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0_%EC%84%A4%EC%A0%95/)**.

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


### Upload Model

> 📘 You can find detailed instructions on how to save the model in the [Upload Model](https://docs.live.mrxrunway.ai/en/Guide/ml_development/dev_instances/%EB%AA%A8%EB%8D%B8_%EC%97%85%EB%A1%9C%EB%93%9C/).

1. Create a sample input data from the training dataset.
2. Use the `save model` option from the Runway code snippet to save the model. And also log the information that are related to the model.
    ```python
    import runway


    # start run
    runway.start_run()

    # log param
    runway.log_parameters(parameters)

    # log metric
    runway.log_metric("mean_train_recon_err", mean_train_recon_err)
    runway.log_metric("mean_valid_recon_err", mean_valid_recon_err)

    # log model
    input_sample = proc_df.sample(1)
    runway.log_model(model_name="pca-model", model=detector, input_samples={"predict": input_sample})

    # stop run
    runway.stop_run()
    ```

## Pipeline Configuration and Saving

> 📘 For specific guidance on creating a pipeline, refer to the [Upload Pipeline](https://docs.live.mrxrunway.ai/en/Guide/ml_development/dev_instances/%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8_%EC%97%85%EB%A1%9C%EB%93%9C/).

1.  Write and verify the pipeline in **Link** to ensure it runs smoothly.
2.  After verifying successful execution, click the **Upload pipeline** button in the Link pipeline panel.
3.  Click the **New Pipeline** button.
4.  Enter the name for the pipeline to be saved in Runway in the **Pipeline** field.
5.  The **Pipeline version** field will automatically select version 1.
6.  Click the **Upload** button.
7.  Once the upload is complete, the uploaded pipeline item will appear on the Pipeline page within the project.
