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

In this tutorial, we will perform tabular regression on the AutoMPG dataset using a runway pipeline. The goal is to load the tabular data, train a regression model, and then save it for future use.

> 📘 For quick execution, you can use the Jupyter Notebook provided below.
> If you download and run the Jupyter Notebook, a model named "auto-mpg-reg-model-sklearn" will be created and saved in Runway.
>
> **[auto mpg model notebook](https://drive.google.com/uc?export=download&id=1v2L3OeycGqgqcc8w2ost9SPX730sVcwg)**

![link pipeline](../../assets/auto_mpg_regression/link_pipeline.png)

## Runway

> We will use the AutoMPG dataset, which contains information about cars released in the late 1970s and early 1980s, including attributes such as the number of cylinders, displacement, horsepower, weight, and origin.
>
> You can download the AutoMPG dataset using the following link:  
> **[auto-mpg.csv](https://runway-tutorial.s3.ap-northeast-2.amazonaws.com/auto-mpg.csv)**

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
    !pip install sklearn pandas numpy
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

#### Model

##### Model Class

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

#### Model Training

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

### Upload Model

> 📘 You can find detailed instructions on how to save the model in the [Upload Model](https://docs.live.mrxrunway.ai/en/Guide/ml_development/dev_instances/%EB%AA%A8%EB%8D%B8_%EC%97%85%EB%A1%9C%EB%93%9C/).
1. Use the `save model` option from the Runway code snippet to save the model.
2. Create a sample input data for the generated code.
    ```python
    import runway

    runway.start_run()
    runway.log_metric("mse", mse)

    input_sample = X_train.sample(1)

    runway.log_model(model_name="auto-mpg-reg-model-sklearn", model=runway_regressor, input_samples={"predict": input_sample})
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

