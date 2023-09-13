# Wind Power Prediction with XGBoost

<h4 align="center">
    <p>
        <b>한국어</b> |
        <a href="README_en.md">English</a>
    <p>
</h4>

<h3 align="center">
    <p>The MLOps platform to Let your AI run</p>
</h3>

## Introduction

Runway에 포함된 Link를 사용하여 XGBoost 모델을 학습하고 저장합니다.  
작성한 모델 학습 코드를 재학습에 활용하기 위해 파이프라인을 구성하고 저장합니다.

> 📘 빠른 실행을 위해 아래의 주피터 노트북을 활용할 수 있습니다.  
> 아래의 주피터 노트북을 다운로드 받아 실행할 경우, "my-xgboost-regressor" 이름의 모델이 생성되어 Runway에 저장됩니다.
>
> **[wind_power_prediction_with_xgboost](https://drive.google.com/uc?export=download&id=16ruQV9Q4sJuxvN7IxrPjTqHSv5gNducc)**

![link pipeline](../../assets/wind_power_prediction_with_xgboost/link_pipeline.png)

## Runway

### 데이터셋 생성

> 📘 이 튜토리얼은 Kaggle 에서 제공하는 [Wind Power Forecasting](https://www.kaggle.com/datasets/theforcecoder/wind-power-forecasting)입니다. 해당 데이터셋을 이용해 발전량 예측을 진행할 수 있습니다.
>
> Wind power forecasting 데이터셋은 아래 항목을 클릭하여 다운로드할 수 있습니다.  
> **[Wind power forecasting dataset](https://drive.google.com/uc?export=download&id=16iE44jF7J6rCa01EGcUP1wuMrKJUdN7J)**

1. Runway 프로젝트 메뉴에서 데이터셋 페이지로 이동합니다.
2. 데이터셋 페이지에서 신규 데이터셋을 생성합니다.
3. 데이터셋 페이지의 우측 상단 `Create Dataset`을 클릭합니다.
4. Tabular Data 영역의 Local file을 클릭합니다.
5. 저장하는 데이터셋의 이름과 설명을 입력합니다.
6. 데이터셋으로 생성할 파일을 파일 탐색기로 선택하거나, Drag&Drop으로 입력합니다.
7. `Create`를 클릭합니다.

## Link

### 패키지 설치

1. 튜토리얼에서 사용할 패키지를 설치합니다.

```python
!pip install xgboost
```

### 데이터

#### 데이터 불러오기

> 📘 데이터 세트 불러오는 방법에 대한 구체적인 가이드는 **[데이터 세트 가져오기](https://docs.mrxrunway.ai/docs/데이터-세트-가져오기)** 가이드 에서 확인할 수 있습니다.

1. Runway 코드 스니펫 메뉴의 **import dataset**을 이용해 프로젝트에 등록되어 있는 데이터셋 목록을 불러옵니다.
2. 생성한 데이터셋을 선택하고 variable 이름을 적습니다.
3. 코드를 생성하고 Link 컴포넌트로 등록합니다.

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

#### 데이터 전처리

1. 데이터를 X, y 로 나눕니다.

    ```python
    X_columns = [
       "activepower",
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

2. 학습 데이터와 평가 데이터를 나눕니다.

    ```python
    from sklearn.model_selection import train_test_split

    ## Split data into training and testing sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_df, y_df, test_size=0.2)
    ```

### 모델 학습

> 📘 Link 파라미터 등록 가이드는 **[파이프라인 파라미터 설정](https://dash.readme.com/project/makinarocks-runway/docs/파이프라인-파라미터-설정)** 문서에서 확인할 수 있습니다.

1. XGBRegressor 에서 사용할 컴포넌트의 개수를 지정하기 위해서 Link 파라미터로 다음 항목들을 등록합니다.

    - `LEARNING_RATE`: 0.1
    - `MAX_DEPTH`: 5
    - `ALPHA`: 10
    - `N_ESTIMATORS`: 10

    ![link parameter](../../assets/wind_power_prediction_with_xgboost/link_parameter.png)

2. XGBoost 의 `XGBRegressor` 모듈을 이용해 모델을 불러옵니다.

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

3. 불러온 모델과 학습용 데이터셋을 활용하여, 모델 학습을 수행하고 평가 데이터로 평가합니다.

    ```python
    regr.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

    y_pred = regr.predict(X_valid)
    mae = mean_absolute_error(y_pred, y_valid)
    mse = mean_squared_error(y_pred, y_valid)
    ```

### 모델 업로드

#### 모델 랩핑 클래스

1. API 서빙에 이용할 수 있도록 `RunwayModel` 클래스를 작성합니다.

    ```python
    import pandas as pd

    class RunwayModel:
        def __init__(self, xgb_regressor):
            self._regr = xgb_regressor

        def predict(self, X):
            return pd.DataFrame(
                {
                    "activepower": self._regr.predict(X),
                }
            )
    ```

2. 학습된 `regr` 을 `RunwayModel` 로 랩핑합니다.

    ```python
    runway_model = RunwayModel(regr)
    ```

#### 모델 업로드

> 📘 모델 업로드 방법에 대한 구체적인 가이드는 **[모델 업로드](https://docs.mrxrunway.ai/docs/%EB%AA%A8%EB%8D%B8-%EC%97%85%EB%A1%9C%EB%93%9C)** 문서에서 확인할 수 있습니다.

1. 모델 학습에 사용한 학습 데이터의 샘플을 생성합니다.

    ```python
    input_sample = X_df.sample(1)
    input_sample
    ```

2. Runway code snippet 의 save model을 사용해 모델을 저장하는 코드를 생성합니다. 또한 저장된 모델의 추가적인 정보인 사용된 파라미터, 평가 지표를 저장합니다.

    ```python
    import runway

    runway.start_run()
    runway.log_parameters(params)
    runway.log_metric("valid_mae", mae)
    runway.log_metric("valid_mse", mse)

    runway.log_model(model_name="my-xgboost-regressor", model=runway_model, input_samples={"predict": input_sample})

    ```

## 파이프라인 구성 및 저장

> 📘 파이프라인 생성 방법에 대한 구체적인 가이드는 **[파이프라인 생성](https://docs.mrxrunway.ai/docs/파이프라인-생성)** 문서에서 확인할 수 있습니다.

1. 파이프라인으로 구성할 코드 셀을 선택하여 컴포넌트로 설정합니다.
2. 파이프라인으로 구성이 완료되면, 전체 파이프라인을 실행하여 정상 동작 여부를 확인합니다.
3. 파이프라인의 정상 동작 확인 후, 파이프라인을 Runway에 저장합니다.
    1. 좌측 패널 영역의 Upload Pipeline을 클릭합니다.
    2. Pipeline 저장 옵션을 선택합니다.
        1. 신규 저장의 경우, New Pipeline을 선택합니다.
        2. 기존 파이프라인의 업데이트일 경우, Version Update를 선택합니다.
    3. 파이프라인 저장을 위한 값을 입력 후, Save를 클릭합니다.
4. Runway 프로젝트 메뉴에서 Pipeline 페이지로 이동합니다.
5. 저장한 파이프라인의 이름을 클릭하면 파이프라인 상세 페이지로 진입합니다.
