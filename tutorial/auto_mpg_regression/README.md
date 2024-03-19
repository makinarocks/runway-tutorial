# Auto MPG Regression

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

Runway에 포함된 Link를 사용하여 테이블 형식 데이터 세트를 로드하고 회귀 모델을 학습하여 저장합니다. 작성한 모델 학습 코드를 재학습에 활용하기 위해 파이프라인을 구성하고 저장합니다.

> 📘 빠른 실행을 위해 아래의 주피터 노트북을 활용할 수 있습니다.  
> 아래의 주피터 노트북을 다운로드 받아 실행할 경우, "auto-mpg-reg-model-sklearn" 이름의 모델이 생성되어 Runway에 저장됩니다.
>
> **[auto mpg model notebook](https://drive.google.com/uc?export=download&id=1v2L3OeycGqgqcc8w2ost9SPX730sVcwg)**

![link pipeline](../../assets/auto_mpg_regression/link_pipeline.png)

## Runway

> 📘 이 튜토리얼은 UC Irvine에서 제공하는 1970년대 후반과 1980년대 초반에 출시된 자동차의 정보가 포함된 AutoMPG 데이터 세트를 사용합니다.  
> 해당 데이터 세트에는 개별 자동차의 실린더 수, 배기량, 마력, 공차 중량, 제조국 등의 특성이 포함되어있습니다.
>
> AutoMPG 데이터셋은 아래 항목을 클릭하여 다운로드할 수 있습니다.  
> **[auto-mpg.csv](https://runway-tutorial.s3.ap-northeast-2.amazonaws.com/auto-mpg.csv)**

### 데이터 세트 생성하기

> 📘 데이터셋 생성에 관한 자세한 내용은 [공식 문서](https://docs.live.mrxrunway.ai/Guide/ml_development/datasets/dataset-runway/)를 참고하세요.

1. Runway 프로젝트 메뉴에서 데이터셋 페이지로 이동합니다.
2. 데이터 세트 메뉴에서 데이터 세트 생성 메뉴에 진입합니다. 
    - 좌측 데이터 세트 목록 상단 `+` 버튼을 클릭합니다.
    - 초기 화면에서 `Create` 버튼을 클릭합니다.
3. 다이얼로그에서 생성할 데이터 세트의 이름을 입력 후 `Create` 버튼을 클릭합니다.

### 데이터 세트 버전 생성하기

1. `Versions 섹션`에서  `Create version` 버튼을 클릭합니다. 
2. 다이얼로그에서 `Local file`을 선택합니다.
3. 저장하는 데이터셋의 이름과 설명을 입력합니다.
4. 데이터셋으로 생성할 파일을 파일 탐색기로 선택하거나, Drag&Drop으로 입력합니다.
5. `Create`를 클릭합니다.

## Link

### 패키지 준비

1. (Optional) 튜토리얼에서 사용할 패키지를 설치합니다.
    ```python
    !pip install sklearn pandas numpy
    ```

### 데이터

#### 데이터 불러오기

> 📘 데이터 세트 불러오는 방법에 대한 구체적인 가이드는 **[데이터 세트 가져오기](https://docs.live.mrxrunway.ai/Guide/ml_development/dev_instances/%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%84%B8%ED%8A%B8_%EA%B0%80%EC%A0%B8%EC%98%A4%EA%B8%B0/)** 가이드 에서 확인할 수 있습니다.

1. 노트북 셀 상단의 **Add Runway Snippet** 버튼을 클릭합니다.
2. **Import Dataset** 를 선택합니다. 
3. 사용할 데이터 세트의 버전을 선택하고 **Save** 버튼을 클릭합니다.
4. 버튼 클릭 시 노트북 셀 내 선택한 데이터 세트 내 파일 목록을 조회할 수 있는 스니펫이 작성되며, 해당 데이터 세트 경로를 값으로 갖는 데이터 세트 파라미터가 추가됩니다.  
5. 데이터 세트를 불러오고자 하는 노트북 셀에서 등록된 데이터 세트 파라미터의 이름을 입력하여 작업에 활용합니다.
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

1. 데이터 세트에 포함된 결측치 값을 제거하고, 학습 특성 데이터 세트와 목표 특성 데이터 세트를 분리합니다.
    ```python
    # Drop NA data in dataset
    data_clean = df.dropna()

    # Select Predictor columns
    X = df[["cylinders", "displacement", "weight", "acceleration", "origin"]]

    # Select target column
    y = df["mpg"]
    ```

2. 데이터셋을 학습용 데이터셋과 테스트용 데이터셋으로 분리합니다.

    ```python
    from sklearn.model_selection import train_test_split

    ## Split data into training and testing sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=2024)
    ```

### 모델

#### 모델 클래스

1. 모델 학습을 위한 모델 클래스를 작성합니다.

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

#### 모델 학습

1. 선언한 모델 클래스와 학습용 데이터셋을 활용하여, 모델의 학습과 관련 정보를 로깅합니다.
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

### 모델 업로드

> 📘 모델 업로드 방법에 대한 구체적인 가이드는 **[모델 업로드](https://docs.live.mrxrunway.ai/Guide/ml_development/dev_instances/%EB%AA%A8%EB%8D%B8_%EC%97%85%EB%A1%9C%EB%93%9C/)** 문서에서 확인할 수 있습니다.

1. Runway code snippet 의 save model을 사용해 모델을 저장하는 코드를 생성합니다.
2. 생성된 코드에 필요한 input_sample 을 작성합니다.
    ```python
    import runway

    runway.start_run()
    runway.log_metric("mse", mse)

    input_sample = X_train.sample(1)

    runway.log_model(model_name="auto-mpg-reg-model-sklearn", model=runway_regressor, input_samples={"predict": input_sample})
    runway.stop_run()
    ```

## 파이프라인 구성 및 저장

> 📘 파이프라인 생성 방법에 대한 구체적인 가이드는 **[파이프라인 업로드](https://docs.live.mrxrunway.ai/Guide/ml_development/dev_instances/%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8_%EC%97%85%EB%A1%9C%EB%93%9C/)** 문서에서 확인할 수 있습니다.

1. **Link**에서 파이프라인을 작성하고 정상 실행 여부를 확인합니다.
2. 정상 실행 확인 후, Link pipeline 패널의 **Upload pipeline** 버튼을 클릭합니다.
3. **New Pipeline** 버튼을 클릭합니다.
4. **Pipeline** 필드에 Runway에 저장할 이름을 작성합니다.
5. **Pipeline version** 필드에는 자동으로 버전 1이 선택됩니다.
6. **Upload** 버튼을 클릭합니다.
7. 업로드가 완료되면 프로젝트 내 Pipeline 페이지에 업로드한 파이프라인 항목이 표시됩니다.

