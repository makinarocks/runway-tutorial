# Robotarm Anomaly Detection

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
> 아래의 주피터 노트북을 다운로드 받아 실행할 경우, "pca-model" 이름의 모델이 생성되어 Runway에 저장됩니다.
>
> **[robotarm anomaly detection notebook](https://drive.google.com/uc?export=download&id=10d2Hc4lYx0WOuEvLOkqNTQMpDezbzVzw)**

![link pipeline](../../assets/robotarm_anomaly_detection/link_pipeline.png)

## Runway

> 📘 이 튜토리얼은 4축 로봇팔을 모사한 데이터를 이용해 이상탐지를 수행하는 모델을 생성합니다.
>
> 로봇팔 데이터셋은 아래 항목을 클릭하여 다운로드 할 수 있습니다.  
> **[robotarm-train.csv](https://drive.google.com/uc?export=download&id=1Ks8SUVBQawiKW0q0zQT1sc9um618cdEE)**

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
    !pip install pandas scikit-learn
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

1. 데이터 세트에 인덱스를 설정하고 id 값을 제거한뒤, 총 1000개의 데이터만 사용합니다.

    ```python
    proc_df = raw_df.set_index("datetime").drop(columns=["id"]).tail(1000)
    ```

2. 데이터셋을 학습용 데이터셋과 테스트용 데이터셋으로 분리합니다.

    ```python
    from sklearn.model_selection import train_test_split

    train, valid = train_test_split(proc_df, test_size=0.2, random_state=2024)
    ```

### 모델

#### 모델 클래스

1. 모델 학습을 위한 모델 클래스를 작성합니다.

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

#### 모델 학습

> 📘 Link 파라미터 등록 가이드는 **[파이프라인 파라미터 설정](https://docs.live.mrxrunway.ai/Guide/ml_development/dev_instances/%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8_%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0_%EC%84%A4%EC%A0%95/)** 문서에서 확인할 수 있습니다.할 수 있습니다.

1. PCA에서 사용할 컴포넌트의 개수를 지정하기 위해서 Link 파라미터로 N_COMPONENTS 에 2 를 등록합니다.

    - `N_COMPONENTS`: 2

    ![link parameter](../../assets/robotarm_anomaly_detection/link_parameter.png)

2. 선언한 모델 클래스에 Link 파라미터를 입력하고 학습용 데이터셋을 활용하여 모델을 학습하고 관련된 정보를 로깅합니다.

    ```python
    parameters = {"n_components": N_COMPONENTS}
    detector = PCADetector(n_components=parameters["n_components"])
    detector.fit(train)

    train_pred = detector.predict(train)
    valid_pred = detector.predict(valid)

    mean_train_recon_err = train_pred.mean()
    mean_valid_recon_err = valid_pred.mean()
    ```

### 모델 업로드

> 📘 모델 업로드 방법에 대한 구체적인 가이드는 **[모델 업로드](https://docs.live.mrxrunway.ai/Guide/ml_development/dev_instances/%EB%AA%A8%EB%8D%B8_%EC%97%85%EB%A1%9C%EB%93%9C/)** 문서에서 확인할 수 있습니다.

1. Runway code snippet 의 save model을 사용해 모델을 저장하는 코드를 생성합니다.
2. 생성된 코드에 필요한 input_sample 을 작성합니다.
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

## 파이프라인 구성 및 저장

> 📘 파이프라인 생성 방법에 대한 구체적인 가이드는 **[파이프라인 업로드](https://docs.live.mrxrunway.ai/Guide/ml_development/dev_instances/%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8_%EC%97%85%EB%A1%9C%EB%93%9C/)** 문서에서 확인할 수 있습니다.

1. **Link**에서 파이프라인을 작성하고 정상 실행 여부를 확인합니다.
2. 정상 실행 확인 후, Link pipeline 패널의 **Upload pipeline** 버튼을 클릭합니다.
3. **New Pipeline** 버튼을 클릭합니다.
4. **Pipeline** 필드에 Runway에 저장할 이름을 작성합니다.
5. **Pipeline version** 필드에는 자동으로 버전 1이 선택됩니다.
6. **Upload** 버튼을 클릭합니다.
7. 업로드가 완료되면 프로젝트 내 Pipeline 페이지에 업로드한 파이프라인 항목이 표시됩니다.

