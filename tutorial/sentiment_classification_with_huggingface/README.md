# Hugging Face 기반 감성 분류 (Sentiment Classification with Huggingface)

<h4 align="center">
    <p>
        <b>한국어</b> |
        <a href="README_en.md">English</a>
    <p>
</h4>

<h3 align="center">
    <p>The MLOps platform to Let your AI run</p>
</h3>

## 소개

이 튜토리얼은 Stanford AI Lab에서 공개한 영화 리뷰 데이터를 이용해 감성분석을 수행하는 Huggingface 모델을 학습하여 저장합니다. 작성한 모델 학습 코드를 재학습에 활용하기 위해 파이프라인을 구성하고 저장합니다.

> 📘 빠른 실행을 위해 아래의 주피터 노트북을 활용할 수 있습니다.  
> 아래의 주피터 노트북을 다운로드 받아 실행할 경우, "my-text-model" 이름의 모델이 생성되어 Runway에 저장됩니다.
>
> **[sentiment classification with huggingface](https://drive.google.com/uc?export=download&id=1lbONDH69PuaJXrlxed3P6UlCfLAWaoqo)**

![link pipeline](../../assets/sentiment_classification_with_huggingface/link_pipeline.png)

### 패키지 설치

1. 튜토리얼에서 사용할 패키지를 설치합니다.

```python
!pip install transformers[torch] datasets evaluate
```

## 데이터

Stanford AI Lab에서 공개한 영화 리뷰(Movie Review) 데이터인 IMDB 데이터 세트를 사용합니다. Runway 튜토리얼 폴더에 포함된 parquet 파일을 불러와 데이터 세트를 생성하고, 전처리합니다.


> 📘 이 튜토리얼에서 사용할 IMDB 데이터 세트는 튜토리얼에 맞게 재가공한 [huggingface의 데이터 세트](https://huggingface.co/datasets/imdb/tree/refs%2Fconvert%2Fparquet/plain_text)입니다. 데이터 세트 파일은 `./dataset` 경로에 위치하고 있으며, 필요할 경우 아래 링크를 통해 데이터를 다운로드할 수 있습니다.
> **[IMDB test dataset](https://drive.google.com/uc?export=download&id=1QlIzPfOw_b0xXnXM6rxnW3Vbr-VDm0At)**

### 데이터 불러오기

1. 파일 탐색기에서 데이터 세트 파일의 경로를 확인합니다.
2. RUNWAY_DATA_PATH 파라미터에 데이터 파일의 경로를 할당합니다.

    ```python
    import os
    import pandas as pd

    RUNWAY_DATA_PATH = "/home/jovyan/workspace/examples/tutorial/sentiment_classification_with_huggingface/dataset"
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
3. Pandas 데이터 프레임으로 Huggingface Dataset을 생성합니다.
    ```python
    from datasets import Dataset

    ds = Dataset.from_pandas(df.sample(1000))
    ds.set_format("pt")
    ```

### 토크나이징

1. Transformer 의 `AutoModelForSequenceClassification` 모듈을 이용해 모델을 불러오고 토크나이저를 초기화합니다.

    ```python
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # model
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ARCH_NAME, num_labels=2, id2label=id2label, label2id=label2id
    )
    model.config.pad_token_id = model.config.eos_token_id

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ARCH_NAME)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # cuda setting if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ```

2. 토크나이저를 적용하여 데이터를 전처리합니다.
   ``` python
    ds_proc = ds.map(lambda x: tokenizer(x["text"], truncation=True))
   ```

## 모델
### 모델 학습

1. `Trainer` API를 사용해 감성 분석 모델을 학습합니다. 

    ```python
    from transformers import TrainingArguments, Trainer, DataCollatorWithPadding


    training_args = TrainingArguments(
        output_dir="tmp",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_proc,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    history = trainer.train()
    ```

### 모델 랩핑 클래스 선언

1. API 서빙에 이용할 수 있도록 `HuggingModel` 클래스를 작성합니다.

    ```python

    import mlflow
    import pandas as pd


    class HuggingModel(mlflow.pyfunc.PythonModel):
        def __init__(self, pipeline):
            self.pipeline = pipeline

        def predict(self, context, X):
            result = self.pipeline(X["text"].to_list())
            return pd.DataFrame.from_dict(result)
    ```

2. Transformer 파이프라인을 생성하고 `HuggingModel`로 랩핑합니다.

    ```python
    from transformers import pipeline


    model = model.to("cpu")
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    hug_model = HuggingModel(pipe)
    ```

### 모델 등록

학습이 완료된 모델을 Runway에 등록하여 추론 서비스에서 사용할 수 있도록 합니다.

1. Runway 플랫폼의 모델 등록 코드 스니펫을 사용하여, 학습이 완료된 모델을 등록(`log_model`)하고 관련 정보를 기록합니다.
    ```python
    import mlflow
    import runway

    with mlflow.start_run():
        mlflow.log_metrics(history.metrics)

        runway.log_model(
            model=hug_model,
            input_samples={"predict": df.sample(1).drop(columns=["label"])},
            model_name="my-text-model",
        )
    ```

## 파이프라인 구성 및 저장

> 📘 파이프라인 생성 방법에 대한 구체적인 가이드는 **[파이프라인 구성](https://docs.live.mrxrunway.ai/guide/core-features/dev-instances/create-a-pipeline/)** 문서에서 확인할 수 있습니다.

1. **Link**에서 파이프라인을 작성하고 정상 실행 여부를 확인합니다.
2. 정상 실행 확인 후, Link pipeline 패널의 **Upload pipeline** 버튼을 클릭합니다.
3. **New Pipeline** 버튼을 클릭합니다.
4. **Pipeline** 필드에 Runway에 저장할 이름을 작성합니다.
5. **Pipeline version** 필드에는 자동으로 버전 1이 선택됩니다.
6. **Upload** 버튼을 클릭합니다.
7. 업로드가 완료되면 프로젝트 내 Pipeline 페이지에 업로드한 파이프라인 항목이 표시됩니다.


## 모델 배포

> 📘 모델 배포 방법에 대한 구체적인 가이드는 **[모델 배포](https://docs.live.mrxrunway.ai/guide/core-features/inference-services/deploying-models/)** 문서에서 확인할 수 있습니다.

## 데모 사이트

1. 배포된 모델을 실험하기 위한 [데모 사이트](http://demo.service.mrxrunway.ai/object)에 접속합니다.
2. 데모사이트에 접속하면 아래와 같은 화면이 나옵니다.

    ![demo web](../../assets/sentiment_classification_with_huggingface/demo-web.png)

3. API Endpoint, 발급 받은 API Token, 예측할 문장을 입력합니다.

    ![demo fill field](../../assets/sentiment_classification_with_huggingface/demo-fill-field.png)

4. 결과를 받을 수 있습니다.

    ![demo result](../../assets/sentiment_classification_with_huggingface/demo-result.png)
