# Object Detection

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

Runway에 포함된 Link를 사용하여 이미지 모델을 학습하고 저장합니다.  
작성한 모델 학습 코드를 재학습에 활용하기 위해 파이프라인을 구성하고 저장합니다.

> 📘 빠른 실행을 위해 아래의 주피터 노트북을 활용할 수 있습니다.  
> 아래의 주피터 노트북을 다운로드 받아 실행할 경우, "my-detection-model" 이름의 모델이 생성되어 Runway에 저장됩니다.
>
> **[object detection notebook](https://drive.google.com/uc?export=download&id=1WgdswAqXZtRE-BMJXpiFIBYHV-oboV4F)**

![link notebook](../../assets/object_detection/link_pipeline.png)

## Runway

> 📘 이 튜토리얼은 COCO 데이터 셋의 일부를 사용해 객체 탐지를 수행하는 모델을 생성합니다.
>
> COCO 샘플 데이터셋은 아래 항목을 클릭하여 다운로드 할 수 있습니다.  
> **[coco-sample-dataset.zip](https://drive.google.com/uc?export=download&id=1TrM3y8aRRmaYnIlDI902p73Lsw0XC89B)**

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

### 패키지 설치

1. 튜토리얼에서 사용할 패키지를 설치합니다.
    ```python
    !pip install torch torchvision Pillow seaborn torchmetrics
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
    from pycocotools.coco import COCO

    # RUNWAY_DATA_PATH was added to pipeline parameters.
    # Pipeline parameters can be used in the cell added as a pipeline component.
    # RUNWAY_DATA_PATH = "/home/jovyan/workspace/dataset/sample-coco"
    config_file = None
    for dirname, _, filenames in os.walk(RUNWAY_DATA_PATH):
        for filename in filenames:
            if filename.endswith(".json"):
                config_file = os.path.join(dirname, filename)

    if config_file is None:
        raise ValueError("Can't find config file in given dataset")

    coco = COCO(config_file)
    ```

#### 예제 데이터 추출

1. 샘플 데이터 하나를 추출 후 이미지를 확인합니다.

    ```python
    from pathlib import Path
    from matplotlib.pyplot import imshow
    from PIL import Image


    sample_image_path = Path(RUNWAY_DATA_PATH).parent / "000000000139.jpg"
    image_filename_list = [sample_image_path]

    img = Image.open(sample_image_path)
    imshow(img)
    ```

    ![sample image](../../assets/object_detection/sample_image.png)

### 학습

#### COCO 데이터셋

1. 모델을 학습하기 위해서 pytorch 에서 제공하는 Dataset 을 생성합니다.

    ```python
    from PIL import Image
    from pathlib import Path
    from pycocotools.coco import COCO
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms as T


    def get_transforms():
        transforms = []
        transforms.append(T.ToTensor())
        return T.Compose(transforms)


    def collate_fn(batch):
        return tuple(zip(*batch))


    class COCODataset(Dataset):
        def __init__(self, data_root, coco, transforms=None):
            self.data_root = Path(data_root)
            self.transforms = transforms
            # pre-loaded variables
            self.coco = coco
            self.ids = list(sorted(self.coco.imgs.keys()))

        def __getitem__(self, index):
            ## refer to https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
            img_id = self.ids[index]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            ann = self.coco.loadAnns(ann_ids)
            img_path = self.data_root / self.coco.loadImgs(img_id)[0]["file_name"]
            img = Image.open(img_path)
            num_objs = len(ann)

            boxes = []
            for i in range(num_objs):
                boxes.append([
                    ann[i]["bbox"][0],
                    ann[i]["bbox"][1],
                    ann[i]["bbox"][2] + ann[i]["bbox"][0],
                    ann[i]["bbox"][3] + ann[i]["bbox"][1],
                ])

            areas = []
            for i in range(num_objs):
                areas.append(ann[i]["area"])

            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.ones((num_objs,), dtype=torch.int64),
                "image_id": torch.tensor([img_id]),
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros((num_objs,), dtype=torch.int64),
            }

            ## transform image
            if self.transforms is not None:
                img = self.transforms(img)

            return img, target

        def __len__(self):
            return len(self.ids)
    ```

2. 선언한 데이터를 이용해 데이터 로더를 생성합니다.

    ```python
    from torch.utils.data import DataLoader

    ## Define Train dataset
    data_root = Path(RUNWAY_DATA_PATH).parent
    dataset = COCODataset(data_root, coco, get_transforms())

    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    ```

### 모델 선언

1. 학습에 사용할 모델을 선언합니다. 튜토리얼에서는 pytorch 의 `fasterrcnn_resnet50_fpn` 모델을 사용합니다.
    ```python
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn


    # Define local variables
    print(torch.cuda.is_available())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    try:
        entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device)
    except:
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None).to(device)
    ```

### 모델 학습

> 📘 Link 파라미터 등록 가이드는 **[파이프라인 파라미터 설정](https://docs.live.mrxrunway.ai/Guide/ml_development/dev_instances/%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8_%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0_%EC%84%A4%EC%A0%95/)** 문서에서 확인할 수 있습니다.할 수 있습니다.

1. 모델을 학습할 Epoch 을 설정할 수 있도록 Link 파라미터로 N_EPOCHS 에 1을 등록합니다.
2. 선언한 모델을 위에서 만든 데이터 로더를 통해 학습하고 모델의 성능을 평가합니다.

    ```python
    import torch.optim as optim
    from torchmetrics.detection import MeanAveragePrecision

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=1e-5)

    model.train()
    for epoch in range(N_EPOCHS):
        for imgs, annotations in data_loader:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

    map_metric = MeanAveragePrecision().to(device)
    model.eval()
    with torch.no_grad():
        preds = []
        annos = []
        for imgs, annotations in data_loader:
            pred = model(list(img.to(device) for img in imgs))
            anno = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            preds.extend(pred)
            annos.extend(anno)

    map_metric.update(preds, annos)
    map_score = map_metric.compute()

    torch.cuda.empty_cache()
    ```

### 모델 추론

#### 모델 랩핑 클래스 선언

1. 학습된 모델을 서빙할 수 있도록 ModelWrapper를 작성합니다.
    ```python
    import io
    import base64

    import torch
    import pandas as pd
    import numpy as np
    from torchvision import transforms
    from PIL import Image, ImageDraw, ImageFont


    class ModelWrapper:
        def __init__(self, model, device):
            self.model = model
            self.device = device

        def bytesarray_to_tensor(self, bytes_array: str):
            # input : "utf-8" decoded bytes_array
            encoded_bytes_array = bytes_array.encode("utf-8")
            # decode encoded_bytes_array with ascii code
            img_64_decode = base64.b64decode(encoded_bytes_array)
            # get image file and transform to tensor
            image_from_bytes = Image.open(io.BytesIO(img_64_decode))
            return transforms.ToTensor()(image_from_bytes).to(self.device)

        def numpy_to_bytesarray(self, numpy_array):
            numpy_array_bytes_array = numpy_array.tobytes()
            numpy_array_64_encode = base64.b64encode(numpy_array_bytes_array)
            bytes_array = numpy_array_64_encode.decode("utf-8")
            return bytes_array

        def draw_detection(self, img_tensor, bboxes, labels, scores, out_img_file):
            """Draw detection result."""
            img_array = img_tensor.permute(1, 2, 0).numpy() * 255
            img = Image.fromarray(img_array.astype(np.uint8))
            
            draw = ImageDraw.Draw(img)    
            font = ImageFont.load_default()
            bboxes = bboxes.cpu().numpy().astype(np.int32)
            labels = labels.cpu().numpy()
            scores = scores.cpu().numpy()
            for box, label, score in zip(bboxes, labels, scores):        
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=1)  
                text = f"{label}: {score:.2f}"
                draw.text((box[0], box[1]), text, fill="red", font=font)
            img.save(out_img_file)
            return img

        @torch.no_grad()
        def predict(self, df):
            self.model.eval()
            # df is 1-d dataframe with bytes array
            tensor_list = list((map(self.bytesarray_to_tensor, df.squeeze(axis=1).to_list())))

            pred_images = []
            pred_image_shape_c = []
            pred_image_shape_h = []
            pred_image_shape_w = []
            pred_image_dtypes = []

            boxes = []
            labels = []
            scores = []

            boxes_dtypes = []
            labels_dtypes = []
            scores_dtypes = []

            for img in tensor_list:
                output = self.model(img.unsqueeze(0))
                detect_img = self.draw_detection(
                    img_tensor=img,
                    bboxes=output[0]["boxes"],
                    labels=output[0]["labels"],
                    scores=output[0]["scores"],
                    out_img_file="test.png",
                )
                detect_img = np.array(detect_img)
                h, w, c = detect_img.shape
                box = output[0]["boxes"].cpu().numpy()
                label = output[0]["labels"].cpu().numpy()
                score = output[0]["scores"].cpu().numpy()

                pred_images += [detect_img]
                boxes += [box]
                labels += [label]
                scores += [score]

                pred_image_shape_c += [c]
                pred_image_shape_h += [h]
                pred_image_shape_w += [w]

                pred_image_dtypes += [str(detect_img.dtype)]
                boxes_dtypes += [str(box.dtype)]
                labels_dtypes += [str(label.dtype)]
                scores_dtypes += [str(score.dtype)]

                torch.cuda.empty_cache()

            meta = pd.DataFrame({
                "pred_image_shape_c": pred_image_shape_c,
                "pred_image_shape_h": pred_image_shape_h,
                "pred_image_shape_w": pred_image_shape_w,
                "output_dtype": pred_image_dtypes,
                "boxes_dtypes": boxes_dtypes,
                "labels_dtypes": labels_dtypes,
                "scores_dtypes": scores_dtypes,
            })
            img_byte = pd.DataFrame({
                "output": pred_images,
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
                # "true": tensor_list,
            }).applymap(lambda x: self.numpy_to_bytesarray(x))
            return pd.concat([meta, img_byte], axis="columns")
    ```

#### 샘플 이미지 추론

1. Runway 에서는 API 서빙을 위한 입력과 출력을 Dataframe 형식만 지원하고 있습니다. 이를 위해서 입력 이미지를 bytearray 로 변환해주는 코드를 작성합니다.

    ```python
    import base64
    import pandas as pd


    def convert_image_to_bytearray(img_binary):
        image_64_encode = base64.b64encode(img_binary)
        bytes_array = image_64_encode.decode("utf-8")
        return bytes_array


    def images_to_bytearray_df(image_filename_list: list):
        df_list = []
        for img_filename in image_filename_list:
            image = open(img_filename, "rb")  # open binary file in read mode
            image_read = image.read()
            df_list.append(convert_image_to_bytearray(image_read))
        return pd.DataFrame(df_list, columns=["image_data"])
    ```

2. 위에서 사용한 데이터와 변환 코드를 이용해 `input_sample` 을 생성하고 모델을 랩핑한 객체를 이용해 추론합니다.

    ```python
    model = model.cpu()
    device = "cpu"
    serve_model = ModelWrapper(model=model, device=device)

    # make input sample
    input_sample = images_to_bytearray_df(image_filename_list)

    # For inference
    pred = serve_model.predict(input_sample)

    output = pred.loc[0]
    data, dtype = output["output"], output["output_dtype"]
    c, h, w = output["pred_image_shape_c"], output["pred_image_shape_h"], output["pred_image_shape_w"]

    type_dict = {"uint8": np.uint8, "float32": np.float32, "int64": np.int64}
    pred_decode = base64.b64decode(data)
    pred_array = np.frombuffer(pred_decode, dtype=type_dict[dtype])

    img = Image.fromarray(pred_array.reshape(h, w, c))

    imshow(img)
    ```

3. 추론 결과를 확인합니다.
   ![predict result](../../assets/object_detection/predict_result.png)

### 모델 업로드

> 📘 모델 업로드 방법에 대한 구체적인 가이드는 **[모델 업로드](https://docs.mrxrunway.ai/docs/모델-저장)** 문서에서 확인할 수 있습니다.

1. Runway code snippet 의 save model을 사용해 모델을 저장하는 코드를 생성합니다. 그리고 모델 과 관련된 정보를 저장합니다.
    ```python
    import runway

    del map_score["classes"]
    runway.start_run()
    runway.log_metrics(map_score)

    runway.log_model(model_name="my-detection-model", model=serve_model, input_samples={'predict': input_sample})
    ```

## 파이프라인 구성 및 저장

> 📘 파이프라인 생성 방법에 대한 구체적인 가이드는 **[파이프라인 생성](https://dash.readme.com/project/makinarocks-runway/docs/파이프라인-생성)** 문서에서 확인할 수 있습니다.

1. **Link**에서 파이프라인을 작성하고 정상 실행 여부를 확인합니다.
2. 정상 실행 확인 후, Link pipeline 패널의 **Upload pipeline** 버튼을 클릭합니다.
3. **New Pipeline** 버튼을 클릭합니다.
4. **Pipeline** 필드에 Runway에 저장할 이름을 작성합니다.
5. **Pipeline version** 필드에는 자동으로 버전 1이 선택됩니다.
6. **Upload** 버튼을 클릭합니다.
7. 업로드가 완료되면 프로젝트 내 Pipeline 페이지에 업로드한 파이프라인 항목이 표시됩니다.


## 모델 배포

> 📘 모델 배포 방법에 대한 구체적인 가이드는 **[모델 배포](https://docs.live.mrxrunway.ai/Guide/ml_serving/model_deployments/%EB%AA%A8%EB%8D%B8_%EB%B0%B0%ED%8F%AC/)** 문서에서 확인할 수 있습니다.

## 데모 사이트

1. 배포된 모델을 실험하기 위한 [데모 사이트](http://demo.service.mrxrunway.ai/object)에 접속합니다.
2. 데모사이트에 접속하면 아래와 같은 화면이 나옵니다.

    ![demo web](../../assets/object_detection/demo-web.png)

3. API Endpoint, 발급 받은 API Token, 예측에 사용할 이미지를 업로드합니다.

    ![demo fill field](../../assets/object_detection/demo-fill-field.png)

4. 결과를 받을 수 있습니다.

    ![demo result](../../assets/object_detection/demo-result.png)
