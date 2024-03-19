# Object Detection

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

We use the Link included in Runway to train and save the image model.
To utilize the written model training code for retraining, we construct and save a pipeline.

> 📘 For quick execution, you can use the Jupyter Notebook provided below.
> If you download and run the Jupyter Notebook, a model named "my-detection-model" will be created and saved in Runway.
>
> **[object detection notebook](https://drive.google.com/uc?export=download&id=1WgdswAqXZtRE-BMJXpiFIBYHV-oboV4F)**

![link notebook](../../assets/object_detection/link_pipeline.png)

## Runway

> 📘 Currently, Runway only supports COCO-format config.
>
> You can download the available config for the received data source by clicking on the items below.
> **[coco-sample-dataset.zip](https://drive.google.com/uc?export=download&id=1TrM3y8aRRmaYnIlDI902p73Lsw0XC89B)**

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

1. Install the required packages for the tutorial.
    ```python
    !pip install torch torchvision Pillow seaborn torchmetrics
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

#### Extract a sample image

1. Extract a sample data and check the image.

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

### Training

#### COCO Dataset

1. To train the model, create a Dataset provided by PyTorch.

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

2. Use the declared data to create a data loader.

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

### Model Declaration

1. Declare the model to be used for training. In this tutorial, we use the `fasterrcnn_resnet50_fpn` model from PyTorch.
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

### Model Training

> 📘 You can find guidance on registering Link parameters in the **[Set Pipeline Parameter](https://docs.live.mrxrunway.ai/en/Guide/ml_development/dev_instances/%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8_%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0_%EC%84%A4%EC%A0%95/)**.

1. Set the number of epochs for model training by registering 1 in the `N_EPOCHS` Link parameter.
2. Train the declared model using the data loader created above and evaluate the trained model.

    ```python
    import torch.optim as optim


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

    model.eval()
    torch.cuda.empty_cache()

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

### Model Inference

#### Model Wrapping Class Declaration

1. Create a ModelWrapper to serve the trained model.

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
2. Wrap the trained model with ModelWrapper.

    ```python
    model = model.cpu()
    device = "cpu"
    serve_model = ModelWrapper(model=model, device=device)
    ```

#### Sample Image Inference

1. Currently, Runway only supports Dataframe format for input and output in API serving. To do this, write code to convert the input images to bytearrays.

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

2. Create an `input_sample` using the data and conversion code above, and  perform inference using the wrapped model.
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

3. Check the inference results.  
   ![predict result](../../assets/object_detection/predict_result.png)

### Upload Model

> 📘 You can find detailed instructions on how to save the model in the [Upload Model](https://docs.live.mrxrunway.ai/en/Guide/ml_development/dev_instances/%EB%AA%A8%EB%8D%B8_%EC%97%85%EB%A1%9C%EB%93%9C/).
1. Generate the code to save the model using the save model option from the Runway code snippet. Also, log the information that are related to model.

    ```python
    import runway

    del map_score["classes"]
    runway.start_run()
    runway.log_metrics(map_score)

    runway.log_model(model_name="my-detection-model", model=serve_model, input_samples={'predict': input_sample})
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


## Model Deployment

> 📘 You can find specific guidance on model deployment in the **[Model Deployment](https://docs.live.mrxrunway.ai/en/Guide/ml_serving/model_deployments/%EB%AA%A8%EB%8D%B8_%EB%B0%B0%ED%8F%AC/)**.
## Demo Service

1. To test the deployed model, you can use the following [demo website](http://demo.service.mrxrunway.ai/object).
2. If you are in demo site you will see the following screen:

    ![demo web](../../assets/object_detection/demo-web.png)

3. Fill in the API Endpoint, API Token, and upload the image for prediction:

    ![demo fill field](../../assets/object_detection/demo-fill-field.png)

4. You will receive the result:

    ![demo result](../../assets/object_detection/demo-result.png)
