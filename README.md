# face_detection_with_opencv

This module detects face with OpenCV DNN.

## Environment
- OS: Ubuntu 18.04LTS (WSL2)
- Python 3.8 and pipenv are installed

## Step
1. Setup
2. Test with sample data
3. Run with your data

## Setup

```shell
sh setup.sh
pipenv sync
```

## Test with sample data

```shell
pipenv run python face_detection.py -i ./data/lena.jpg
```

<img src="https://github.com/Qstairs/face_detection_with_opencv/raw/main/data/lena.jpg" width="360">

<img src="https://github.com/Qstairs/face_detection_with_opencv/raw/main/result/lena.jpg" width="360">


## Run with your data

```shell
pipenv run python face_detection.py -i {your data path}
```
You can use image data and video data.
