## FacePlus

A simple model for face recognition and facial expression recognition

[![](https://img.shields.io/badge/Python-3.7-yellow)](https://www.python.org/)
[![](https://img.shields.io/badge/PyTorch-1.6.0-brightgreen)](https://github.com/pytorch/pytorch)
[![](https://img.shields.io/badge/Numpy-1.19.1-red)](https://github.com/numpy/numpy/)
[![](https://img.shields.io/badge/Cv2-4.1.2-blue)](https://github.com/opencv/opencv)
[![](https://img.shields.io/badge/CUDA-10.0-orange)](https://developer.nvidia.com/cuda-downloads)
[![](https://img.shields.io/badge/Lmdb-1.0-brightyellow)](https://pypi.org/project/lmdb/)

### Usage

#### prepare dataset

project structure is as follows

```
├── lib
|   ├── FaceRecognizer
|   |   ├── data
|   |   |   ├── face
|   |   |   |   ├── resource # origin dataset
|   |   |   |   ├── transformed  # transformed dataset
|   |   |   |   ├── image2lmdb.py
|   |   |   |   ├── ....
|   |   ├── ...
│   ├── FacialExpressionRecognizer
|   |   ├── data
|   |   |   ├── fer2013
|   |   |   |   ├── resource # origin dataset
|   |   |   |   ├── transformed  # transformed dataset
|   |   |   |   ├── image2lmdb.py
|   |   |   |   ├── ....
|   |   ├── ...
│   |── ...
│
├── log # log files
│
├── main.py # training code
├── config.py # some configs
├── ...
```

You can download face dataset from [this](https://aistudio.baidu.com/aistudio/datasetdetail/27604)(thanks very much to the provider of the dataset), and unzip it in `./lib/FaceRecognizer/data/face/resource`, but you must put `train.txt` and `test.txt` in `./lib/FaceRecognizer/data/face`. Then you can run

```bash
python image2lmdb.py
```

to transform dataset.

In the same way, download facial expression dataset from [this](https://www.kaggle.com/deadskull7/fer2013), and put the `fer2013.csv` in `./lib/FacialExpressionRecognizer/data/fer2013/resource`.
Then you can run

```bash
python image2lmdb.py
```

to transform dataset.

#### train

Run

```bash
python main.py [--configs]
```

to train model. You can see more arguments in `config.py`

#### infer

For face detect, put your face images in `./lib/FaceRecognizer/faceback/images` just like this

```
├── facebank
|   ├── person1
|   |   ├── 1.jpg
|   |   ├── ...
|   ├── person2
|   |   ├── 1.jpg
|   |   ├── ...
|   ├── ...
```

and run

```
python genFacebank.py
```

to generate features. Then you can run `infer.py` to perform face retrieval.

For facial expression recognition, you could just run `infer.py`.

### Reference

[A Real-time Facial Expression Recognizer using Deep Neural Network.](https://www.researchgate.net/publication/311489862_A_Real-time_Facial_Expression_Recognizer_using_Deep_Neural_Network)

[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)

[InsightFace_Pytorch by TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)