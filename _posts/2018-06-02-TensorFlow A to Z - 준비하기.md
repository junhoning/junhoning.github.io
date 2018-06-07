---
title: "TensorFlow A to Z - 준비하기"
categories:
  - TensorFlow Basic
tags:
  - TensorFlow
  - Basic
---

# TensorFlow A to Z - 준비하기

딥러닝을 공부하면서 이론도 공부 했고, 기본 예제들도 많이 돌려본 사람들이 많을 것이다. 하지만 부분부분 중간 예제들만 돌려봤을 뿐 본인이 처음부터 끝까지 만들어본다거나 직접 데이터들을 캐글이나 온라인에서 얻은 데이터들을 직접 돌리다 어려워 하는 사람들이 많을 것이다. 그래서 직접 데이터를 넣고 직접 학습을 해보고 싶은 사람들을 위해 데이터를 넣고, 평가까지 하는 과정을 소개하는 글을 준비 했다. 

전체적인 다이아그램은 아래의 그림과 같다. 

![](https://www.notion.so/file/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fe7a56dde-954c-4571-803d-7b2fb40c0cbf%2FNNDiagram.jpg)

사람들마다 패키징 하는 스타일들이 다들 다양하게 가지고 있겠지만, 위의 방법은 내가 쓰는 방법을
 예시로 그렸다. 먼저 가운데 상자 안에 담긴 Graph라는 상자 안에 Model, Optimizer이 담겨 있다. Grpah 안에 학습을 돌리기 위한 Data를 넣어줄 것인데. 어떤 것을 어떻게 넣을 것인지는 Data Manager가 그 역할을 해줄 것이다. 

Model, Optimization이 담긴 Graph 는 우리가 학습하고자 하는 것은 이 graph 인 것이다. 학습까지 완성 시킨 모델을 이용해 예측하기 위해선 Freeze를 하여 pb 파일로 뽑아내는데. 이때 Graph 안에서 테스트할 데이터가 들어갈 Input을 먼저 지정해줄 것이고, Model 결과를 output으로 출력 할 수도 있고, softmax나 다른 후처리까지 거친 과정을 output으로 지정 해줄 수 있다. 

당연하겠지만 우린 Graph 안의 Model을 학습 시키는 것이 목표다. Model에서 나온 Inference 결과를 Optimizer로 최적의 weight 값을 찾아갈 것이고, 그 최적의 값을 찾아가는 것은 Train에서 할 것이다. 

## 진행과정

![](https://www.notion.so/file/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fc974a26f-ae30-42d2-a4ad-0571073af7d7%2Fprocess.jpg)

파이썬 파일은 크게 5개의 파일로 나누어서 만들 것이다.  제일 먼저 모델을 다 만들면 학습을 시작한다. 학습을 하면서 나온 checkpoint 를 통해 모델을 freeze 하여 pb 파일을 뽑아낼 것이다. 뽑아낸 pb 파일을 통해 Test 할 그림을 넣어 작동이 잘 되는지 최종 확인 할 것이다. 그렇게 되면 당신만의 신경망 모델이 완성 되는 것이다. 

## 준비

요리사들에게 요리를 하기 위해 음식 재료와 부엌이 필요하듯 우리에게 필요한 것은 데이터셋이다. 나는 잘 정돈 되어있는 캐글에서 데이터셋을 얻도록 하겠다. 

[https://www.kaggle.com/scolianni/mnistasjpg/data](https://www.kaggle.com/scolianni/mnistasjpg/data)

위의 링크를 따라가면 mnist 데이터를 다운 받을 수 있다. MNIST를 다루다보면 csv 형태로 담겨있거나 binary나 1d array 형태로 있을 때가 있는데가 있는데. 보통 우리가 받는 이미지 데이터셋하고 형태가 달라 개인적으로 입문용으로 권하진 않는다. 우린 위의 링크에서 jpg 형태로 담겨져 있는 MNIST 데이터셋을 받을 것이다. 

trainingSet.tar.gz을 받고서 압축을 풀어서 열어보면 trainingSet이라는 폴더가 있다. 우린 원하는 위치에 폴더를 아무 이름으로 (예로 mnist_basic)을 만들고 그 안에 담을 것이다. 

그리고나서 그 폴더 안에는 같은 경로 안에 파이썬 파일 data_manager.py, model.py, [train.p](http://train.py)y, [util.py](http://util.py) 를 만들고 나면 준비는 완료다. 

data_manager.py 안에는 데이터의 경로를 지정하여 image와 label 데이터셋을 준비하고 model에 넣는 역할을 만들 것이고, model.py에는 tensorflow로 구현한 모델, [graph.py](http://graph.py)는 그 모델을 학습 시키기 위한 graph와 optimizer를 만들 것이고, train.py에서 모델을 학습시키고 test까지 가능하도록 할 것이다. 그리고 마지막으로 util.py에서는 학습한 모델을 freeze 해서 pb로 뽑고, checkpoint 등 부가적으로 필요한 기능들을 넣을 계획이다. 

글이 너무 길어지니 다음 글에서는 data_manager부터 구현할 것이다.
