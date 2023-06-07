---
layout: post
title:  "딥러닝 기초 이론"
date:   2023-06-03
featured_image: deeplearning.png
tags: [Deep-Learning]
---
### Machine Learning
> 머신러닝이란, 컴퓨터에게 학습을 명령하는 행위를 말한다.

* 머신러닝의 종류
1. Supervised Learning : 데이터에 정답이 있고, 정답 예측모델을 만들 때. 
ex) 강아지 사진들을 보고 학습 후, 강아지 사진을 맞추는 예측 모델

2. Unsupervised Learning : 정답이 없는 데이터에서, 컴퓨터가 알아서 분류하는 모델
ex) 옷 추천, 영화 추천, 기사 추천등등 

3. Reinforcement Learning : 보상을 주어주고, 이에 가장 적합한 답을 도출하도록 trial and Error 발생시키면서 학습하게 하는 모델

### Deep Learning
> 딥러닝이란, 사람과 같은 인공 신경망인 뉴럴 네트워크를 설계하여, 머신러닝을 진행하는 것을 딥러닝이라고 한다.

* 딥러닝의 주요한 분야
1. Image Classification / Object Detection → computer vision
2. Sequence data 분석, 예측 → 번역, 유전자 시퀀스 등등..

##### Perceptron
- 인공 신경망을 구성하는 하나의 작은 단위라고 생각하면 편하다. 생명에서는 뉴런이랑 비슷한 개념이다.
- 인공 신경망 perceptron도 여러 신호를 받아서, 중요도에 따라 가중치(W)를 곱해주고, 해당 Activation Function을 적용한 후 다음 perceptron으로 넘겨준다. 
![퍼셉트론 이해 그림](https://static.javatpoint.com/tutorial/machine-learning/images/perceptron-in-machine-learning2.png)

모델을 학습시킨다는 것은, 라벨이 있는 데이터를 바탕으로 모든 가중치(w)와 편(bias)의 양호한 값을 학습 (결정)하는 것이다. 머신러닝 알고리즘은 많은 예시를 검사하고 손실을 최소화하는 모델을 찾으려고 시도함으로써 모델을 빌드한다.
##### Loss Function
- 손실은 하나의 예에 대한 모델의 예측이 얼마나 잘못되었는지를 나타내는 숫자라고 한다.
- 손실 함수는 예측모델의 성능, 모델이 일반화할 수 있는 정도를 파악하는 데 도움이 된다.
- 모델 학습의 목표는 모든 예시에서 평균적으로 손실이 낮은 가중치와 편향의 집합을 찾는 것이다.

가장 많이 쓰이는 회귀 선형 모델은 Mean Squared Error 모델이다. 흔히 쓰이지만, 유일하게 쓰이는 방법은 아님!!
**MSE**  = $$1 \over n$$ $$\sum$$ ($$\hat{y}$$-$$y$$)^2 = $$1 \over n$$ $$\sum$$ $$(observation - prediction(x))^2$$

##### Activation Function
- 중간에 있는 Hidden Layer가 있든 없든, 결과는 비슷하다. "뇌처럼 생각하는 공간" 역할을 제대로 하지 않음.
- 원하는 출력에 필요한 뉴런을 활성화하고 선형 입력을 비선형 출력으로 변환함.
- hyperbolic tangent / sigmoid / softmax/ rectified linear 등등의 활성함수들이 존재한다.

##### 경사하강법
- input에 곱해지는 가중치 값(w)의 최적값을 구하기 위해서는 경사하강법(Gradient Descent)라는 간단한 방법을 이용한다.
- 현재 w1값에서의 접선의 기울기를 w1(w1이 조금 변하면 E는 얼마나 변하는가, 편미분)에서 뺌 ($$w1 = w1$$ - a(= learning rate) $$\sigma E \over \sigma w$$)
- 딥러닝 학습과정
1. w값을 무작위로 찍음 
2. *w값을 바탕으로 총손실 E를 계산함*
3. *경사하강법을 이용하여 w를 업데이트 함*
4. *w값을 바탕으로 총손실 E를 계산함*
5. *경사하강법을 이용하여 w를 업데이트 함*
6. ...총손실 E가 더 이상 안줄어들 때까지 계속 업데이트 함