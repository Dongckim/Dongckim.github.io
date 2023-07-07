---
layout: post
title:  "Keras와 전처리를 통한 딥러닝 이해하기"
date:   2023-06-11
featured_image: keras.png
tags: [Deep-Learning]
---
### Keras
케라스는 유저가 손쉽게 딥 러닝을 구현할 수 있도록 도와주는 상위 레벨의 인터페이스로 딥 러닝을 쉽게 구현할 수 있도록 해준다.(파이썬 라이브러리임)

![딥러닝모델](https://wikidocs.net/images/page/152764/2.PNG)
위의 모델처럼 하나의 Layer에는 activation function이 존재해야만 한다.

```
import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential([    #딥러닝 모델 디자인하기
    tf.keras.layers.Dense(64, activation='tanh' ),    #layers.Dense() : 신경망 레이어를 만들어줌
    tf.keras.layers.Dense(128, activation='tanh'),   #()안에는 노드 개수, 개수는 내 마음대로임
    tf.keras.layers.Dense(1, activation='sigmoid'),     #마지막 레이어는 항상 예측결과를 뱉어야 한다.
])

model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
model.fit(np.array(x데이터), np.array(y데이터), epochs=10)  #epochs는 딥러닝 학습횟수를 일컫는다.
```
- 0~1사이의 확률을 뱉고 싶으면 sigmoid함수를 주로 이용한다.
- optimizer의 경우 adam, adagrad, adadelta, rmsprop, sgd등등 있다. 보통은 adam을 제일 많이 사용한다.
- 손실함수의 경우 결과가 0~1사이의 분류/확률문제에서는 binary cross entropy 함수를 많이 쓴다고 함. (외에는 MSE 많이 씀)
- 여기서 x데이터는 정답 예측에 필요한 모든 인풋을, y데이터는 정답 그 자체를 말한다.

### 데이터 전처리 과정
```
import pandas as pd

data = pd.read_csv('gpascore.csv')

print(data.isnull().sum())  #비어있는 칸이 몇개인지를 알 수 있음
data = data.dropna())   #빈칸이 있는 행 삭제
data.fillna()   #()안에 원하는 값으로 빈칸을 채울 수 있음
print(data['gpa'])  #.min() .max() .count() : 내가 원하는 열을 출력, 해당 열중의 최고값, 최대값, 총 갯수 모두 뽑아낼 수 있음

y데이터 = data['admit'].values  #[열 1번째, 열 2번째, 열 3번째, 열 4번째, ...]
x데이터 = []

for i, rows in data.iterrows(): #rows: 줄마다 들어가는 해당 데이터
    x데이터.append([rows['gre'], rows['gpa'], rows['rank']])    #x데이터의 형식은 [[예측에 필요한 첫번째 인풋], [예측에 필요한 두번째 인풋]...]
```

### 학습된 모델로 예측하기
- GRE성적 700점, 학점 3.7점, Rank4 대학에 진학할 확률은요?
```
예측값 = model.predict([700, 3.7, 4])    #뒤에 예측하고 싶은 데이터들 여러개 넣어도 됌.
print(예측값)
```
- 학습모델의 성능향상을 위해서는 연구가 필요하다
- 파라미터 튜닝
- 데이터 전처리의 완성도
