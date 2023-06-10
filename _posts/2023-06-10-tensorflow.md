---
layout: post
title:  "Tensorflow & Tensor"
date:   2023-06-10
featured_image: Tensorflow.png
tags: [Deep-Learning]
---
### Tensorflow
- 딥러닝 계산을 조금 더 쉽게 도와주는 라이브러리일뿐이다.
- 직접 파이썬 쌩 코딩으로 짜면 매우 코드가 길고 복잡하기 때문에 Tensor라는 자료형을 이용한 Tensorflow를 이용한다.
- 최종 예측값(yHat)
- 가중치(w)값 업데이트 (back propagation)
- learning rate 만들기
- loss function 만들기
- 레이어 만들기

> **리스트와 tensor의 차이점**
: 기존의 리스트에도 원하는 W값들을 저장해두고 사용할 수 있다.
-> 그치만 자료들의 dimension이라고 불리는 차원이 좀 높아지면, tensor로 다루기가 훨씬 편하다.(이미지와 같은 3차원 자료형들)

![노드](https://codingapple.com/wp-content/uploads/2020/09/%EC%BA%A1%EC%B2%981.png)
```
노드1 = 10*w1 + 20*w2 + 30*w3 + 40*w4
```
1차원이라 겨우 계산할 정도이지, 차원이 높아질수록 계산이 어려워짐
그렇기 때문에 행렬 연산을 실시한다.
![노드](https://codingapple.com/wp-content/uploads/2020/09/%EC%BA%A1%EC%B2%982-2.png)

```
행렬X = tf.constant([10,20,30,40])
행렬W = tf.constant([w1,w2,w3,w4])
행렬W뒤집은거 = tf.transpose(행렬W)

노드1 = tf.matmul(행렬X, 행렬W뒤집은거)
print(노드1)
```

### 여러가지 tf메서드들
- tf.zeros() : 0이 담긴 텐서를 만들어줌
```
ex) tf.zeros([2,2,3])
= tf.Tensor([[[0.0.0.],[0.0.0.]], [[0.0.0.],[0.0.0.]]])
```
- tf.shape() : tf 행렬의 형태를 출력함
```
ex) tf.constant([[1,2,3],[3,4,5]])
= (2,3)
```
- tensor의 datatype : (1)정수는 int, (2)실수는 float -> tf.cast() : 자료의 데이터타입형을 바꿀 수 있음
- tf.constant 는 변하지 않는 상수를 지정 / tf.Variable() 는 변수인데, 딥러닝 상에서는 가중치값(W)라고 생각하면 된다.
```
w = tf.Variable(1.0) //변수 생성 및 1.0 할당
w.assign(2) //변수 재할당(수정)
```

### 단순 개수의 데이터 딥러닝 문제 풀어보기 (키로 신발사이즈를 추론해보자)
```
import tensorflow as tf
키 = 170
신발 = 260
# 신발 = 키 * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def 손실함수():
    예측값 = 키*a+b
    return tf.square(260 - 예측값)

opt = tf.keras.optimizers.Adam(learning_rate=0.1) //경사하강법 메서드(optimizers), gradient를 알아서 스마트하게 바꿔줌(Adam)
opt.minimize(손실함수, var_list=[a,b]) // var_list는 경사하강법을 이용해서 업데이트할 변수 목록을 모두 적어준다

# 경사하강법 1번 이루어짐
```

```
import tensorflow as tf
키 = 170
신발 = 260
# 신발 = 키 * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def 손실함수():
    예측값 = 키*a+b
    return tf.square(260 - 예측값)

opt = tf.keras.optimizers.Adam(learning_rate=0.1)
for i in range(300):
    opt.minimize(손실함수, var_list=[a,b])
    print(a.numpy(),b.numpy())
    
# 반복문 수만큼 반복하여 알맞는 a,b 값이 나옴
```


### 여러개의 데이터 딥러닝 문제 풀어보기 (행렬간의 관계 풀어내보기)
```
import tensorflow as tf
train_x = [1,2,3,4,5,6,7]
train_y = [3,5,7,9,11,13,15]

a = tf.Variable(0.1)
b = tf.Variable(0.1)

def 손실함수(a,b):
    예측_y = train_x * a + b //tensorflow가 행렬의 곱처럼 알아서 하나하나씩 곱셈해줌
    return tf.keras.losses.mse(train_y, 예측_y) //손실함수 중 하나인 Mean Square Error를 tensorflow에서 메서드로 지원해줌

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

for i in range(400):
    opt.minimize(lambda:손실함수(a,b), var_list=[a,b])
    print(a.numpy(),b.numpy())
    
```
