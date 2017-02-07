#Chapter 1

##소개

Deep convolutional nerual networks (CNNs)은 딥러닝의 진보의 핵심이었다.. CNNs이 글자 인식작업의 9할에 사용되었음에도 불구하고 현재의 다양한 작업들은 훨씬 최근에, deep CNN이 ImageNet 이미지 인식 첼린지☆에서 챔피언을 먹을때 나오게 되었다.
따라서 Convolutional neural networks는 머신러닝을 배우는 사람들에겐 매우 편리한 도구가 되었다. 하지만, 처음으로 CNNs를 배우는건 일반적으로 어려운 경험이다. Convolutional layer의 output의 형태는 이의 input의 형태, kernel shape, zero padding과 strides, 그리고 이의 속성들 사이의 관계가 trivial인지 infer인지에 영향을 받는다. 이건 output size가 input size와 전혀 상관 없는 fully-connected layers와 비교된다. 게다가, CNN은 fully-connected network와는 또 다른 수준의 복잡성을 가지는, pooling stage를 가지기도 한다. 마지막으로, transposed convolutional layers (fractionally strided convolutional layers로도 알려져 있음)은 최근에 더 많은 작업에 사용되어지고 있고, 이 convolutional layers과의 관계는 딱히 명확하게 설명이 되질 않는다.

이 가이드의 목적은 두개다:

1. convolutional layers와 transposed convolutional layers과의 관계에 대해 설명.
2. [input shape, kernel shape, zero padding, strides and output shape in convolutional, pooling and transposed convolutional layers]들의 관계에 대한 직관적인 이해.

광범위하게 응용할 수 있게 하기 위해, 이 가이드에서 보여지는 결과는 자세한 구현을 하기보다는 Theano, Torch, Tensorflow, Caffe같이 주로 쓰이는 머신러닝 프레임워크를 사용하였다.
이 챕터에서는 간단하게 CNNs의 메인이 되는 discrete convolutions과 pooling에 대해 알아볼것이다. 이 주제에 대해 자세히 알아보고 싶다면, Deep Learning textbook의 Chapter 9를 보자.

##1.1 Discrete convolutions
Neural networks의 가장 기본적인 개념은 아핀 변환(y = Ax + b, input x와 output y는 벡터이고 A는 행렬, 바이어스 b는 역시 벡터)이다. 이건 input이 무엇임에도 상관없이 응용가능하다. 이미지, 사운드 클립 등,그들이 몇차원임에도 상관없이 벡터로 변환하여 1차원화시킬 수 있기 때문이다. 이미지나 사운드 클립과 많은 비슷한 형태의 자료는 고유의 구조가 있다. 대표적으로, 그들은 다음과 같은 중요한 속성들을 공유한다:
* 요것들은 다차원 배열로 저장이 가능하다.
* 요것들은 정렬가능한 하나 이상의 축(*기준)이 있다. (이미지의 가로, 세로축이라던가 사운드 클립의 시간 축)
* 채널 축(channel axis)라 불리는 축은 데이터가 다양한 값을 가질 수 있게 한다. (컬러 이미지의 빨강 초록 파랑 채널이라던가 오디오의 좌우 음향 채널)

이들 속성은 아핀 변환이 적용될 때는 사용되지 않는다. 사실, 모든 축들은 똑같은 방식으로 처리되는데, 여기서 위상 정보는 고려되지 않는다. 여전히, 자료를 함축적인 구조로 만드는 것으로 얻는 이득으로는 컴퓨터 비젼이나 음성 인식과 같은 작업들을 매우 편하게 만들어주고 이러한 상황에서 이것을 저장(?)하기엔 최고일 수 있다. Discrete convolution은 순서를 바꾸지 않는 선형 변환이다. 이것은 오직 몇개의 input unit만이 output unit에 영향을 주고, 매개변수를 재사용한다 (같은 가중치가 여러개의 input에 영향을 준다).

>(그림)
>Figure 1.1: discrete convolution에서의 output 값을 게산.

>(그림)
>Figure 1.2: N=2, i1=i2=5, k1=k2=3, s1=s2=2, p1=p2=1 일 때 discrete convolution의 output 값을 계산.

Figure 1.1 은 discrete convolution에 관한 예이다. 하늘색의 그리드는 input feature map이라 부른다. 그림을 간단히 하기 위해 오직 한개의 input feature map을 주지만 한개의 feature map위에 다른 feature map 여러개가 쌓이는건 이상한 일이 아니다. 값의 Kernel (어두운 구역)은 input feature map을 다 지나간다. 각 위치에서, kernel의 각 원소와 input 원소의 곱들이 계산되고 이들의 합이 현재 위치의 output이 된다. 이 절차는 원하는 만큼 다양한 결과를 내기 위해 다른 kernel을 사용하여 반복될 수 있다 (Figure 1.3). 이 절차의 최종 output은 output feature maps이라 불린다. 만약 input feature map이 한개가 아니었다면, kernel은 3차원이 되거나 distinct kernel과 각 feature map이 엮여서 결과가 되는 feature map들은 output feature map을 만들기 위해 더해져야 할 것이다 (둘은 실질적으로 같음).
Figure 1.1에서 설명했던 convolution은 2차원 convolution의 인스턴스이지만, 이건 N차원으로 일반화할 수 있다. 예를 들어, 3차원 convolution에서는 kernel은 직육면체가 되어 가로 세로 깊이로 input feature map을 지나간다.
Discrete convolution을 정의하는 kernel의 콜렉션은 순열 (n, m, k1, …, kN) 과 같은 모양을 가지고 있는데 이때
* n = output feature map의 개수
* m = input feature map의 개수
* kj = j축에서의 kernel의 크기

축 j에 따라, 다음 속성들은 convolutional layer output의 크기인 oj에 영향을 미친다:

* ij: 축 j의 input size
* kj: 축 j의 kernel size
* sj: 축 j의 stride (kernel의 두 consecutive 위치의 거리
* pj: 축 j의 zero padding (축의 처음과 끝의 0의 개수)

예를 들어, Figure 1.2는 3*3의 kernel이 5*5의 input을 0으로 된 1*1의 border로  채우고 2*2의 보폭(stride)으로 움직이는 걸 보여준다.
stride가 subsampling의 모양을 구성한다는 것을 기억하자. As an alternative to being interpreted as a measure of how much the kernel is translated, strides can also be viewed as how much of the output is retained. 예를 들어, kernel을 두칸씩 움직이는 것은 kernel을 한칸씩 움직이고 output의 홀수번째 원소만 남기는 것과 같다. (Figure 1.4)

>(그림)
>Figure 1.3: kernel w의 3*2*3*3 콜렉션을 사용하여 두개의 input feature maps에서 세개의 output feature maps으로 나오는 convolution mapping. 왼쪽의 경로에서, input feature map 1은 kernel w1,1과 엮이고 input feature map 2는 kernel w1,2와 엮이고 결과는 요소별로 합쳐져 첫번째 output feature map을 구성한다. 같은 방법으로 가운데와 오른쪽 경로에서 두번째와 세번째 feature maps가 나오고 모든 세개의 feature maps는 함께 묶어서 output을 이룬다.

>(그림)
>Figure 1.4: stirde들을 보는 선택적인 방법. 3*3 kernel을 왼쪽처럼 s=2로 하여 이동하는 대신, 오른쪽처럼 kernel을 1만큼 움직이고 output의 원소들을 s=2로 움직여 둘중 하나만 유지시킨다.

##1.2 Pooling
