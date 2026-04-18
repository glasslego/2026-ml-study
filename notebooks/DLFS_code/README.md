# 처음 시작하는 딥러닝

이 저장소에는 한빛미디어에서 2020년 8월 발간된 '처음 시작하는 딥러닝'의 예제 코드가 실려 있다.

이 저장소의 애초 목적은 집필 과정에서 작성한 예제 코드를 보관하기 위해서였으나, 이 저장소의 예제 코드를 활용해 학습에 도움이 되기를 바란다.

## 예제 코드의 구성

각 장마다 2개의 노트북이 있다. "Code" 노트북은 해당 장의 예제 코드를 담은 노트북이고, "Math" 노트북은 책에 수록된 수식의 LaTeX 코드가 담겨 있다.


## Google Colab에서 실행하기

이 저장소의 모든 노트북은 Google Colab에서 바로 열 수 있도록 정리했다.
각 노트북 첫 부분에 `Open in Colab` 배지와 **공통 실행 환경 준비** 셀이 추가돼 있다.

실행 순서는 다음과 같다.

1. 원하는 노트북을 Colab에서 연다.
2. 맨 위의 **공통 실행 환경 준비** 셀을 먼저 실행한다.
3. 이후 셀은 위에서 아래로 순서대로 실행한다.

공통 실행 환경 준비 셀은 아래 작업을 자동으로 처리한다.

- Colab 런타임에 이 저장소를 `/content/DLFS_code`로 clone
- 현재 노트북이 속한 폴더로 작업 디렉토리 이동
- `lincoln` 패키지가 필요한 장에서 import 경로 자동 연결
- matplotlib 한글 폰트 자동 설정

## 주요 노트북 Colab 링크

- [01_foundations/Code.ipynb](https://colab.research.google.com/github/glasslego/2026-ml-study/blob/main/notebooks/DLFS_code/01_foundations/Code.ipynb)
- [02_fundamentals/Code.ipynb](https://colab.research.google.com/github/glasslego/2026-ml-study/blob/main/notebooks/DLFS_code/02_fundamentals/Code.ipynb)
- [03_dlfs/Code.ipynb](https://colab.research.google.com/github/glasslego/2026-ml-study/blob/main/notebooks/DLFS_code/03_dlfs/Code.ipynb)
- [04_extensions/Code.ipynb](https://colab.research.google.com/github/glasslego/2026-ml-study/blob/main/notebooks/DLFS_code/04_extensions/Code.ipynb)
- [05_convolutions/Code.ipynb](https://colab.research.google.com/github/glasslego/2026-ml-study/blob/main/notebooks/DLFS_code/05_convolutions/Code.ipynb)
- [05_convolutions/Numpy_Convolution_Demos.ipynb](https://colab.research.google.com/github/glasslego/2026-ml-study/blob/main/notebooks/DLFS_code/05_convolutions/Numpy_Convolution_Demos.ipynb)
- [06_rnns/RNN_DLFS.ipynb](https://colab.research.google.com/github/glasslego/2026-ml-study/blob/main/notebooks/DLFS_code/06_rnns/RNN_DLFS.ipynb)
- [07_PyTorch/Code.ipynb](https://colab.research.google.com/github/glasslego/2026-ml-study/blob/main/notebooks/DLFS_code/07_PyTorch/Code.ipynb)

### `lincoln` 패키지

4장, 5장, 7장에 해당하는 폴더에서는 `lincoln` 패키지의 클래스가 사용된다.
이전에는 사용자가 직접 `PYTHONPATH`를 수정해야 했지만, 이제는 각 노트북의 공통 실행 환경 준비 셀이 해당 경로 설정을 자동으로 처리한다.
따라서 로컬 Jupyter나 Colab 모두에서 별도의 셸 설정 없이 실습할 수 있다.

### 5장: 넘파이로 구현한 합성곱 신경망 데모

본문에는 지면 관계상 다 싣지 못했으나 넘파이로 구현한 배치 학습 및 다채널 합성곱 연산 전체 코드를 제공한다(구현 내용과 일부 코드가 부록에 실려 있다) 이 [노트북](05_convolutions/Numpy_Convolution_Demos.ipynb)에서 단일 층으로 구성된 합성곱 신경망을 넘파이만으로 직접 구현해 MNIST 데이터셋에 대해 90% 이상의 정확도를 얻는 것을 확인할 수 있다.
