"""MNIST 원본 파일을 내려받아 NumPy 배열로 준비하는 유틸리티."""

import gzip
import pickle
from urllib import request

import numpy as np

"""
Credit: https://github.com/hsjeong5
"""

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"],
]


def download_mnist():
    """MNIST gzip 원본 파일을 현재 작업 디렉토리로 내려받는다."""

    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        request.urlretrieve(base_url + name[1], name[1])
    print("Download complete.")


def save_mnist():
    """gzip 파일을 읽어 NumPy 배열로 바꾼 뒤 pickle로 저장한다."""

    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], "rb") as f:
            # 이미지 파일은 앞 16바이트가 헤더이므로 건너뛴다.
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(
                -1, 28 * 28
            )
    for name in filename[-2:]:
        with gzip.open(name[1], "rb") as f:
            # 라벨 파일은 헤더 크기가 8바이트다.
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", "wb") as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def init():
    """다운로드와 pickle 저장을 한 번에 수행한다."""

    download_mnist()
    save_mnist()


def load():
    """저장된 MNIST pickle을 불러와 train/test 세트를 반환한다."""

    with open("mnist.pkl", "rb") as f:
        mnist = pickle.load(f)
    return (
        mnist["training_images"],
        mnist["training_labels"],
        mnist["test_images"],
        mnist["test_labels"],
    )
