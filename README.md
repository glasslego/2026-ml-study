# 2026 ML Study

머신러닝 학습 프로젝트 - Google Colab 실습

## 환경 설정

### 로컬 (uv)

```bash
uv sync
source .venv/bin/activate
jupyter notebook
```

### Google Colab

각 노트북 상단의 **Open in Colab** 배지를 클릭하면 바로 실행할 수 있습니다.

## 프로젝트 구조

```
2026-ml-study/
├── notebooks/          # Jupyter 실습 노트북 (Colab 호환)
├── src/                # Python 모듈
│   └── utils/          # 유틸리티 함수
├── data/               # 데이터 파일 (git 미추적)
├── docs/               # 학습 노트/참고 자료
├── pyproject.toml      # 프로젝트 설정 (uv)
└── README.md
```

## 노트북 목록

| # | 주제 | Colab |
|---|------|-------|
| 00 | 환경 설정 테스트 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glasslego/2026-ml-study/blob/main/notebooks/00_setup_test.ipynb) |
| 08 | RNN→LSTM→Attention→BERT 감정분류 비교 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glasslego/2026-ml-study/blob/main/notebooks/ch08_rnn_lstm_attention_bert.ipynb) |
| -- | Colab Git 연동 설정 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glasslego/2026-ml-study/blob/main/notebooks/colab_git_setup.ipynb) |
