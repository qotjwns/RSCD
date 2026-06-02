# ChangeVG

원격탐사 변화 이해(Remote Sensing Change Understanding)를 위한 프로젝트입니다. 현재 이 저장소에서 핵심적으로 사용하는 코드는 `Dual_Branch`이며, 두 시점의 원격탐사 이미지 A/B를 입력으로 받아 변화 마스크와 변화 설명 문장을 함께 예측합니다.

## 파일 구조

```text
ChangeVG/
├── README.md
├── data/
│   └── coding/
│       ├── datasets/
│       │   ├── LEVIR-MCI-dataset/
│       │   │   └── images/            # LEVIR-MCI 이미지 데이터
│       │   │       ├── train/
│       │   │       ├── val/
│       │   │       └── test/
│       │   └── LEVIR-MCI-dataset-fft/
│       │       └── images/            # 저주파 suppress 복원 이미지 데이터
│       └── muti_task_data/            # ChangeIMTI 계열 task json
├── models_ckpt/
│   └── baseline_.../                  # train.py 기본 checkpoint 저장 위치
└── Dual_Branch/
    ├── train.py                       # Dual Branch 학습 루프
    ├── test.py                        # 테스트셋 평가 스크립트
    ├── predict.py                     # 단일 이미지 pair 추론 유틸
    ├── download_segformer.py          # SegFormer MiT 가중치 다운로드/변환
    ├── make_fft_dataset.py            # FFT 저주파 suppress 데이터 생성
    ├── preprocess_data.py             # caption token/vocab 전처리
    ├── changevg_qwen_infer.py         # Dual Branch + Qwen 추론 연결
    ├── eval_dual_prior_predictions.py # 예측 결과 후처리/평가
    ├── requirement.txt                # Dual Branch 의존성
    ├── data/
    │   ├── LEVIR_MCI.py               # PyTorch Dataset
    │   └── LEVIR_MCI/
    │       ├── train.txt              # train split 목록
    │       ├── val.txt                # validation split 목록
    │       ├── test.txt               # test split 목록
    │       ├── vocab.json             # caption vocabulary
    │       ├── tokens.zip             # caption token 압축본
    │       └── tokens/                # image별 caption token txt
    ├── model/
    │   ├── model_encoder_att.py       # Encoder + AttentiveEncoder
    │   ├── model_decoder.py           # Transformer caption decoder
    │   ├── segformer.py               # SegFormer MiT backbone 정의
    │   └── pretrained/
    │       └── mit_b1.pth             # SegFormer MiT-B1 pretrained weight
    ├── utils_tool/
    │   ├── utils.py                   # loss/accuracy/eval score 유틸
    │   └── metrics.py                 # segmentation metric 계산
    └── models_ckpt/
        └── Dual_Branch.pth            # 제공된 추론 checkpoint
```

## 모델 입력과 출력

입력은 같은 지역을 서로 다른 시점에 촬영한 이미지 2장입니다.

```text
imgA: 변화 전 이미지, 대략 (B, 3, 256, 256)
imgB: 변화 후 이미지, 대략 (B, 3, 256, 256)
```

출력은 두 가지입니다.

```text
seg_pre: 변화 검출 segmentation logits, 대략 (B, 3, H, W)
caption: 변화 설명 문장 token sequence
```

segmentation class는 다음과 같습니다.

```text
0: background
1: road
2: building
```

## 환경 설정

가상환경이 이미 있다면 활성화합니다.

```bash
source venv/bin/activate
```

필요한 패키지는 `Dual_Branch/requirement.txt` 기준으로 설치합니다.

```bash
pip install -r Dual_Branch/requirement.txt
```

주의할 점은 현재 코드가 `.cuda()`를 직접 호출한다는 것입니다. 따라서 기본 학습/테스트는 CUDA 사용 가능한 PyTorch 환경을 전제로 합니다. CPU-only 환경에서 실행하려면 `.cuda()` 호출부를 device 기반 코드로 바꿔야 합니다.

## 데이터 구조

이미지 데이터셋은 아래 구조를 기대합니다.

```text
LEVIR-MCI-dataset/
└── images/
    ├── train/
    │   ├── A/                         # 변화 전 이미지
    │   ├── B/                         # 변화 후 이미지
    │   └── label/                     # 변화 mask label
    ├── val/
    │   ├── A/
    │   ├── B/
    │   └── label/
    └── test/
        ├── A/
        ├── B/
        └── label/
```

caption token과 vocab 파일은 아래 위치에 있어야 합니다.

```text
Dual_Branch/data/LEVIR_MCI/
├── vocab.json                         # caption vocabulary
├── train.txt                          # train image list
├── val.txt                            # validation image list
├── test.txt                           # test image list
└── tokens/                            # image별 caption token
```

`tokens/` 폴더가 없고 `tokens.zip`만 있다면 압축을 먼저 풉니다.

```bash
cd Dual_Branch/data/LEVIR_MCI
unzip tokens.zip
```

## FFT 데이터 생성

`make_fft_dataset.py`는 `Dual_Branch/data/LEVIR_MCI/{train,val,test}.txt` 목록을 기준으로 원본 A/B 이미지를 읽고, FFT 영역에서 중심 저주파 영역을 suppress한 뒤 inverse FFT로 복원한 이미지를 새 데이터셋으로 저장합니다. `label`, `label_rgb`는 변환하지 않고 그대로 복사합니다.

```text
입력:
data/coding/datasets/LEVIR-MCI-dataset/
└── images/

출력:
data/coding/datasets/LEVIR-MCI-dataset-fft/
└── images/
    ├── train/
    │   ├── A/
    │   ├── B/
    │   ├── label/
    │   └── label_rgb/
    ├── val/
    └── test/
```

실행 명령:

```bash
./venv/bin/python Dual_Branch/make_fft_dataset.py
```

저주파 suppress 설정은 [Dual_Branch/make_fft_dataset.py](/Users/baeseojun/ChangeVG/Dual_Branch/make_fft_dataset.py:19) 상단 전역변수로 조정합니다.

```python
LOW_FREQ_SUPPRESS_RADIUS_RATIO = 0.08
LOW_FREQ_SUPPRESS_STRENGTH = 0.7
```

의미:

```text
LOW_FREQ_SUPPRESS_RADIUS_RATIO
  이미지 짧은 변 기준 저주파 영역 반지름 비율입니다.
  예: 256x256 이미지에서 0.08이면 약 20px 반지름입니다.

LOW_FREQ_SUPPRESS_STRENGTH
  1.0 = 저주파 완전 제거
  0.5 = 절반 약화
  0.0 = 원본 유지
```

## SegFormer 가중치

`download_segformer.py`는 Hugging Face의 `nvidia/mit-b1` weight를 다운로드한 뒤, 현재 `segformer.py`가 읽을 수 있는 key 형식으로 변환해서 아래 위치에 저장합니다.

```text
Dual_Branch/model/pretrained/
└── mit_b1.pth                         # segformer.py가 로드하는 최종 weight
```

실행 명령:

```bash
./venv/bin/python Dual_Branch/download_segformer.py
```

## 학습 실행

현재 `train.py`의 기본 경로는 저장소 루트 `ChangeVG`에서 실행하는 기준으로 맞춰져 있습니다.

RGB 데이터로 학습:

```bash
./venv/bin/python -u Dual_Branch/train.py
```

FFT 데이터로 학습:

```bash
./venv/bin/python -u Dual_Branch/train.py --use_fft
```

기본 학습 설정은 다음과 같습니다.

```text
train_goal = 2
train_stage = s1
network = segformer-mit_b1
train_batchsize = 64
num_epochs = 250
```

`--use_fft`를 사용하면 학습 데이터 경로가 자동으로 아래 위치로 바뀝니다.

```text
data/coding/datasets/LEVIR-MCI-dataset-fft/images
```

`train_goal=2`일 때 학습 순서는 다음과 같습니다.

```text
1. goal=2: 변화 검출 + captioning 공동 학습
2. goal=1: captioning branch fine-tuning
3. goal=0: detection branch fine-tuning
```

학습 로그에는 raw loss와 normalized loss가 함께 출력됩니다.

```text
Raw Det_Loss
Norm Det_Loss
Raw Cap_loss
Norm Cap_loss
```

공동 학습 단계에서는 normalized loss가 optimization에 사용되고, raw loss는 실제 loss 추세 확인을 위해 로그에 남깁니다.

저장되는 checkpoint 이름에는 데이터 종류가 들어갑니다.

```text
LEVIR_MCI_RGB_bts_...
LEVIR_MCI_FFT_bts_...
```

best checkpoint alias도 분리됩니다.

```text
models_ckpt/
└── baseline_.../
    └── baseline_.../
        ├── Dual_Branch.pth            # RGB 학습 best alias
        └── Dual_Branch_FFT.pth        # FFT 학습 best alias
```

## 테스트 실행

테스트는 기본적으로 `Dual_Branch/test.py`를 사용합니다. 현재 `test.py`는 `--use_fft` 인자가 없으므로, FFT 데이터로 테스트하려면 `--data_folder`를 직접 지정합니다.

```bash
./venv/bin/python -u Dual_Branch/test.py \
  --data_folder data/coding/datasets/LEVIR-MCI-dataset/images \
  --list_path Dual_Branch/data/LEVIR_MCI/ \
  --token_folder Dual_Branch/data/LEVIR_MCI/tokens/ \
  --checkpoint models_ckpt/<run_dir>/<run_dir>/Dual_Branch.pth
```

FFT 데이터 테스트:

```bash
./venv/bin/python -u Dual_Branch/test.py \
  --data_folder data/coding/datasets/LEVIR-MCI-dataset-fft/images \
  --list_path Dual_Branch/data/LEVIR_MCI/ \
  --token_folder Dual_Branch/data/LEVIR_MCI/tokens/ \
  --checkpoint models_ckpt/<run_dir>/<run_dir>/Dual_Branch_FFT.pth
```

결과는 기본적으로 아래 위치에 저장됩니다.

```text
Dual_Branch/
└── predict_result/                    # 저장된 mask/caption 예측 결과
```

## 주요 평가 지표

변화 검출 성능:

```text
Pixel Accuracy
Class Accuracy
mIoU
FWIoU
class별 IoU
```

captioning 성능:

```text
BLEU-1
BLEU-2
BLEU-3
BLEU-4
METEOR
ROUGE_L
CIDEr
```
