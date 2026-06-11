# ChangeVG / RSCD

```text
1. Dual_Branch
   SegFormer-B1 기반 vision-guided module입니다.
   이미지 A/B에서 변화 마스크와 coarse caption을 예측합니다.

2. Qwen/VLM adapter
   Dual_Branch가 만든 visual prior를 VLM prompt에 넣어 ChangeVG 추론/튜닝에 사용합니다.
```

## Repository Layout

```text
RSCD/
├── README.md
├── qwen2vl_lora_sft.yaml
├── finetune_from_yaml.py
├── infer/
├── data/
│   └── coding/
│       ├── datasets/
│       │   ├── LEVIR-MCI-dataset/
│       │   └── LEVIR-MCI-dataset-fft/
│       └── muti_task_data/
└── Dual_Branch/
    ├── train.py
    ├── test.py
    ├── train_2.py
    ├── test2.py
    ├── checkpoint_loader.py
    ├── download_segformer.py
    ├── make_fft_dataset.py
    ├── changevg_qwen_infer.py
    ├── qwen_lora_single_image_infer.py
    ├── preprocess_data.py
    ├── requirement.txt
    ├── data/
    ├── model/
    ├── utils_tool/
    └── weights/
```

## Dual_Branch Structure

```text
Dual_Branch/
├── train.py                  # RGB Dual_Branch 학습
├── test.py                   # RGB test split 추론/평가
├── train_2.py                # RGB + FFT branch 학습
├── test2.py                  # RGB + FFT branch 추론/평가
├── checkpoint_loader.py      # .pth / .safetensors 공용 checkpoint loader
├── download_segformer.py     # nvidia/mit-b1 다운로드 및 key 변환
├── make_fft_dataset.py       # FFT-suppressed 데이터셋 생성
├── changevg_qwen_infer.py    # Dual_Branch visual prior + Qwen 추론
├── qwen_lora_single_image_infer.py  # 단일 이미지 + Qwen LoRA adapter 추론
├── preprocess_data.py        # caption token/vocab 전처리
├── requirement.txt
├── data/
│   ├── LEVIR_MCI.py          # PyTorch Dataset
│   └── LEVIR_MCI/
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│       ├── vocab.json
│       └── tokens.zip
├── model/
│   ├── model_encoder_att.py  # Encoder + AttentiveEncoder
│   ├── model_decoder.py      # Transformer caption decoder
│   └── segformer.py          # SegFormer / MiT backbone
├── utils_tool/
│   ├── metrics.py
│   └── utils.py
└── weights/
    ├── Dual_Branch/
    │   ├── Dual_Branch.config.json
    │   └── Dual_Branch.safetensors
    ├── Dual_Branch_FFT/
    │   ├── Dual_Branch_FFT.config.json
    │   └── Dual_Branch_FFT.safetensors
    └── adapter/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

## Data Layout

현재 이미지 데이터셋은 아래 구조를 기대합니다.

```text
data/coding/datasets/
├── LEVIR-MCI-dataset/
│   └── images/
│       ├── train/{A,B,label,label_rgb}
│       ├── val/{A,B,label,label_rgb}
│       └── test/{A,B,label,label_rgb}
└── LEVIR-MCI-dataset-fft/
    └── images/
        ├── train/{A,B,label,label_rgb}
        ├── val/{A,B,label,label_rgb}
        └── test/{A,B,label,label_rgb}
```

Caption 학습/평가에 필요한 split, vocab, token 파일은 아래 위치를 사용합니다.

```text
Dual_Branch/data/LEVIR_MCI/
├── train.txt
├── val.txt
├── test.txt
├── vocab.json
└── tokens.zip
```

`train.py`와 `test.py`는 기본적으로 `Dual_Branch/data/LEVIR_MCI/tokens/` 폴더를 찾습니다. 현재 `tokens.zip`만 있다면 먼저 압축을 풉니다.

```bash
cd RSCD/Dual_Branch/data/LEVIR_MCI
unzip tokens.zip
```

## Environment

CUDA 사용 가능한 PyTorch 환경을 권장합니다. 현재 Dual_Branch 코드에는 `.cuda()` 호출이 직접 들어있어서 CPU-only 환경에서는 수정이 필요합니다.

```bash
cd RSCD
pip install -r Dual_Branch/requirement.txt
```

## SegFormer-B1 Pretrained Weight

논문 구조에서 Dual_Branch feature extractor는 shared SegFormer-B1 encoder를 사용합니다. 학습 재현을 하려면 먼저 SegFormer-B1 pretrained weight를 준비하는 것이 좋습니다.

```bash
cd RSCD
python Dual_Branch/download_segformer.py
```

실행 후 아래 파일이 생성됩니다.

```text
Dual_Branch/model/pretrained/mit_b1.pth
```

`model/segformer.py`는 이 파일이 있으면 로드하고, 없으면 경고를 출력한 뒤 random initialization으로 학습을 시작합니다. 이미 학습된 `.pth` checkpoint를 불러 추론만 하는 경우에는 checkpoint의 `encoder_dict`가 encoder 전체를 덮어쓰므로 pretrained 파일이 없어도 동작할 수 있습니다.

## RGB Dual_Branch Training

권장 실행 위치는 `RSCD/` 루트입니다.

```bash
cd RSCD
python -u Dual_Branch/train.py
```

기본 설정:

```text
data_folder = ./data/coding/datasets/LEVIR-MCI-dataset/images
list_path = ./Dual_Branch/data/LEVIR_MCI/
token_folder = ./Dual_Branch/data/LEVIR_MCI/tokens/
network = segformer-mit_b1
train_goal = 2
num_epochs = 250
```

`train_goal=2`일 때 학습 흐름:

```text
1. goal=2: detection + captioning joint training
2. goal=1: captioning branch fine-tuning
3. goal=0: detection branch fine-tuning
```

학습 중 저장되는 checkpoint는 `encoder_dict`, `encoder_trans_dict`, `decoder_dict`를 포함합니다.

## RGB Test / Evaluation

`test.py`는 `.pth` checkpoint와 `Dual_Branch/weights/Dual_Branch/Dual_Branch.safetensors`를 모두 읽을 수 있습니다. 기본값은 `Dual_Branch/weights` 아래 safetensors입니다.

`RSCD/` 루트에서 실행할 때는 경로를 명시하는 편이 안전합니다.

```bash
cd RSCD
python -u Dual_Branch/test.py \
  --data_folder ./data/coding/datasets/LEVIR-MCI-dataset/images \
  --list_path ./Dual_Branch/data/LEVIR_MCI/ \
  --token_folder ./Dual_Branch/data/LEVIR_MCI/tokens/ \
  --save_mask \
  --save_caption
```

주요 출력:

```text
predict_result/
├── dual_prior.json
├── dual_prior.jsonl
├── score.json
├── *_mask.png
├── *_gt.png
└── *_cap.txt
```

평가 지표:

```text
Segmentation: Pixel Accuracy, Class Accuracy, mIoU, FWIoU, class IoU
Captioning: BLEU-1/2/3/4, METEOR, ROUGE_L, CIDEr
```

## FFT Branch

FFT 데이터셋은 `LEVIR-MCI-dataset-fft` 아래에 있습니다. 새로 생성해야 할 때는 다음 스크립트를 사용합니다.

```bash
cd RSCD
python Dual_Branch/make_fft_dataset.py
```

RGB + FFT 모델은 별도 스크립트를 사용합니다.

```bash
cd RSCD
python -u Dual_Branch/train_2.py
python -u Dual_Branch/test2.py
```

일반 `train.py`에도 `--use_fft` 옵션이 있지만, 이 옵션은 RGB 입력 대신 FFT-suppressed 이미지 데이터셋을 사용하도록 바꾸는 용도입니다. RGB branch와 FFT branch를 함께 쓰는 모델은 `train_2.py` / `test2.py` 쪽을 확인하세요.

## Inference Weights

배포용 safetensors weight는 `Dual_Branch/weights/` 아래에 둡니다.

```text
Dual_Branch/weights/
├── Dual_Branch/
│   ├── Dual_Branch.safetensors
│   └── Dual_Branch.config.json
├── Dual_Branch_FFT/
│   ├── Dual_Branch_FFT.safetensors
│   └── Dual_Branch_FFT.config.json
└── adapter/
    ├── adapter_model.safetensors
    └── adapter_config.json
```

`test.py`와 `test2.py`는 `checkpoint_loader.py`를 통해 `.pth`와 `.safetensors`를 모두 처리합니다.

```text
Dual_Branch/test.py
  기본 checkpoint = Dual_Branch/weights/Dual_Branch/Dual_Branch.safetensors

Dual_Branch/test2.py
  기본 checkpoint = Dual_Branch/weights/Dual_Branch_FFT/Dual_Branch_FFT.safetensors
```

safetensors loader는 같은 폴더의 `.config.json`에 들어있는 `checkpoint_groups`를 읽고, 기존 `.pth` checkpoint와 같은 dict 구조로 복원합니다.

```text
Dual_Branch.safetensors
  -> encoder_dict
  -> encoder_trans_dict
  -> decoder_dict

Dual_Branch_FFT.safetensors
  -> encoder_dict
  -> encoder_rgb_dict
  -> encoder_fft_dict
  -> fusion_a_dict
  -> fusion_b_dict
  -> encoder_trans_dict
  -> decoder_dict
```

`adapter/`는 `test.py` / `test2.py`가 직접 사용하는 weight가 아니라 Qwen/VLM adapter 쪽 파일입니다. `changevg_qwen_infer.py --use_lora` 또는 `qwen_lora_single_image_infer.py`에서 base Qwen 위에 얹어 사용합니다.

## Token Usage

Caption token은 학습과 평가에서 쓰입니다.

```text
training:
  decoder.forward(feat1, feat2, token, token_len)
  정답 caption token을 teacher forcing으로 사용

validation/test/inference:
  decoder.sample(feat1, feat2)
  <START> token부터 autoregressive하게 caption 생성
```

즉 실제 이미지 추론에는 외부 `tokens/*.txt`를 모델 입력으로 넣지 않습니다. 다만 BLEU, METEOR, ROUGE, CIDEr 같은 caption 평가를 하려면 reference caption으로 token 파일이 필요합니다.

## Dual_Branch Output

Dual_Branch는 다음 정보를 만듭니다.

```text
1. seg_pre
   변화 segmentation logits

2. pred_mask
   argmax 결과 mask
   class 0 = background
   class 1 = road
   class 2 = building

3. global_caption
   decoder.sample()이 생성한 coarse change caption

4. dual_prior
   caption, road/building count, road/building location
```

이 `dual_prior`는 `changevg_qwen_infer.py`에서 Qwen/VLM prompt에 넣는 visual prior로 사용할 수 있습니다.

## ChangeVG Qwen Inference

`changevg_qwen_infer.py`는 세 가지 축을 argument로 조합해서 실험합니다.

```text
1. Qwen 입력 모드
   --mode qwen_only | guided_text | guided_full

2. Dual_Branch 종류
   --dual-mode rgb | rgb_fft

3. Qwen LoRA 튜닝 여부
   기본 Qwen: --use_lora를 주지 않음
   LoRA Qwen: --use_lora --lora_adapter_path Dual_Branch/weights/adapter
```

모드별 의미는 다음과 같습니다.

```text
qwen_only
  Qwen에 RGB A/B와 원본 ChangeIMTI instruction만 넣습니다.
  Dual_Branch를 사용하지 않으므로 --dual-mode 값은 결과에 영향을 주지 않습니다.

guided_text
  Qwen에 RGB A/B를 넣고, prompt 앞에 Dual_Branch visual prior를 text로 붙입니다.
  visual prior에는 caption, road/building count, road/building location이 들어갑니다.

guided_full
  guided_text와 같지만 Dual_Branch가 예측한 mask 이미지를 세 번째 image로 Qwen에 추가합니다.
```

기본 Qwen + RGB dual prior:

```bash
cd RSCD
python Dual_Branch/changevg_qwen_infer.py \
  --model-path /workspace/.hf_home/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --mode guided_text \
  --dual-mode rgb
```

기본 Qwen + RGB+FFT dual prior:

```bash
python Dual_Branch/changevg_qwen_infer.py \
  --model-path /workspace/.hf_home/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --mode guided_text \
  --dual-mode rgb_fft
```

LoRA Qwen + RGB+FFT dual prior:

```bash
python Dual_Branch/changevg_qwen_infer.py \
  --model-path /workspace/.hf_home/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --mode guided_text \
  --dual-mode rgb_fft \
  --use_lora \
  --lora_adapter_path Dual_Branch/weights/adapter
```

결과 JSONL에는 `mode`, `dual_mode`, `use_lora`, `lora_adapter_path`, `prompt`, `prediction`, `reference`, `dual_prior`가 함께 저장됩니다.

## Qwen2.5-VL LoRA Fine-tuning from YAML

Qwen2.5-VL LoRA/SFT는 LLaMA-Factory YAML 설정으로 실행합니다. 기본 설정 파일은 repo root의 `qwen2vl_lora_sft.yaml`입니다.

```text
qwen2vl_lora_sft.yaml
  model_name_or_path: /data/coding/Qwen2.5-VL-7B-Instruct
  stage: sft
  finetuning_type: lora
  lora_rank: 8
  lora_target: all
  dataset: caption
  eval_dataset: test_data_extra
  output_dir: saves/qwen2_vl-7b/lora/caption
```

실행 코드는 `finetune_from_yaml.py`입니다. 이 스크립트는 먼저 LLaMA-Factory의 `data/dataset_info.json`에 ChangeVG caption dataset을 등록한 뒤, YAML로 학습을 시작합니다.

```bash
cd RSCD
python finetune_from_yaml.py \
  --config qwen2vl_lora_sft.yaml \
  --llama-factory-dir /data/coding/LLaMA-Factory
```

자동 등록되는 dataset 이름은 YAML의 `dataset`, `eval_dataset`과 맞춰져 있습니다.

```text
caption
  -> data/coding/muti_task_data/train_task_data/caption.json

test_data_extra
  -> data/coding/muti_task_data/test_task_data/caption.json
```

등록되는 데이터 포맷은 ShareGPT multimodal 형식입니다.

```text
conversations -> messages
images        -> images
human         -> user
gpt           -> assistant
```

학습 전에 명령과 dataset 등록 내용을 확인하려면 `--dry-run`을 사용합니다.

```bash
python finetune_from_yaml.py \
  --config qwen2vl_lora_sft.yaml \
  --llama-factory-dir /data/coding/LLaMA-Factory \
  --dry-run
```

실제 실행 시 필요한 조건:

```text
1. LLaMA-Factory repository가 --llama-factory-dir 경로에 있어야 합니다.
2. llamafactory-cli가 현재 shell PATH에 있어야 합니다.
3. qwen2vl_lora_sft.yaml의 model_name_or_path가 실제 Qwen2.5-VL-7B-Instruct 경로여야 합니다.
4. caption/test_data_extra JSON 내부 image path가 학습 서버에서 접근 가능해야 합니다.
```

`llamafactory-cli`가 없는 환경에서는 먼저 LLaMA-Factory 환경을 activate한 뒤 실행하세요. `dataset_info.json`을 직접 관리하고 싶으면 `--skip-dataset-info`를 줄 수 있습니다.

## Single Image Qwen LoRA Inference

LoRA adapter가 제대로 붙는지 단일 이미지로 확인하려면 다음 스크립트를 사용합니다.

```bash
cd RSCD
python Dual_Branch/qwen_lora_single_image_infer.py \
  --model-path /workspace/.hf_home/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --image Dual_Branch/test.png \
  --prompt "Describe this image."
```

LoRA adapter는 단독 모델이 아니므로 항상 base Qwen 경로와 adapter 경로가 함께 필요합니다. 기본 adapter 경로는 `Dual_Branch/weights/adapter`입니다.

## Notes

- `download_segformer.py`는 학습 재현 기준으로 보관해야 합니다.
- `tokens.zip`은 압축 해제 후 `tokens/` 폴더가 필요합니다.
- `test.py`와 `test2.py`는 `.pth`와 `.safetensors` checkpoint를 모두 지원합니다.
- 기본 safetensors 위치는 `Dual_Branch/weights/`입니다.
- `qwen2vl_lora_sft.yaml`과 `finetune_from_yaml.py`는 Qwen2.5-VL LoRA fine-tuning 실행용 파일입니다.
- `adapter/`는 학습된 Qwen LoRA adapter를 추론 때 연결하는 위치입니다.
- caption metric 계산 중 `FileNotFoundError: java`가 나면 JRE를 설치해야 합니다.

```bash
apt update
apt install -y default-jre
```
