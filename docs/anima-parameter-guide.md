# Anima LoRA 학습 파라미터 가이드

> `anima_lora_train_colab.ipynb`의 `@param` 폼에서 조절하는 모든 파라미터를  
> 모델 아키텍처·훈련 원리와 연결하여 설명하는 문서입니다.
>
> - 아키텍처 전체 구조 → [anima-architecture.md](anima-architecture.md)  
> - 딥러닝 기법 해설 → [anima-deep-learning-notes.md](anima-deep-learning-notes.md)

---

## 목차

1. [파라미터 전체 맵](#1-파라미터-전체-맵)
2. [데이터셋 파라미터 — 셀 4-1](#2-데이터셋-파라미터--셀-4-1)
   - 2.1 [RESOLUTION](#21-resolution)
   - 2.2 [NUM_REPEATS](#22-num_repeats)
   - 2.3 [CACHE_SHUFFLE_NUM](#23-cache_shuffle_num)
   - 2.4 [MIN_AR / MAX_AR / NUM_AR_BUCKETS](#24-min_ar--max_ar--num_ar_buckets)
3. [학습 핵심 파라미터 — 셀 4-2](#3-학습-핵심-파라미터--셀-4-2)
   - 3.1 [EPOCHS](#31-epochs)
   - 3.2 [LORA_RANK](#32-lora_rank)
   - 3.3 [BATCH_SIZE / GRAD_ACCUM](#33-batch_size--grad_accum)
   - 3.4 [BLOCKS_TO_SWAP](#34-blocks_to_swap)
   - 3.5 [WARMUP_STEPS](#35-warmup_steps)
   - 3.6 [GRADIENT_CLIPPING](#36-gradient_clipping)
   - 3.7 [TIMESTEP_SAMPLE](#37-timestep_sample)
   - 3.8 [LLM_ADAPTER_LR](#38-llm_adapter_lr)
4. [옵티마이저 파라미터 — 셀 4-2](#4-옵티마이저-파라미터--셀-4-2)
   - 4.1 [OPTIMIZER](#41-optimizer)
   - 4.2 [OPTIMIZER_LR](#42-optimizer_lr)
   - 4.3 [WEIGHT_DECAY](#43-weight_decay)
5. [저장 · 체크포인트 파라미터 — 셀 4-2](#5-저장--체크포인트-파라미터--셀-4-2)
   - 5.1 [SAVE_EVERY_N_EPOCHS](#51-save_every_n_epochs)
   - 5.2 [CHECKPOINT_MINUTES](#52-checkpoint_minutes)
6. [고정 설정 (변경 비권장)](#6-고정-설정-변경-비권장)
7. [파라미터 조합 레시피](#7-파라미터-조합-레시피)

---

## 1. 파라미터 전체 맵

노트북에서 조절 가능한 파라미터가 어떤 TOML에 기록되고, 모델의 어느 부분에 영향을 주는지 한눈에 보여줍니다.

```
anima_lora_train_colab.ipynb
│
├── 셀 4-1: dataset.toml 생성
│   ├── RESOLUTION ─────────→ [VAE] latent 크기 결정 (H/8 × W/8)
│   │                          [PatchEmbed] 패치 수 = 시퀀스 길이
│   │                          [3D RoPE] 위치 임베딩 그리드 크기
│   │
│   ├── NUM_REPEATS ────────→ [DataLoader] 한 epoch당 학습 횟수
│   ├── CACHE_SHUFFLE_NUM ──→ [텍스트 인코더] 캐시 태그 셔플 변형 수
│   └── MIN/MAX_AR, BUCKETS → [DataLoader] AR 버킷 분할 방식
│
├── 셀 4-2: training.toml 생성
│   ├── EPOCHS ─────────────→ [학습 루프] 전체 반복 횟수
│   ├── LORA_RANK ──────────→ [LoRA] 저랭크 행렬의 차원 r
│   ├── BATCH_SIZE ─────────→ [DataLoader] 미니배치 크기
│   ├── GRAD_ACCUM ─────────→ [DeepSpeed] 가상 배치 크기 확대
│   ├── BLOCKS_TO_SWAP ─────→ [Block Swap] CPU 오프로드 블록 수
│   ├── WARMUP_STEPS ───────→ [Optimizer] LR 웜업 스텝 수
│   ├── GRADIENT_CLIPPING ──→ [학습 안정성] gradient norm 클리핑
│   ├── TIMESTEP_SAMPLE ────→ [Flow Matching] 시간 t의 분포
│   ├── LLM_ADAPTER_LR ────→ [LLM Adapter] 별도 learning rate
│   │
│   ├── OPTIMIZER ──────────→ [Optimizer] 종류 선택
│   ├── OPTIMIZER_LR ───────→ [Optimizer] 학습률
│   ├── WEIGHT_DECAY ───────→ [Optimizer] 가중치 감쇠
│   │
│   ├── SAVE_EVERY_N_EPOCHS → [저장] LoRA 체크포인트 주기
│   └── CHECKPOINT_MINUTES ─→ [저장] DeepSpeed 체크포인트 주기
│
└── 고정 (코드 내 하드코딩)
    ├── dtype = bfloat16 ───→ [Mixed Precision] 연산 정밀도
    ├── pipeline_stages = 1 → [DeepSpeed] 파이프라인 병렬화
    ├── activation_checkpointing = true → [메모리] gradient checkpoint
    ├── save_dtype = bfloat16 → [저장] LoRA 가중치 정밀도
    └── partition_method = parameters → [DeepSpeed] 파이프라인 분할 방식
```

---

## 2. 데이터셋 파라미터 — 셀 4-1

이 파라미터들은 `anima_dataset.toml`에 기록되며, **학습 데이터가 모델에 입력되기 전**의 전처리 과정을 제어합니다.

### 2.1 RESOLUTION

```python
RESOLUTION = 512  #@param [512, 768, 1024] {type:"raw"}
```

**TOML:** `resolutions = [512]`

**의미:** 학습 이미지의 **최대 한 변 길이** (픽셀). 이미지가 이 해상도에 맞게 리사이즈됩니다.

**아키텍처와의 연결:**

```
입력 이미지 (512×512)
    │
    ▼ VAE Encoder (8× 다운샘플)
latent (64×64, 16채널)
    │
    ▼ PatchEmbed (2×2 패치)
시퀀스 (32×32 = 1024 토큰, 2048차원)
    │
    ▼ 3D RoPE (32×32 그리드)
위치 임베딩 적용
    │
    ▼ 28 × Block (Self-Attention)
Attention 연산량 = O(1024²) = O(~1M)
```

해상도에 따른 변화:

| 해상도 | latent 크기 | 패치 후 시퀀스 길이 | Attention 비용 | VRAM 사용량 |
|---|---|---|---|---|
| **512** | 64×64 | 1,024 | 1× (기준) | ~10 GB |
| **768** | 96×96 | 2,304 | ~5× | ~16 GB |
| **1024** | 128×128 | 4,096 | ~16× | ~28 GB |

> ⚠️ Attention은 시퀀스 길이의 **제곱**에 비례하므로, 해상도를 2배 올리면 VRAM은 약 4배 증가합니다. Colab T4(16GB)에서는 **512가 안전**, 768은 `blocks_to_swap`을 크게 올려야 합니다.

**3D RoPE와의 관계:** 해상도가 올라가면 RoPE 그리드가 커지고, 학습 시 본 적 없는 위치에서 추론하려면 NTK-Aware Scaling이 필요합니다. 학습 해상도 = 추론 해상도가 이상적입니다.

---

### 2.2 NUM_REPEATS

```python
NUM_REPEATS = 2  #@param {type:"integer"}
```

**TOML:** `num_repeats = 2`

**의미:** 데이터셋의 각 이미지를 한 epoch 내에서 **몇 번 반복** 사용할지 결정합니다.

**학습 루프와의 연결:**

```
데이터셋: 이미지 50장, num_repeats=2
  → 한 epoch의 유효 이미지 수 = 50 × 2 = 100장
  → batch_size=10이면, 1 epoch = 10 스텝

데이터셋: 이미지 50장, num_repeats=4
  → 한 epoch의 유효 이미지 수 = 50 × 4 = 200장
  → batch_size=10이면, 1 epoch = 20 스텝
```

**왜 필요한가?** 소량 데이터(10~50장)로 LoRA를 학습할 때, 한 epoch이 너무 짧으면 optimizer가 의미 있는 업데이트를 하기 전에 epoch이 끝나버립니다. `num_repeats`로 한 epoch을 인위적으로 늘릴 수 있습니다.

> 💡 **주의:** `num_repeats`를 너무 높이면 한 epoch에서 같은 이미지를 반복 학습하므로 과적합 위험이 있습니다. `num_repeats × epochs`의 총 학습 횟수를 기준으로 판단하세요.

---

### 2.3 CACHE_SHUFFLE_NUM

```python
CACHE_SHUFFLE_NUM = 1  #@param {type:"integer"}
```

**TOML:** `cache_shuffle_num = 1`

**의미:** 텍스트 캡션의 **태그 순서를 셔플한 변형 버전** 수. `cache_shuffle_delimiter`(기본 `', '`)를 기준으로 태그를 분리한 후 순서를 무작위로 바꿉니다.

**텍스트 인코딩과의 연결:**

```
원본 캡션: "1girl, solo, long hair, blue eyes, school uniform"

cache_shuffle_num = 1 (셔플 안 함):
  → 항상 같은 순서로 인코딩
  → Qwen3 hidden states가 동일

cache_shuffle_num = 3 (3개 변형 생성):
  → 변형 1: "solo, blue eyes, 1girl, school uniform, long hair"
  → 변형 2: "long hair, school uniform, solo, blue eyes, 1girl"
  → 변형 3: "school uniform, 1girl, long hair, solo, blue eyes"
  → 각 변형이 다른 Qwen3 embedding을 생성 → 증강 효과
```

**왜 중요한가?** Qwen3-0.6B는 **autoregressive** 모델이라 토큰 순서에 민감합니다. "1girl, solo" vs "solo, 1girl"은 다른 hidden state를 생성합니다. 태그 순서를 셔플하면:

1. 모델이 **태그 순서에 과적합**하는 것을 방지
2. 소량 데이터에서 **텍스트 임베딩 증강** 효과
3. 추론 시 사용자가 어떤 순서로 태그를 입력해도 안정적으로 동작

> 💡 **주의:** 셔플 변형은 사전 캐싱 시점에 생성되므로, 값이 클수록 캐시 용량이 비례 증가합니다. 이미지 50장, `cache_shuffle_num=5`면 캐시 250건.

---

### 2.4 MIN_AR / MAX_AR / NUM_AR_BUCKETS

```python
MIN_AR = 0.5    #@param {type:"number"}
MAX_AR = 2.0    #@param {type:"number"}
NUM_AR_BUCKETS = 7  #@param {type:"integer"}
```

**TOML:**
```toml
enable_ar_bucket = true
min_ar = 0.5
max_ar = 2.0
num_ar_buckets = 7
```

**의미:** 다양한 종횡비(Aspect Ratio)의 이미지를 효율적으로 배치하기 위한 **AR 버킷** 설정.

**AR 버킷이 필요한 이유:**

```
배치 텐서는 같은 크기여야 한다!

❌ 패딩 방식 (낭비):
  512×512 + 512×768 → 둘 다 512×768 텐서, 빈 공간은 0 패딩
  → 패딩 영역도 Attention에 포함되어 연산량 낭비

✅ AR 버킷 방식 (효율적):
  비슷한 종횡비의 이미지끼리 그룹화
  각 그룹은 같은 해상도로 리사이즈
  → 패딩 없이 배치 구성 가능
```

**버킷 분할 예시 (RESOLUTION=512, NUM_AR_BUCKETS=7):**

```
AR 범위: 0.5 ~ 2.0

총 픽셀 수 ≈ 512 × 512 = 262,144로 고정

버킷 1: AR=0.50 → 362×724 (세로로 긴 이미지)
버킷 2: AR=0.67 → 418×626
버킷 3: AR=0.80 → 456×570
버킷 4: AR=1.00 → 512×512 (정사각형)
버킷 5: AR=1.25 → 570×456
버킷 6: AR=1.50 → 626×418
버킷 7: AR=2.00 → 724×362 (가로로 긴 이미지)
```

**PatchEmbed·RoPE와의 관계:** 각 AR 버킷마다 VAE 후 latent 크기가 다르고, PatchEmbed 후 시퀀스 길이도 다르지만, 총 픽셀 수가 비슷하므로 VRAM 사용량은 거의 일정합니다. 3D RoPE는 각 (H, W) 그리드에 맞춰 동적으로 생성됩니다.

> 💡 **팁:** 데이터셋이 대부분 정사각형이면 `num_ar_buckets=3` 정도로 줄여도 됩니다. 반대로 세로·가로 이미지가 혼합되어 있으면 7~9가 적절합니다.

---

## 3. 학습 핵심 파라미터 — 셀 4-2

이 파라미터들은 `anima_training.toml`에 기록되며, **모델 학습의 핵심 동작**을 제어합니다.

### 3.1 EPOCHS

```python
EPOCHS = 100  #@param {type:"integer"}
```

**TOML:** `epochs = 100`

**의미:** 전체 데이터셋을 몇 번 반복 학습할지.

**실제 학습량 계산:**

```
총 학습 스텝 = (이미지 수 × num_repeats × epochs) / (batch_size × grad_accum)

예: 이미지 50장, repeats=2, epochs=100, batch=10, accum=1
  → 총 스텝 = (50 × 2 × 100) / (10 × 1) = 1,000 스텝

예: 이미지 20장, repeats=4, epochs=50, batch=2, accum=2
  → 총 스텝 = (20 × 4 × 50) / (2 × 2) = 1,000 스텝
```

**과적합과의 관계:** LoRA는 파라미터가 적으므로 (rank=32 기준 ~1.3M) 수백 epoch에서도 과적합 속도가 느리지만, 소량 데이터(10~20장)에서는 50~100 epoch에서도 과적합이 시작될 수 있습니다. TensorBoard에서 loss가 더 이상 의미 있게 감소하지 않으면 과적합을 의심하세요.

> 💡 **실전 팁:** 처음에는 epochs=100으로 시작하고, `save_every_n_epochs=1`로 모든 epoch를 저장한 다음, ComfyUI에서 각 epoch의 LoRA를 비교하여 최적 epoch를 찾는 것이 일반적입니다.

---

### 3.2 LORA_RANK

```python
LORA_RANK = 32  #@param {type:"integer"}
```

**TOML:** `rank = 32` (under `[adapter]`)

**의미:** LoRA 저랭크 분해의 차원 $r$. 원본 가중치 행렬 $W \in \mathbb{R}^{d_{out} \times d_{in}}$ 을 $B \in \mathbb{R}^{d_{out} \times r}$, $A \in \mathbb{R}^{r \times d_{in}}$ 으로 분해합니다.

**Rank에 따른 파라미터 수 변화:**

Anima의 attention 레이어: $d_{in} = d_{out} = 2048$

| rank | LoRA 파라미터/레이어 | 전체 (28블록, 6타겟) | 원본 대비 비율 |
|---|---|---|---|
| **8** | 32,768 | ~5.5M | ~0.27% |
| **16** | 65,536 | ~11M | ~0.54% |
| **32** | 131,072 | ~22M | ~1.1% |
| **64** | 262,144 | ~44M | ~2.2% |
| **128** | 524,288 | ~88M | ~4.4% |

**표현력 vs 과적합:**

```
rank 작음 (8~16):
  + 과적합 위험 낮음
  + 파일 크기 작음 (~20-40 MB)
  + VRAM 절약
  - 복잡한 스타일/캐릭터 재현 어려움
  - 수정 가능한 "용량"이 부족

rank 큼 (64~128):
  + 세밀한 스타일/디테일 표현 가능
  - 과적합 빠름 (특히 소량 데이터)
  - 파일 크기 큼 (~100-200 MB)
  - VRAM 더 많이 사용

rank 32 (기본값):
  균형점 — 대부분의 캐릭터 LoRA에 충분
```

**LoRA 수식과의 연결:**

$$W' = W + \frac{\alpha}{r} \cdot B \cdot A$$

노트북에서는 `alpha = rank`로 설정되므로 (diffusion-pipe 기본값), 스케일링 팩터 $\alpha/r = 1$이 됩니다. 즉 rank를 바꿔도 LoRA 출력의 절대적 스케일은 일정합니다.

---

### 3.3 BATCH_SIZE / GRAD_ACCUM

```python
BATCH_SIZE = 10     #@param [1, 2, 4, 6, 8, 10, 12] {type:"raw"}
GRAD_ACCUM = 1      #@param [1, 2, 4] {type:"raw"}
```

**TOML:**
```toml
micro_batch_size_per_gpu = 10
gradient_accumulation_steps = 1
```

**의미:**
- `BATCH_SIZE`: GPU가 **한 번에 처리하는** 이미지 수
- `GRAD_ACCUM`: gradient를 **누적한 후 한 번에 업데이트**하는 횟수
- **유효 배치 크기** = `BATCH_SIZE × GRAD_ACCUM`

**왜 gradient accumulation이 필요한가?**

```
큰 배치 = 안정적인 gradient 방향 → 안정적인 학습

하지만: batch_size=32 → VRAM 부족!

해결: batch_size=8, grad_accum=4
  → 유효 배치 크기 = 32 (동일한 효과)
  → VRAM은 batch_size=8만큼만 사용

동작 방식:
  스텝 1: batch1의 gradient 계산 (optimizer 업데이트 X)
  스텝 2: batch2의 gradient 계산 → 누적 (업데이트 X)
  스텝 3: batch3의 gradient 계산 → 누적 (업데이트 X)
  스텝 4: batch4의 gradient 계산 → 누적 → optimizer 업데이트! ← 여기서만 가중치 변경
```

**VRAM과의 관계:** `BATCH_SIZE`가 직접적으로 VRAM을 좌우합니다. Anima에서는 latent+text embedding이 **사전 캐싱**되므로 배치 이미지를 실시간으로 VAE에 통과시키지 않습니다. 따라서 `BATCH_SIZE=10`도 T4에서 가능합니다(block swap과 함께).

> 💡 **Prodigy 옵티마이저와의 관계:** Prodigy는 gradient 크기에서 LR을 자동 결정합니다. 유효 배치 크기가 달라지면 gradient의 절대 크기가 변하므로 자동 LR도 달라집니다. 유효 배치 크기를 바꾸면 학습 결과도 달라질 수 있음을 유의하세요.

---

### 3.4 BLOCKS_TO_SWAP

```python
BLOCKS_TO_SWAP = 8  #@param {type:"integer"}
```

**TOML:** `blocks_to_swap = 8`

**의미:** DiT의 28개 블록 중 **CPU RAM에 오프로드**할 블록 수. GPU에 상주하는 블록 수 = 28 - blocks_to_swap.

**Block Swap 동작 원리:**

```
blocks_to_swap = 8 (기본값):

GPU에 상주: 20블록 (Block 0~19)
CPU에 대기:  8블록 (Block 20~27)

Forward 시:
  Block 19 실행 중 → Block 20을 GPU로 비동기 전송 (CUDA Stream)
  Block 20 실행 시작 → Block 0을 CPU로 비동기 전송 (VRAM 확보)
  ...

→ 연산과 전송이 겹치므로(overlap) 속도 저하 최소화
→ 하지만 완벽한 중첩은 불가능 → swap이 많을수록 느려짐
```

**VRAM 절약 효과:**

| blocks_to_swap | GPU 상주 블록 | VRAM 절약 (대략) | 속도 영향 |
|---|---|---|---|
| **0** | 28 | 0 | 가장 빠름 |
| **4** | 24 | ~15% | 거의 무시 가능 |
| **8** | 20 | ~30% | 약간 느림 |
| **14** | 14 | ~50% | 눈에 띄게 느림 |
| **20** | 8 | ~70% | 많이 느림 |
| **26** | 2 (최소) | ~90% | 매우 느림 |

> ⚠️ **LoRA 파라미터는 swap에서 제외됩니다.** Optimizer state가 항상 GPU에서 접근 가능해야 하기 때문입니다. 따라서 swap으로 절약되는 것은 **frozen 원본 가중치**의 VRAM입니다.

**OOM 발생 시 대처:**

```
OOM 발생!
  → blocks_to_swap 먼저 올리기 (8 → 14 → 18)
  → 그래도 부족하면 RESOLUTION 내리기 (768 → 512)
  → batch_size 줄이기 (10 → 4 → 1)
```

---

### 3.5 WARMUP_STEPS

```python
WARMUP_STEPS = 100  #@param {type:"integer"}
```

**TOML:** `warmup_steps = 100`

**의미:** 학습 시작 후 learning rate를 **0에서 목표값까지 서서히 올리는** 스텝 수.

**왜 필요한가?**

```
LR 스케줄 (warmup_steps=100):

LR
 │          ┌──────────────────────────────
 │         ╱
 │        ╱
 │       ╱
 │      ╱
 │     ╱
 │    ╱
 │   ╱
 │  ╱
 │_╱
 └────────────────────────────────────────→ step
  0    50    100   200   300   ...
       ↑ warmup ↑
```

학습 초기에는 LoRA 가중치가 (거의) 0이고, 모델은 거의 원본 상태입니다. 이 상태에서 갑자기 큰 LR로 업데이트하면 gradient가 불안정해질 수 있습니다. 서서히 LR을 올리면:

1. 초기 gradient 방향이 안정화될 때까지 대기
2. Optimizer의 2차 모멘텀(Adam의 $v_t$)이 충분히 추정될 시간 확보
3. LoRA 가중치가 서서히 활성화

**Prodigy와의 관계:** Prodigy는 LR을 자동 조정하므로 warmup이 덜 중요하다고 생각할 수 있지만, Prodigy의 내부 `d` 추정도 초기에 불안정합니다. warmup을 100 정도 주면 `d` 추정이 안정된 후에 본격 학습이 시작됩니다.

> 💡 **팁:** 데이터셋이 매우 소량(10~20장)이면 총 스텝 수도 적으므로 `warmup_steps=20~50`으로 줄이세요.

---

### 3.6 GRADIENT_CLIPPING

```python
GRADIENT_CLIPPING = 1.0  #@param {type:"number"}
```

**TOML:** `gradient_clipping = 1.0`

**의미:** Gradient의 전체 L2 norm이 이 값을 초과하면 **비례 축소**합니다.

**동작 방식:**

```python
# 개념적으로:
total_norm = sqrt(sum(p.grad.norm()**2 for p in model.parameters()))
if total_norm > gradient_clipping:
    scale = gradient_clipping / total_norm
    for p in model.parameters():
        p.grad *= scale
```

```
예: gradient_clipping = 1.0

gradient norm = 0.5 → 그대로 사용 (문제없음)
gradient norm = 3.0 → 모든 gradient를 ×(1/3)으로 축소
gradient norm = 100  → 모든 gradient를 ×(1/100)으로 축소

→ 방향은 유지, 크기만 제한
→ "gradient 폭발" 방지
```

**Flow Matching 학습에서의 중요성:** Rectified Flow에서 $t \approx 1$ (거의 순수 노이즈) 근처의 loss는 매우 클 수 있습니다. 이런 샘플이 배치에 포함되면 gradient가 갑자기 커질 수 있으므로 clipping이 안전장치 역할을 합니다.

> 💡 **조절 가이드:** 기본값 1.0이 대부분 적절합니다. TensorBoard에서 loss가 갑자기 튀는 현상이 자주 보이면 0.5로 낮춰보세요.

---

### 3.7 TIMESTEP_SAMPLE

```python
TIMESTEP_SAMPLE = "logit_normal"  #@param ["logit_normal", "uniform", "sigmoid"] {type:"string"}
```

**TOML:** `timestep_sample_method = 'logit_normal'`

**의미:** Flow Matching에서 학습 시 시간 $t$를 **어떤 분포**에서 샘플링할지 결정합니다.

**각 분포의 특성:**

```
┌─ t의 확률 밀도 ─────────────────────────────┐
│                                              │
│ uniform:      ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬   │
│               t=0 부터 t=1 까지 균등         │
│                                              │
│ logit_normal:         ╱╲                     │
│                      ╱  ╲                    │
│               ─────╱    ╲─────              │
│               t≈0.5 근처에 집중 (shift 적용)  │
│                                              │
│ sigmoid:            ╱──╲                     │
│                   ╱      ╲                   │
│               ──╱          ╲──              │
│               logit_normal과 유사하나         │
│               꼬리가 더 두꺼움                │
└──────────────────────────────────────────────┘
  0.0          0.5          1.0
  (깨끗)       (중간)        (노이즈)
```

**Flow Matching 학습과의 연결:**

$$x_t = (1-t) \cdot x_0 + t \cdot \varepsilon, \quad v = \varepsilon - x_0$$

- **$t \approx 0$** (거의 깨끗): 모델이 미세한 노이즈를 제거하는 법을 학습 → 디테일에 영향
- **$t \approx 0.5$** (중간): 구조와 디테일을 동시에 학습 → **가장 정보량이 많은 구간**
- **$t \approx 1$** (거의 노이즈): 전체 구조를 잡는 법 학습 → 대략적 구도

`logit_normal`은 중간 구간에 집중하여 학습 효율을 높입니다. Anima에서는 추가로 `shift=3.0`을 적용하여 분포를 약간 노이즈 쪽으로 이동시킵니다:

$$t_{\text{shifted}} = \frac{3.0 \cdot t}{1 + 2.0 \cdot t}$$

> 💡 **실전:** 대부분의 경우 `logit_normal`(기본값)이 최적입니다. `uniform`은 이론적으로 단순하지만 학습 효율이 낮습니다.

---

### 3.8 LLM_ADAPTER_LR

```python
LLM_ADAPTER_LR = 0  #@param {type:"number"}
```

**TOML:** `llm_adapter_lr = 0`

**의미:** LLM Adapter (6-layer Transformer) 에 적용할 **별도의 learning rate**. 0이면 LLM Adapter는 학습하지 않습니다 (frozen).

**LLM Adapter의 역할 (아키텍처 복습):**

```
Qwen3-0.6B hidden states (B, S, 1024)
        │
        ▼
┌──────────────────────────┐
│     LLM Adapter          │
│  6-layer Transformer     │
│  + T5 token ID 조회      │
│  → T5 vocab embedding    │
│  → cross_attn_emb 생성   │
└──────────────┬───────────┘
               │
               ▼
DiT Cross-Attention (B, S, 1024)
```

**왜 기본값이 0인가?**

- LLM Adapter는 **사전 학습된 가중치**를 포함하고 있음
- LoRA 학습에서는 DiT의 LoRA만 학습하는 것이 일반적
- Adapter까지 학습하면 텍스트 이해 능력이 변할 수 있어 위험

**0이 아닌 값을 쓰는 경우:**

```
llm_adapter_lr = 0       → DiT LoRA만 학습 (안전, 기본)
llm_adapter_lr = 1e-6    → Adapter도 미세 조정 (고급)
llm_adapter_lr = 1e-5    → Adapter를 적극 학습 (텍스트 이해 변경 가능)
```

diffusion-pipe에서는 `per_component_lr` 기능으로 구현됩니다. LLM Adapter의 파라미터에만 별도의 optimizer group을 생성하여 다른 LR을 적용합니다.

> ⚠️ **주의:** 0이 아닌 값을 사용하면 LoRA가 아닌 **LLM Adapter의 전체 가중치**가 학습됩니다 (full fine-tuning). 저장된 체크포인트 크기도 크게 증가합니다.

---

## 4. 옵티마이저 파라미터 — 셀 4-2

### 4.1 OPTIMIZER

```python
OPTIMIZER = "Prodigy"  #@param ["Prodigy", "AdamW8bitKahan", "adamw_optimi"] {type:"string"}
```

**TOML:** `type = 'Prodigy'` (under `[optimizer]`)

**의미:** 가중치 업데이트에 사용할 optimizer 알고리즘.

**각 옵티마이저 비교:**

| | **Prodigy** | **AdamW8bitKahan** | **adamw_optimi** |
|---|---|---|---|
| **LR 설정** | lr=1 고정 (자동 조정) | 수동 (5e-5~5e-6) | 수동 (2e-5) |
| **메모리** | 보통 | **적음** (8bit 상태) | 보통 |
| **특징** | 학습 전 LR 탐색 불필요 | Kahan 보정으로 bf16 오차 방지 | diffusion-pipe 기본값 |
| **난이도** | 초보자 친화적 | 경험 필요 | 경험 필요 |
| **betas** | [0.9, 0.99] | [0.9, 0.99] | [0.9, 0.999] |

**Prodigy 동작 원리 (간략):**

```
기존 Adam: 사용자가 LR을 정해줘야 함 (너무 크면 발산, 너무 작으면 안 배움)

Prodigy:
  1. 내부 변수 d를 유지 (LR의 대리값)
  2. gradient의 크기와 가중치 변화량을 관찰
  3. 적절한 d를 자동으로 추정
  4. 실제 LR = 사용자 lr × d

  → lr=1로 설정하면 d가 곧 실제 LR이 됨
  → TensorBoard에서 d 값의 변화를 관찰 가능
```

**AdamW8bitKahan의 장점:** Adam optimizer는 1차 모멘텀($m_t$)과 2차 모멘텀($v_t$)을 파라미터당 저장하므로, 모델 가중치의 2배 추가 메모리가 필요합니다. 8bit 양자화로 이 상태를 절반으로 줄이고, Kahan summation으로 bf16에서의 부동소수점 누적 오차를 보정합니다.

> 💡 **추천:** 처음 시작할 때는 **Prodigy** (LR 고민 없음). 결과를 보고 세밀하게 제어하고 싶으면 **AdamW8bitKahan**으로 전환.

---

### 4.2 OPTIMIZER_LR

```python
OPTIMIZER_LR = 1  #@param {type:"number"}
```

**TOML:** `lr = 1`

**의미:** 옵티마이저에 전달하는 learning rate.

**옵티마이저별 권장 값:**

| Optimizer | 권장 LR | 설명 |
|---|---|---|
| **Prodigy** | **1** (고정) | 실제 LR은 내부 d가 결정, lr은 스케일 팩터 |
| **AdamW8bitKahan** | 5e-5 ~ 5e-6 | 높으면 발산, 낮으면 학습 안 됨 |
| **adamw_optimi** | 2e-5 | diffusion-pipe 예제 기본값 |

**LR이 학습에 미치는 영향:**

```
LR 너무 높음:
  → gradient × LR이 너무 큼 → 가중치가 크게 요동 → loss 발산 (NaN)
  → LoRA에서는: 스타일이 뭉개지거나 "burned" 이미지

LR 적절:
  → 점진적으로 학습, loss 안정적으로 감소
  → LoRA에서는: 원본 스타일 유지하면서 새 특징 학습

LR 너무 낮음:
  → 가중치 변화가 너무 작음 → 100 epoch 돌려도 변화 없음
  → LoRA에서는: 원본과 거의 동일한 출력
```

> 💡 **Prodigy 사용자:** LR은 건드리지 마세요. 학습이 안 된다면 LR이 아니라 `weight_decay`, `rank`, `epochs`, `batch_size`를 먼저 점검하세요.

---

### 4.3 WEIGHT_DECAY

```python
WEIGHT_DECAY = 0.01  #@param {type:"number"}
```

**TOML:** `weight_decay = 0.01`

**의미:** 매 업데이트마다 가중치를 **조금씩 0 방향으로 줄이는** 정규화 기법.

**수식:**

$$W_{t+1} = W_t - \text{lr} \cdot \nabla\mathcal{L} - \text{lr} \cdot \lambda \cdot W_t$$

여기서 $\lambda$ = weight_decay. 마지막 항 $-\text{lr} \cdot \lambda \cdot W_t$가 가중치를 0쪽으로 당깁니다.

**LoRA에서의 의미:**

```
LoRA 가중치(B, A)에 weight decay 적용
  → 불필요하게 큰 LoRA 가중치를 억제
  → 과적합 방지 효과
  → 원본 모델(W)에서 최소한으로 벗어나도록 유도

weight_decay = 0:    제약 없음, 자유롭게 학습
weight_decay = 0.01: 약한 제약 (기본, 대부분 적절)
weight_decay = 0.1:  강한 제약 (변화가 적음, 과적합 억제 강함)
```

> 💡 **팁:** 과적합이 의심되면 `weight_decay`를 0.05~0.1로 올려보세요. 반대로 학습이 너무 느리면 0.001로 내릴 수 있습니다.

---

## 5. 저장 · 체크포인트 파라미터 — 셀 4-2

### 5.1 SAVE_EVERY_N_EPOCHS

```python
SAVE_EVERY_N_EPOCHS = 1  #@param {type:"integer"}
```

**TOML:** `save_every_n_epochs = 1`

**의미:** 몇 epoch마다 **LoRA 가중치 파일** (`adapter_model.safetensors`)을 저장할지.

**저장되는 파일:**

```
/content/training_output/run_name/
├── epoch1/
│   └── adapter_model.safetensors    ← LoRA 가중치 (~50-100 MB)
├── epoch2/
│   └── adapter_model.safetensors
├── ...
└── epoch100/
    └── adapter_model.safetensors
```

이 파일은 **ComfyUI에서 바로 사용 가능**한 LoRA입니다. 각 epoch의 LoRA를 비교하여 최적의 epoch를 찾는 것이 LoRA 학습의 핵심 워크플로우입니다.

**디스크 용량 고려:**

```
rank=32 기준 LoRA 파일 ≈ 50~80 MB
100 epoch × 매 epoch 저장 = 5~8 GB

Colab 디스크: ~78 GB (기본)
→ 모델(~5.5 GB) + 캐시 + LoRA 100개 → 약 15 GB 사용

절약하려면: save_every_n_epochs = 5 (20개만 저장 → ~1.5 GB)
```

> 💡 **실전:** 처음 학습할 때는 `save_every_n_epochs=1`로 모든 epoch를 저장하세요. 최적 epoch를 파악한 후, 재학습 시에는 간격을 넓힐 수 있습니다.

---

### 5.2 CHECKPOINT_MINUTES

```python
CHECKPOINT_MINUTES = 30  #@param {type:"integer"}
```

**TOML:** `checkpoint_every_n_minutes = 30`

**의미:** 몇 분마다 **DeepSpeed 전체 학습 상태**를 저장할지.

**LoRA 저장 vs DeepSpeed 체크포인트:**

| | LoRA 저장 (epoch별) | DeepSpeed 체크포인트 (분별) |
|---|---|---|
| 내용 | LoRA 가중치만 | 모든 가중치 + optimizer state + 학습 위치 |
| 크기 | ~50-100 MB | ~4-8 GB |
| 용도 | 추론 (ComfyUI) | **이어 학습** (Colab 끊김 후) |
| 호환성 | ComfyUI/WebUI | diffusion-pipe 전용 |

**Colab 끊김 복구 흐름:**

```
Colab 세션 끊김!
  │
  ▼ 셀 7-1에서 HF에 업로드해뒀다면:
  │
  ▼ 새 세션 → 셀 1~5 재실행 → 셀 6-2 (이어 학습) 실행
  │
  ▼ 마지막 체크포인트에서 자동 복구:
    - optimizer state 복원 (모멘텀 등)
    - 학습 위치 복원 (epoch, step)
    - LoRA 가중치 복원

→ checkpoint_minutes가 작을수록 끊겨도 잃는 학습량이 적음
→ 하지만 저장 자체가 몇 분 걸리므로 너무 잦으면 학습 지연
```

> 💡 **Colab 무료 세션:** 90분 제한이 있으므로 `checkpoint_minutes=30`이면 최대 30분 분량만 잃습니다. Pro 계정이면 60으로 올려도 됩니다.

---

## 6. 고정 설정 (변경 비권장)

노트북 상단과 코드 내에 하드코딩된 설정들입니다. T4 환경에서 이들을 변경하면 NaN이나 OOM이 발생할 수 있습니다.

### dtype = bfloat16

```toml
dtype = 'bfloat16'        # [model] 섹션
save_dtype = 'bfloat16'   # 전역
```

**왜 고정?**
- **fp16은 NaN 위험:** fp16의 최대값은 65,504. Flow Matching에서 $t \approx 1$ 근처의 loss가 이 범위를 넘어 NaN 발생 가능
- **bf16은 안전:** float32와 같은 범위(±3.4×10³⁸)를 가지면서 메모리는 절반
- **RMSNorm에서 fp32 강제:** `@torch.autocast('cuda', dtype=torch.float32)` — 정규화 연산은 bf16에서도 내부적으로 fp32

### pipeline_stages = 1

```toml
pipeline_stages = 1
```

**왜 고정?**
- pipeline_stages > 1은 **여러 GPU에 모델을 분할**하는 DeepSpeed Pipeline Parallelism
- Colab은 GPU 1개이므로 항상 1
- LoRA 학습에서는 1이 표준 (multi-GPU에서도)

### activation_checkpointing = true

```toml
activation_checkpointing = true
```

**왜 고정?**
- 28개 블록의 중간 활성화를 모두 저장하면 ~10 GB+ 추가 VRAM
- T4(16GB)에서는 이것 없이 학습 불가능
- 속도는 ~30% 느려지지만, VRAM 절약이 필수

---

## 7. 파라미터 조합 레시피

### 🟢 안전한 시작 (Colab T4, 첫 학습)

```
RESOLUTION = 512
NUM_REPEATS = 2
CACHE_SHUFFLE_NUM = 1
EPOCHS = 100
LORA_RANK = 32
BATCH_SIZE = 10
GRAD_ACCUM = 1
BLOCKS_TO_SWAP = 8
WARMUP_STEPS = 100
GRADIENT_CLIPPING = 1.0
TIMESTEP_SAMPLE = "logit_normal"
LLM_ADAPTER_LR = 0
OPTIMIZER = "Prodigy"
OPTIMIZER_LR = 1
WEIGHT_DECAY = 0.01
SAVE_EVERY_N_EPOCHS = 1
CHECKPOINT_MINUTES = 30
```

이 설정은 노트북의 기본값이며, T4에서 안정적으로 동작합니다.

### 🟡 소량 데이터 (10~20장)

```
NUM_REPEATS = 4         # epoch당 학습량 확보
CACHE_SHUFFLE_NUM = 3   # 텍스트 증강으로 과적합 방지
EPOCHS = 50             # 총 학습량: 20×4×50 = 4000
LORA_RANK = 16          # 과적합 방지를 위해 줄임
WEIGHT_DECAY = 0.05     # 정규화 강화
WARMUP_STEPS = 30       # 총 스텝이 적으므로 줄임
```

### 🟡 고해상도 학습 (768px, VRAM 여유 있을 때)

```
RESOLUTION = 768
BLOCKS_TO_SWAP = 14     # 메모리 추가 확보
BATCH_SIZE = 4          # 시퀀스 길이 증가 → 배치 줄여야
GRAD_ACCUM = 2          # 유효 배치 크기 유지
```

### 🔴 극한 메모리 절약 (VRAM 부족할 때)

```
RESOLUTION = 512
BLOCKS_TO_SWAP = 20     # 매우 공격적 오프로드
BATCH_SIZE = 1
GRAD_ACCUM = 4          # 유효 배치 = 4
LORA_RANK = 16          # LoRA 파라미터 줄임
```

### 🔵 빠른 프로토타이핑 (결과 빨리 보기)

```
EPOCHS = 20
SAVE_EVERY_N_EPOCHS = 1
NUM_REPEATS = 4
WARMUP_STEPS = 20
CHECKPOINT_MINUTES = 60  # 저장 빈도 줄여서 속도 확보
```

---

## 부록: 파라미터 → TOML 키 매핑

| 노트북 변수 | TOML 키 | TOML 파일 | 섹션 |
|---|---|---|---|
| `RESOLUTION` | `resolutions` | dataset.toml | 전역 |
| `NUM_REPEATS` | `num_repeats` | dataset.toml | `[[directory]]` |
| `CACHE_SHUFFLE_NUM` | `cache_shuffle_num` | dataset.toml | `[[directory]]` |
| `MIN_AR` | `min_ar` | dataset.toml | 전역 |
| `MAX_AR` | `max_ar` | dataset.toml | 전역 |
| `NUM_AR_BUCKETS` | `num_ar_buckets` | dataset.toml | 전역 |
| `EPOCHS` | `epochs` | training.toml | 전역 |
| `LORA_RANK` | `rank` | training.toml | `[adapter]` |
| `BATCH_SIZE` | `micro_batch_size_per_gpu` | training.toml | 전역 |
| `GRAD_ACCUM` | `gradient_accumulation_steps` | training.toml | 전역 |
| `BLOCKS_TO_SWAP` | `blocks_to_swap` | training.toml | 전역 |
| `WARMUP_STEPS` | `warmup_steps` | training.toml | 전역 |
| `GRADIENT_CLIPPING` | `gradient_clipping` | training.toml | 전역 |
| `TIMESTEP_SAMPLE` | `timestep_sample_method` | training.toml | `[model]` |
| `LLM_ADAPTER_LR` | `llm_adapter_lr` | training.toml | `[model]` |
| `OPTIMIZER` | `type` | training.toml | `[optimizer]` |
| `OPTIMIZER_LR` | `lr` | training.toml | `[optimizer]` |
| `WEIGHT_DECAY` | `weight_decay` | training.toml | `[optimizer]` |
| `SAVE_EVERY_N_EPOCHS` | `save_every_n_epochs` | training.toml | 전역 |
| `CHECKPOINT_MINUTES` | `checkpoint_every_n_minutes` | training.toml | 전역 |

---

*이 문서의 내용은 `anima_lora_train_colab.ipynb` 셀 4-1, 4-2의 코드와 `AnimaLoraToolkit/anima_train.py`, `diffusion-pipe/train.py`의 동작을 기반으로 작성되었습니다.*
