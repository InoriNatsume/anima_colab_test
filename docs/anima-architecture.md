# Anima 모델 아키텍처 · 훈련 · 추론 구조 해설

> **대상 버전**: Anima 2.2 (Preview) — CircleStone Labs  
> **베이스 모델**: NVIDIA Cosmos-Predict2 (논문: arXiv 2511.00062v1, 단 논문은 v2.5 / T2I 한정)  
> **훈련 프레임워크**: [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) (DeepSpeed 기반)  
> **노트북**: `anima_lora_train_colab.ipynb`

---

## 목차

1. [전체 파이프라인 개요](#1-전체-파이프라인-개요)
2. [모델 구성 요소](#2-모델-구성-요소)
   - 2.1 [Transformer (DiT) — MiniTrainDIT](#21-transformer-dit--minitraindit)
   - 2.2 [LLM Adapter — Qwen3 ↔ T5 브릿지](#22-llm-adapter--qwen3--t5-브릿지)
   - 2.3 [텍스트 인코더 — Qwen3-0.6B + T5 토크나이저](#23-텍스트-인코더--qwen3-06b--t5-토크나이저)
   - 2.4 [VAE — Wan 2.1 VAE](#24-vae--wan-21-vae)
3. [Transformer 블록 상세 구조](#3-transformer-블록-상세-구조)
   - 3.1 [PatchEmbed — 패치 임베딩](#31-patchembed--패치-임베딩)
   - 3.2 [3D RoPE — 위치 임베딩](#32-3d-rope--위치-임베딩)
   - 3.3 [Timestep Embedding — 시간 조건부](#33-timestep-embedding--시간-조건부)
   - 3.4 [Block — Self-Attn + Cross-Attn + MLP + AdaLN](#34-block--self-attn--cross-attn--mlp--adaln)
   - 3.5 [FinalLayer — 출력 레이어](#35-finallayer--출력-레이어)
4. [훈련 방식 — Rectified Flow (Flow Matching)](#4-훈련-방식--rectified-flow-flow-matching)
   - 4.1 [노이즈 스케줄과 시간 샘플링](#41-노이즈-스케줄과-시간-샘플링)
   - 4.2 [손실 함수](#42-손실-함수)
   - 4.3 [LoRA / LoKr 적용](#43-lora--lokr-적용)
   - 4.4 [Block Swap — 메모리 최적화](#44-block-swap--메모리-최적화)
5. [추론 방식 — ER-SDE Solver](#5-추론-방식--er-sde-solver)
   - 5.1 [시그마 스케줄 (Simple Scheduler)](#51-시그마-스케줄-simple-scheduler)
   - 5.2 [CFG (Classifier-Free Guidance)](#52-cfg-classifier-free-guidance)
   - 5.3 [ER-SDE-Solver-3](#53-er-sde-solver-3)
6. [diffusion-pipe 통합 구조](#6-diffusion-pipe-통합-구조)
   - 6.1 [파이프라인 스테이지 분해](#61-파이프라인-스테이지-분해)
   - 6.2 [캐싱 전략](#62-캐싱-전략)
   - 6.3 [Component별 Learning Rate](#63-component별-learning-rate)
7. [Cosmos-Predict2 논문과의 관계](#7-cosmos-predict2-논문과의-관계)
8. [부록: 모델 설정 상수](#8-부록-모델-설정-상수)

---

## 1. 전체 파이프라인 개요

Anima는 텍스트 → 이미지 생성 모델로, 아래 순서로 작동합니다:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Anima T2I Pipeline                             │
│                                                                         │
│  Text Prompt                                                            │
│    │                                                                    │
│    ├──→ [Qwen3-0.6B Tokenizer] ──→ [Qwen3-0.6B Model] ──→ hidden_states│
│    │                                                     (B, S, 1024)   │
│    │                                                         │          │
│    └──→ [T5 Tokenizer] ──→ token_ids ───────────────────────┐│          │
│                              (B, S')                         ││          │
│                                                              ▼▼          │
│                                                     ┌──────────────┐    │
│                                                     │ LLM Adapter  │    │
│                                                     │ (6-layer Tx) │    │
│                                                     └──────┬───────┘    │
│                                                            │            │
│                                                   cross_attn_emb        │
│                                                     (B, S', 1024)       │
│                                                            │            │
│  Random Noise (B,16,1,H/8,W/8)                            │            │
│    │                                                       │            │
│    ▼                                                       ▼            │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │              MiniTrainDIT (Cosmos-Predict2 기반)               │      │
│  │  ┌──────────┐ ┌────────┐ ┌────────────────────────────────┐  │      │
│  │  │PatchEmbed│→│ +RoPE  │→│ N × Block (Self+Cross+MLP)     │  │      │
│  │  └──────────┘ └────────┘ │   + AdaLN(timestep_embedding)  │  │      │
│  │                          └──────────────┬─────────────────┘  │      │
│  │                                         ▼                    │      │
│  │                                   FinalLayer                 │      │
│  │                                   + Unpatchify               │      │
│  └─────────────────────────────────────┬───────────────────────┘      │
│                                        │                               │
│                                        ▼                               │
│                             Velocity prediction                        │
│                                   (B,16,1,H/8,W/8)                    │
│                                        │                               │
│                               [ER-SDE Solver]                          │
│                             (iterative denoising)                      │
│                                        │                               │
│                                        ▼                               │
│                              Denoised latent                           │
│                                        │                               │
│                                ┌───────┴───────┐                      │
│                                │   Wan 2.1 VAE  │                      │
│                                │   (Decoder)    │                      │
│                                └───────┬───────┘                      │
│                                        ▼                               │
│                                  Output Image                          │
│                                   (B,3,H,W)                           │
└─────────────────────────────────────────────────────────────────────────┘
```

**핵심 특징:**
- **Dual Text Encoder**: Qwen3-0.6B (의미 추출) + T5 Tokenizer (토큰 구조 제공)
- **LLM Adapter**: 두 인코더 출력을 결합하는 6-layer Transformer 브릿지
- **Cosmos-Predict2 DiT**: AdaLN-LoRA 변조가 적용된 DiT 아키텍처
- **Rectified Flow**: velocity 예측 기반 Flow Matching 훈련
- **ER-SDE Solver**: 추론 시 고속 수렴 SDE 솔버

---

## 2. 모델 구성 요소

### 2.1 Transformer (DiT) — MiniTrainDIT

`MiniTrainDIT`는 NVIDIA Cosmos-Predict2의 DiT(Diffusion Transformer) 구현체입니다. Anima 클래스는 이를 상속하여 LLM Adapter를 추가합니다.

```
Anima (extends MiniTrainDIT)
├── x_embedder: PatchEmbed
│     spatial_patch=2, temporal_patch=1
│     in_channels=16+1(padding_mask)=17 → model_channels
│
├── pos_embedder: VideoRopePosition3DEmb
│     3D RoPE (Temporal × Height × Width)
│     head_dim 분할: dim_h = dim_w = dim//6*2, dim_t = dim - 2*dim_h
│
├── t_embedder: Timesteps → TimestepEmbedding
│     sinusoidal encoding → SiLU → Linear
│     AdaLN-LoRA: Linear(D, 256) → Linear(256, 3D)
│
├── t_embedding_norm: RMSNorm(model_channels)
│
├── llm_adapter: LLMAdapter  ← Anima 고유
│     source_dim=1024, target_dim=1024, 6 layers
│
├── blocks: N × Block
│     각 블록 = Self-Attn + Cross-Attn + MLP
│     모든 서브레이어에 AdaLN-LoRA 변조
│
└── final_layer: FinalLayer
      LayerNorm → Linear(D, patch²×out_channels)
      AdaLN 변조 (shift + scale)
```

**모델 크기 변종 (checkpoint에서 자동 감지):**

| model_channels | num_blocks | num_heads | 예상 파라미터 |
|:---:|:---:|:---:|:---|
| 2048 | 28 | 16 | ~2B (Anima Preview) |
| 5120 | 36 | 40 | ~14B (대형 모델) |

### 2.2 LLM Adapter — Qwen3 ↔ T5 브릿지

Anima의 가장 독특한 구조입니다. Qwen3의 풍부한 의미 임베딩을 T5 토큰 구조 위에 매핑하여, DiT의 cross-attention에 전달합니다.

```
LLM Adapter 내부 구조
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

입력:
  source_hidden_states: Qwen3 hidden states (B, S_qwen, 1024)
  target_input_ids:     T5 token IDs        (B, S_t5)

처리 흐름:
  ┌─ T5 token IDs ──→ nn.Embedding(32128, 1024) ──→ in_proj ──→ x
  │                                                              │
  │  Qwen3 hidden states ──────────────────────────→ context     │
  │                                                              │
  │  position_ids ──→ RotaryEmbedding ──→ cos, sin              │
  │                                                              │
  │  ┌──────────────────────────────────────────────────────┐   │
  │  │ × 6 LLMAdapterTransformerBlock                       │   │
  │  │   ├─ Self-Attention (x→x, RoPE 적용)                 │   │
  │  │   ├─ Cross-Attention (x→context, RoPE 적용)          │   │
  │  │   └─ MLP (Linear→GELU→Linear)                        │   │
  │  └──────────────────────────────────────────────────────┘   │
  │                              │                               │
  │                              ▼                               │
  │                    out_proj(model_dim → target_dim)          │
  │                              │                               │
  │                         RMSNorm(1024)                        │
  │                              │                               │
  └──────────────────────────────┘                               │
                                                                 ▼
  출력: cross_attn_emb (B, S_t5, 1024)
```

**설계 의도:**
- T5 Embedding은 토큰 위치의 **구조적 틀**을 제공 (vocab size = 32128)
- Qwen3 hidden states는 **의미 정보**를 cross-attention으로 주입
- Self-attention은 T5 토큰 간 관계를 학습
- Cross-attention은 Qwen3의 의미를 각 T5 토큰에 주입
- 6-layer 깊이로 충분한 표현력 확보

### 2.3 텍스트 인코더 — Qwen3-0.6B + T5 토크나이저

Anima는 **실제 T5 모델을 사용하지 않습니다.** T5 토크나이저만 사용합니다.

| 구성 요소 | 역할 | 출력 |
|---|---|---|
| **Qwen3-0.6B** | CausalLM의 마지막 hidden state 추출 | `(B, S, 1024)` 임베딩 |
| **T5 Tokenizer** | 프롬프트 → 토큰 ID 변환 | `(B, S')` 정수 ID |

```python
# Qwen3 인코딩 (anima_train.py)
outputs = qwen_model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    output_hidden_states=True,  # ← 핵심: 마지막 hidden state 사용
    return_dict=True,
    use_cache=False,
)
hidden = outputs.hidden_states[-1]  # (B, S, 1024)

# T5 토크나이즈 (토큰 ID만, 모델 실행 없음)
t5_ids = t5_tokenizer(prompt, ...)["input_ids"]  # (B, S')
```

**diffusion-pipe에서의 처리:**
- `llm_path`가 **파일**이면 Qwen3-0.6B safetensors로 직접 로드
- `llm_path`가 **디렉토리**면 HuggingFace AutoModelForCausalLM으로 로드
- T5 토크나이저는 항상 내장 사전(`configs/t5_old/`)에서 로드

### 2.4 VAE — Wan 2.1 VAE

```
Wan 2.1 VAE 구성
━━━━━━━━━━━━━━━━━━
  dim=96, z_dim=16
  dim_mult=[1, 2, 4, 4]
  num_res_blocks=2
  temporal_downsample=[False, True, True]

인코딩: pixel (B,3,1,H,W) → latent (B,16,1,H/8,W/8)
디코딩: latent → pixel

정규화 (16채널):
  mean = [-0.7571, -0.7089, -0.9113, 0.1075, ...]
  std  = [ 2.8184,  1.4541,  2.3275, 2.6558, ...]
  
  encode: z_normalized = (z - mean) / std
  decode: z = z_normalized * std + mean
```

---

## 3. Transformer 블록 상세 구조

### 3.1 PatchEmbed — 패치 임베딩

입력 텐서를 2×2 spatial 패치로 분할하고 선형 임베딩합니다.

```
입력: (B, C, T, H, W) — C=16(latent) + 1(padding_mask) = 17
      예: (1, 17, 1, 128, 128)

Rearrange: "b c (t r) (h m) (w n) → b t h w (c r m n)"
           r=1(temporal_patch), m=n=2(spatial_patch)

Linear: (17 × 1 × 2 × 2) = 68 → model_channels

출력: (B, T, H/2, W/2, D)
      예: (1, 1, 64, 64, 2048)
```

### 3.2 3D RoPE — 위치 임베딩

`VideoRopePosition3DEmb`는 시간·높이·너비 3축에 대한 Rotary Position Embedding을 생성합니다.

```
head_dim 분할:
  dim_h = head_dim // 6 * 2   (높이 차원)
  dim_w = dim_h               (너비 차원)
  dim_t = head_dim - 2*dim_h  (시간 차원)

각 축의 주파수:
  freq_h = 1 / (θ_h ^ (range / dim_h))
  freq_w = 1 / (θ_w ^ (range / dim_w))
  freq_t = 1 / (θ_t ^ (range / dim_t))

  θ 기본값 = 10000.0 × NTK_factor

NTK Extrapolation (해상도 외삽):
  h_extrapolation_ratio = 4.0 (in_channels==16일 때) 또는 3.0
  w_extrapolation_ratio = 4.0 또는 3.0
  t_extrapolation_ratio = 1.0

최종: (T×H×W, 1, 1, head_dim) 형태로 self-attention에 적용
```

**NTK-Aware Scaling**은 Cosmos 논문(§3.1)에서 설명된 방식으로, 훈련 해상도를 넘는 이미지에서도 위치 인코딩이 안정적으로 작동하도록 합니다.

### 3.3 Timestep Embedding — 시간 조건부

```
timestep (B, T)  ← T=1 for images
    │
    ▼
Timesteps: sinusoidal frequency encoding → (B, T, D)
    │
    ▼
TimestepEmbedding (AdaLN-LoRA 모드):
    Linear(D, D, bias=False) → SiLU → Linear(D, 3D, bias=False)
    │                                      │
    │                                      ▼
    │                              adaln_lora (B, T, 3D)
    ▼                              → 각 Block에 shift/scale/gate 제공
  emb (B, T, D)
    │
    ▼
  RMSNorm(D) → t_embedding
```

`adaln_lora`는 `TimestepEmbedding`이 직접 `3 × D` 크기의 텐서를 출력하여, 각 블록의 AdaLN 변조에 **residual로 더해지는** low-rank 보정값을 제공합니다. 이는 별도의 AdaLN 네트워크(`adaln_lora_dim=256`)와 합산되어 최종 modulation 파라미터를 결정합니다.

### 3.4 Block — Self-Attn + Cross-Attn + MLP + AdaLN

각 블록은 3개의 서브레이어로 구성되며, **모든 서브레이어에 독립적인 AdaLN 변조**가 적용됩니다.

```
Block Forward Pass
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

입력: x (B, T, H, W, D), t_emb (B, T, D), cross_emb (B, S, 1024)

[1] extra_pos_emb이 있으면 x에 더함 (learnable absolute position)

[2] AdaLN 변조값 계산 (per sub-layer):
    for sub in [self_attn, cross_attn, mlp]:
        (shift, scale, gate) = adaln_modulation(t_emb) + adaln_lora
        각각 (B, T, D) → broadcast (B, T, 1, 1, D)

[3] Self-Attention:
    x_norm = LayerNorm(x) * (1 + scale_sa) + shift_sa
    x_flat = rearrange(x_norm, "b t h w d → b (t h w) d")
    attn_out = MultiHeadAttention(
        Q = RMSNorm(q_proj(x_flat)),    ← RoPE 적용
        K = RMSNorm(k_proj(x_flat)),    ← RoPE 적용
        V = v_proj(x_flat)
    )
    x = x + gate_sa * rearrange(attn_out, "b (t h w) d → b t h w d")

[4] Cross-Attention:
    x_norm = LayerNorm(x) * (1 + scale_ca) + shift_ca
    x_flat = rearrange(x_norm, "b t h w d → b (t h w) d")
    attn_out = MultiHeadAttention(
        Q = RMSNorm(q_proj(x_flat)),      ← query: 이미지 토큰
        K = RMSNorm(k_proj(cross_emb)),   ← key: 텍스트 임베딩 (RoPE 없음)
        V = v_proj(cross_emb)             ← value: 텍스트 임베딩
    )
    x = x + gate_ca * rearrange(attn_out, "b (t h w) d → b t h w d")

[5] MLP (GPT2FeedForward):
    x_norm = LayerNorm(x) * (1 + scale_mlp) + shift_mlp
    mlp_out = Linear(D, 4D) → GELU → Linear(4D, D)
    x = x + gate_mlp * mlp_out

출력: x (B, T, H, W, D)
```

**AdaLN-LoRA 변조 수식:**

$$\text{modulation} = \text{adaln\_modulation}(t\_emb) + \text{adaln\_lora}$$

여기서 `adaln_modulation`은 `SiLU → Linear(D, 256) → Linear(256, 3D)` 구조이고, `adaln_lora`는 timestep embedding으로부터 직접 나오는 `(B, T, 3D)` 텐서입니다.

$$x' = \text{LayerNorm}(x) \cdot (1 + \text{scale}) + \text{shift}$$

$$\text{output} = x + \text{gate} \cdot \text{sublayer}(x')$$

### 3.5 FinalLayer — 출력 레이어

```
x (B, T, H, W, D)
    │
    ▼
AdaLN: (shift, scale) = adaln_modulation(t_emb) + adaln_lora[:, :, :2D]
    │
    ▼
LayerNorm(x) * (1 + scale) + shift
    │
    ▼
Linear(D → patch_s² × patch_t × out_channels)
= Linear(D → 2² × 1 × 16 = 64)
    │
    ▼
Unpatchify: rearrange("B T H W (p1 p2 t C) → B C (T t) (H p1) (W p2)")
    │
    ▼
출력: (B, 16, T, H×2, W×2) — 원본 latent 크기로 복원
```

---

## 4. 훈련 방식 — Rectified Flow (Flow Matching)

### 4.1 노이즈 스케줄과 시간 샘플링

Anima는 **Rectified Flow** (Flow Matching) 패러다임을 사용합니다. 이는 Flux, SD3와 동일한 계열입니다.

**시간 샘플링 (logit-normal):**

$$t \sim \sigma(\mathcal{N}(0, 1))$$

여기서 $\sigma$는 시그모이드 함수입니다. 이렇게 샘플된 $t$에 shift를 적용합니다:

$$t_{\text{shifted}} = \frac{\alpha \cdot t}{1 + (\alpha - 1) \cdot t}, \quad \alpha = 3.0$$

**노이즈 혼합 (CONST 스케줄):**

$$x_t = (1 - t) \cdot x_0 + t \cdot \varepsilon$$

여기서 $x_0$는 clean latent, $\varepsilon \sim \mathcal{N}(0, I)$

**학습 타겟 (velocity):**

$$v = \varepsilon - x_0$$

모델은 $x_t$와 $t$로부터 $v$를 예측합니다:

$$\hat{v} = f_\theta(x_t, t, c_{\text{text}})$$

### 4.2 손실 함수

**기본: MSE Loss**

$$\mathcal{L} = \mathbb{E}_{t, \varepsilon} \left[ \| \hat{v} - v \|_2^2 \right]$$

diffusion-pipe에서는 추가 옵션을 지원합니다:

- **Pseudo-Huber Loss** (선택적): 이상치에 강건한 변형
  
  $$\mathcal{L}_{\text{huber}} = c^2 \cdot (\sqrt{1 + (\hat{v} - v)^2 / c^2} - 1)$$

- **Multiscale Loss** (선택적): 이미지 크기가 1024×0.9을 초과할 때, 2× 다운샘플 버전에서도 MSE를 계산하여 가중 합산

> **참고:** NVIDIA 원본은 $1/(1-t)^2$ loss weighting을 사용하지만, diffusion-pipe는 이를 생략합니다.

### 4.3 LoRA / LoKr 적용

**Standard LoRA:**

$$W' = W + \frac{\alpha}{r} \cdot BA$$

- $B \in \mathbb{R}^{d_{out} \times r}$, $A \in \mathbb{R}^{r \times d_{in}}$
- Kaiming uniform init for $A$, zero init for $B$

**LoKr (LyCORIS):**

$$W' = W + \frac{\alpha}{r} \cdot \text{kron}(W_1, W_2)$$

- $W_1 \in \mathbb{R}^{f \times f}$, $W_2 \in \mathbb{R}^{d_{out}/f \times d_{in}/f}$
- `factor` $f$는 $d_{in}$과 $d_{out}$을 동시에 나눌 수 있는 값

**주입 대상 (기본):**
```
q_proj, k_proj, v_proj, output_proj, mlp.layer1, mlp.layer2
```

**ComfyUI 호환 저장 포맷:**
```
lora_unet_blocks_0_self_attn_q_proj.lora_down.weight
lora_unet_blocks_0_self_attn_q_proj.lora_up.weight
lora_unet_blocks_0_self_attn_q_proj.alpha
```

### 4.4 Block Swap — 메모리 최적화

diffusion-pipe의 핵심 VRAM 절약 기법입니다. DiT의 N개 블록 중 일부를 CPU에 오프로드합니다.

```
블록 배치 (blocks_to_swap=8 예시, 28블록 기준):

Forward 진행:
  GPU: [B0, B1, B2, ..., B19]   ← 20개 상주
  CPU: [B20, B21, ..., B27]     ← 8개 오프로드

  B20 실행 직전 → B20을 GPU로 async 이동
  B0 실행 완료 → B0을 CPU로 async 이동
  (CUDA stream으로 비동기 전송)

Backward 진행:
  역순으로 동일한 swap 수행
  backward hook으로 자동 관리

⚠️ LoRA 파라미터는 swap에서 제외 (optimizer가 항상 GPU에서 접근해야 함)
```

---

## 5. 추론 방식 — ER-SDE Solver

### 5.1 시그마 스케줄 (Simple Scheduler)

ComfyUI의 `ModelSamplingDiscreteFlow` + `simple_scheduler`를 재현합니다.

```python
# 1. 균등 타임스텝 생성
ts = arange(1, 1001) / 1000  # (0, 1]

# 2. SNR Shift 적용 (shift=3.0)
sigmas = (alpha * ts) / (1 + (alpha - 1) * ts)   # alpha=3.0

# 3. steps개로 서브샘플
sigmas = [sigmas[-(1 + int(i * len/steps))] for i in range(steps)]
sigmas.append(0.0)  # 끝에 0 추가

# 4. 첫 시그마가 1.0이면 약간 낮춤 (logit 발산 방지)
if sigmas[0] >= 1.0:
    sigmas[0] = shift(1.0 - 1e-4)
```

결과: `[σ_0, σ_1, ..., σ_{N-1}, 0.0]` — 높은 값에서 0까지 단조 감소

### 5.2 CFG (Classifier-Free Guidance)

```python
v_cond   = model(x_t, t, positive_prompt_emb)
v_uncond = model(x_t, t, negative_prompt_emb)
v = v_uncond + cfg_scale * (v_cond - v_uncond)
```

- 기본 `cfg_scale = 4.0`
- 기본 negative prompt: `"worst quality, low quality, blurry, ..."`

**Denoised 예측 (CONST flow):**

$$\hat{x}_0 = x_t - \sigma \cdot v$$

### 5.3 ER-SDE-Solver-3

Extended Reverse-Time SDE Solver의 3-stage 구현입니다. ComfyUI의 `sample_er_sde`를 재현합니다.

```
각 스텝 i에서:

[Stage 1] Euler 기본 업데이트:
  x = r_α · r · x + α_t · (1 - r) · denoised

  여기서:
    r_α = α_t / α_s                  (alpha ratio)
    r = noise_scaler(λ_t) / noise_scaler(λ_s)
    noise_scaler(λ) = λ · exp(λ^0.3) + 10λ

[Stage 2] 2차 보정 (i ≥ 1):
  denoised_d = (denoised - old_denoised) / (λ_s - λ_{s-1})
  x += α_t · (Δλ + s · noise_scaler(λ_t)) · denoised_d

  여기서 s = ∫ (1/noise_scaler(λ)) dλ  (수치 적분)

[Stage 3] 3차 보정 (i ≥ 2):
  denoised_u = (denoised_d - old_d) / ((λ_s - λ_{s-2}) / 2)
  x += α_t · (Δλ²/2 + s_u · noise_scaler(λ_t)) · denoised_u

[Stochastic] SDE 노이즈 항:
  sde_scale = sqrt(max(0, λ_t² - λ_s² · r²))
  x += α_t · noise · s_noise · sde_scale
```

**half_log_snr** (CONST 스케줄):

$$\lambda = -\text{logit}(\sigma) = \log\frac{1-\sigma}{\sigma}$$

$$\text{er\_lambda} = e^{-\lambda} = \frac{\sigma}{1-\sigma}$$

---

## 6. diffusion-pipe 통합 구조

### 6.1 파이프라인 스테이지 분해

diffusion-pipe는 DeepSpeed의 PipelineModule을 사용합니다. Anima 모델은 다음과 같이 분해됩니다:

```
Stage 1: EmbedAndEncode
  ├─ PatchEmbed (x_embedder)
  ├─ 3D RoPE 생성 (pos_embedder)
  ├─ Timestep Embedding (t_embedder + norm)
  └─ Padding mask 처리

Stage 2: LLMAdapterLayer
  └─ Qwen3 embeddings + T5 IDs → cross_attn_emb

Stage 3~N+2: BlockLayer × N
  └─ 각 DiT Block (with block-swap 지원)

Stage N+3: FinalLayer
  ├─ AdaLN modulation
  ├─ Linear projection
  └─ Unpatchify
```

### 6.2 캐싱 전략

diffusion-pipe는 **Latent + Text Embedding 사전 캐싱**을 수행합니다:

```
학습 시작 전 (1회):
  1. [VAE] 모든 이미지 → latent 변환 → npz 저장
  2. [Qwen3] 모든 캡션 → hidden states → 캐시
  3. [T5 Tokenizer] 모든 캡션 → token IDs → 캐시

학습 루프:
  - 캐시에서 (latent, text_emb, t5_ids) 로드
  - VAE/Qwen3 모델은 메모리에서 해제 가능
  - LLM Adapter + DiT만 GPU에 로드
```

**Colab 노트북의 셔플 캐싱:**
```toml
cache_shuffle_num = 1        # 캡션 변형 수 (태그 순서 셔플)
cache_shuffle_delimiter = ', '  # 태그 구분자
```

### 6.3 Component별 Learning Rate

diffusion-pipe는 모델 구성 요소별로 별도의 학습률을 지원합니다:

| 파라미터 그룹 | 설정 키 | 기본값 |
|---|---|---|
| Base (embedder, norm 등) | `base_lr` | optimizer lr |
| Self-Attention | `self_attn_lr` | base_lr |
| Cross-Attention | `cross_attn_lr` | base_lr |
| MLP | `mlp_lr` | base_lr |
| AdaLN Modulation | `adaln_lr` | base_lr |
| **LLM Adapter** | `llm_adapter_lr` | base_lr |

> **중요:** `llm_adapter_lr = 0`으로 설정하면 LLM Adapter가 **동결**됩니다. 소규모 데이터셋에서는 Adapter 동결이 권장됩니다 (Colab 노트북 기본값).

---

## 7. Cosmos-Predict2 논문과의 관계

> 논문: *"Cosmos World Foundation Model Platform for Physical AI"* (arXiv: 2511.00062v1)

### 공통점 (Anima 2.2가 계승한 부분)

| 항목 | 설명 |
|---|---|
| **DiT 아키텍처** | Video-aware DiT with AdaLN modulation — 논문의 핵심 구조 |
| **3D RoPE** | Temporal × Height × Width 분리 주파수, NTK-Aware Scaling |
| **PatchEmbed** | 3D 패치 (spatial=2, temporal=1) + padding mask concat |
| **RMSNorm** | Q/K 정규화에 RMSNorm 사용 |
| **Flow Matching** | Rectified Flow / velocity 예측 방식 |
| **AdaLN-LoRA** | Low-rank factorized adaptive layer normalization |

### 차이점 (Anima 2.2 vs 논문 v2.5)

| 항목 | Cosmos-Predict2 (v2.5, 논문) | Anima 2.2 |
|---|---|---|
| **목적** | Video + Image (범용) | Image Only (T2I, 애니메이션 특화) |
| **텍스트 인코더** | T5-XXL (직접 사용) | Qwen3-0.6B + T5 Tokenizer (LLM Adapter 브릿지) |
| **모델 크기** | 14B (Full) | ~2B (Preview) |
| **FPS 변조** | ✅ fps-aware RoPE | ❌ T=1 고정 (이미지 전용) |
| **Guardrail** | NVIDIA Aegis 안전 필터 | 없음 (오픈 모델) |
| **Loss Weighting** | $1/(1-t)^2$ | 균등 (가중치 없음) |
| **Extra Pos Emb** | per-block learnable abs. pos. emb. | 설정 가능하나 Preview에서 비활성 |
| **Context Parallel** | ✅ 분산 학습 지원 | ❌ 단일 GPU 포커스 |

### Anima가 추가한 요소

| 항목 | 설명 |
|---|---|
| **LLM Adapter** | Qwen3→T5 브릿지 (6-layer Transformer), Cosmos에는 없는 구조 |
| **Dual Tokenization** | 하나의 프롬프트를 Qwen3 + T5 두 가지로 토크나이즈 |
| **Weighted Tags** | `(tag:1.5)` 형식의 가중치 태그 지원 (ComfyUI 호환) |
| **ER-SDE Solver** | 추론 시 고차 SDE 솔버 (ComfyUI 표준) |
| **LoKr 지원** | LyCORIS Kronecker Product 분해 LoRA |

---

## 8. 부록: 모델 설정 상수

### Anima Preview (~2B) 기본 설정

```python
config = dict(
    # 공간/시간 범위
    max_img_h=240,           # 패치 단위 (실제 픽셀 × patch_spatial = 480)
    max_img_w=240,           # → VAE 8× 고려하면 latent 공간 480/8=60
    max_frames=128,          # 비디오용 (이미지에서는 T=1)

    # 채널
    in_channels=16,          # VAE z_dim
    out_channels=16,         # velocity 예측 채널
    patch_spatial=2,         # 2×2 패치
    patch_temporal=1,        # 시간 패치 없음

    # 모델 구조
    concat_padding_mask=True,  # 17채널 입력 (16+1)
    model_channels=2048,       # hidden dimension
    num_blocks=28,             # transformer block 수
    num_heads=16,              # attention head 수
    crossattn_emb_channels=1024,  # 텍스트 임베딩 차원

    # 위치 임베딩
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",

    # AdaLN-LoRA
    use_adaln_lora=True,
    adaln_lora_dim=256,

    # RoPE Extrapolation
    rope_h_extrapolation_ratio=4.0,
    rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0,
)

# LLM Adapter
llm_adapter_config = dict(
    source_dim=1024,    # Qwen3 hidden size
    target_dim=1024,    # DiT crossattn_emb_channels
    model_dim=1024,
    num_layers=6,
    num_heads=16,
    use_self_attn=True,
    layer_norm=False,   # RMSNorm 사용
)
```

### 추론 기본값 (ComfyUI 호환)

| 항목 | 값 |
|---|---|
| Sampler | `er_sde` |
| Scheduler | `simple` |
| Steps | 25 |
| CFG Scale | 4.0 |
| Shift | 3.0 |
| s_noise | 1.0 |
| max_stage | 3 |

### VAE 정규화 상수

```python
mean = [-0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
         0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921]

std  = [ 2.8184,  1.4541,  2.3275,  2.6558,  1.2196,  1.7708,  2.6052,  2.0743,
         3.2687,  2.1526,  2.8652,  1.5579,  1.6382,  1.1253,  2.8251,  1.9160]
```

---

*이 문서는 `anima_lora_train_colab.ipynb`, `AnimaLoraToolkit/`, `diffusion-pipe/` 소스 코드와 Cosmos-Predict2 논문(arXiv:2511.00062v1)을 기반으로 작성되었습니다.*
