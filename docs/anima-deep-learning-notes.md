# Anima 코드로 배우는 딥러닝 핵심 기법

> Anima / diffusion-pipe 코드베이스에서 발견되는 현대 딥러닝 테크닉을  
> **파이토치 초보 관점**에서 "왜 이렇게 하는지" 중심으로 정리한 문서입니다.
>
> 아키텍처 전체 구조는 [anima-architecture.md](anima-architecture.md)를 참고하세요.

---

## 목차

1. [einops — 텐서 형상 변환의 가독성 혁명](#1-einops--텐서-형상-변환의-가독성-혁명)
2. [RMSNorm vs LayerNorm — 왜 두 가지를 섞어 쓸까](#2-rmsnorm-vs-layernorm--왜-두-가지를-섞어-쓸까)
3. [RoPE (Rotary Position Embedding) — 위치 정보를 회전으로](#3-rope-rotary-position-embedding--위치-정보를-회전으로)
4. [Attention — Self / Cross / Scaled Dot-Product](#4-attention--self--cross--scaled-dot-product)
5. [AdaLN (Adaptive Layer Normalization) — 조건부 정규화](#5-adaln-adaptive-layer-normalization--조건부-정규화)
6. [Residual Connection + Gating — 안정적인 깊은 네트워크](#6-residual-connection--gating--안정적인-깊은-네트워크)
7. [가중치 초기화 — 왜 init_weights()를 직접 작성할까](#7-가중치-초기화--왜-init_weights를-직접-작성할까)
8. [Mixed Precision (bf16) — 메모리 절반, 속도 2배](#8-mixed-precision-bf16--메모리-절반-속도-2배)
9. [Gradient Checkpointing — VRAM을 시간으로 교환](#9-gradient-checkpointing--vram을-시간으로-교환)
10. [Flow Matching vs DDPM — 확산 모델의 두 갈래](#10-flow-matching-vs-ddpm--확산-모델의-두-갈래)
11. [LoRA / LoKr — 거대 모델을 적은 파라미터로 미세조정](#11-lora--lokr--거대-모델을-적은-파라미터로-미세조정)
12. [Block Swap — GPU 메모리 한계를 넘는 법](#12-block-swap--gpu-메모리-한계를-넘는-법)
13. [register_buffer — 학습하지 않지만 모델에 속한 텐서](#13-register_buffer--학습하지-않지만-모델에-속한-텐서)
14. [torch.no_grad / inference_mode — 추론 시 메모리 절약](#14-torchno_grad--inference_mode--추론-시-메모리-절약)

---

## 1. einops — 텐서 형상 변환의 가독성 혁명

### 뭐가 문제인가?

파이토치에서 텐서의 차원을 바꾸려면 `view`, `permute`, `reshape`, `transpose`를 조합해야 합니다. 5차원 비디오 텐서 `(B, C, T, H, W)`를 다루면 이런 코드가 나옵니다:

```python
# ❌ 읽기 어려운 방식
x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, T*H*W, C)
```

### Anima에서의 사용

```python
from einops import rearrange, repeat

# ✅ einops로 쓰면 의도가 명확
x = rearrange(x, "b c t h w -> b (t h w) c")           # 5D → 3D: 공간 축을 시퀀스로 펼침
x = rearrange(x, "b (t h w) d -> b t h w d", t=T, h=H, w=W)  # 다시 복원

# PatchEmbed에서 패치 분할
x = Rearrange("b c (t r) (h m) (w n) -> b t h w (c r m n)", r=1, m=2, n=2)
# 의미: temporal을 1개씩, spatial을 2×2로 잘라서 채널 방향으로 합침
```

**`rearrange` 읽는 법:**
- 왼쪽 `→` 오른쪽: "이 형상을 저 형상으로 바꿔라"
- `()` 안의 차원들은 곱해져서 하나로 합침(flatten) 또는 나눠짐(split)
- 변수 이름이 곧 문서

```python
# repeat: 특정 축을 반복 복사
half_emb_h = repeat(half_emb_h, "h d -> t h w d", t=T, w=W)
# 높이 임베딩을 시간·너비 방향으로 타일링
```

> 💡 **팁:** einops는 "이 코드가 뭘 하는지" 주석을 쓸 필요가 없어집니다. 패턴 문자열 자체가 주석입니다.

---

## 2. RMSNorm vs LayerNorm — 왜 두 가지를 섞어 쓸까

### LayerNorm (표준)

```python
# 파이토치 내장
nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
```

평균을 빼고, 분산으로 나눕니다:

$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

### RMSNorm (경량화)

```python
# Anima 직접 구현 (cosmos_predict2_modeling.py)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 학습 가능한 스케일

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    @torch.autocast('cuda', dtype=torch.float32)  # ← 정밀도 보호!
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

**평균을 빼지 않고**, RMS(Root Mean Square)로만 나눕니다:

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$$

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

### Anima에서의 사용 분담

| 위치 | 사용하는 Norm | 이유 |
|---|---|---|
| Q/K 정규화 (`q_norm`, `k_norm`) | **RMSNorm** | attention score 스케일만 맞추면 됨, 평균 이동 불필요 |
| DiT Block 내부 (`layer_norm_*`) | **LayerNorm** (affine=False) | AdaLN이 scale/shift를 대신 제공하므로 기본 정규화만 수행 |
| LLM Adapter 내부 | **RMSNorm** | LLM 계열(Llama, Qwen)의 표준 선택 |
| Timestep Norm (`t_embedding_norm`) | **RMSNorm** | 경량화 |

> 💡 **눈여겨볼 점:** `@torch.autocast('cuda', dtype=torch.float32)` — 정규화 연산은 **항상 fp32**로 수행합니다. bf16에서 제곱·평균·제곱근을 하면 수치 오류가 커지기 때문입니다. 이것은 현업에서 매우 흔한 패턴입니다.

---

## 3. RoPE (Rotary Position Embedding) — 위치 정보를 회전으로

### 왜 필요한가?

Transformer는 입력 순서를 모릅니다. "고양이가 물고기를 먹는다"와 "물고기가 고양이를 먹는다"를 구분 못 합니다. 위치 정보를 따로 넣어줘야 합니다.

### 기존 방식의 한계

- **Learned Positional Embedding**: 학습 시 본 길이까지만 동작, 외삽 불가
- **Sinusoidal (sin/cos)**: 고정이라 유연하지 않음

### RoPE의 아이디어

위치 정보를 벡터에 **더하지 않고**, 벡터를 **회전**시킵니다:

```python
def rotate_half(x):
    """벡터의 반을 부호 뒤집어서 교차 배치"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    """각 위치마다 다른 각도로 회전"""
    return (x * cos) + (rotate_half(x) * sin)
```

**핵심 성질:** 위치 $i$의 Q와 위치 $j$의 K를 내적하면, 결과는 **상대 거리** $|i-j|$에만 의존합니다. 즉 "내가 어디에 있는지"가 아니라 "둘 사이의 거리가 얼마인지"를 자동으로 인코딩합니다.

### Anima의 3D RoPE

이미지/비디오는 (시간, 높이, 너비) 3개 축이 있습니다. Anima는 head_dim을 3등분하여 각 축에 독립적인 RoPE를 적용합니다:

```python
# head_dim을 시간/높이/너비로 분할
dim_h = head_dim // 6 * 2   # 높이에 할당
dim_w = dim_h               # 너비에 할당 (같은 크기)
dim_t = head_dim - 2*dim_h  # 나머지 → 시간에 할당

# 각 축마다 주파수 생성
h_freqs = 1.0 / (theta_h ** (range / dim_h))
w_freqs = 1.0 / (theta_w ** (range / dim_w))
t_freqs = 1.0 / (theta_t ** (range / dim_t))

# 3축의 임베딩을 concat하여 하나의 RoPE 텐서 생성
em = cat([
    repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
    repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
    repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
] * 2, dim=-1)   # *2는 cos/sin 양쪽에 쓰기 위해
```

### NTK-Aware Scaling

훈련할 때 본 해상도보다 **큰 이미지**를 생성하고 싶을 때, RoPE의 기저 주파수 $\theta$를 키워서 외삽합니다:

```python
# extrapolation_ratio=4.0이면:
ntk_factor = 4.0 ** (dim_h / (dim_h - 2))
theta_adjusted = 10000.0 * ntk_factor  # 기본 10000보다 훨씬 큰 값
```

> 💡 **팁:** RoPE는 LLM(Llama, Qwen 등)에서도 표준입니다. Anima가 Qwen3와 DiT 양쪽에 RoPE를 사용하므로, 이 코드를 이해하면 LLM 코드도 읽을 수 있게 됩니다.

---

## 4. Attention — Self / Cross / Scaled Dot-Product

### 기본 Multi-Head Attention

```python
# Anima의 Attention 클래스에서 핵심 부분만 추출
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, n_heads=8, head_dim=64):
        # context_dim=None이면 Self-Attention, 있으면 Cross-Attention
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim

        # Q, K, V 각각 별도의 Linear
        self.q_proj = nn.Linear(query_dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, n_heads * head_dim, bias=False)

        # Q/K 정규화 — 학습 안정성을 위해
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, context=None, rope_emb=None):
        context = x if context is None else context  # Self-attn이면 자기 자신이 context

        q = self.q_proj(x)    # (B, S_q, D) → (B, S_q, H*d)
        k = self.k_proj(context)  # (B, S_k, D) → (B, S_k, H*d)
        v = self.v_proj(context)

        # Multi-head로 분리: (B, S, H*d) → (B, S, H, d)
        q, k, v = [rearrange(t, "b s (h d) -> b s h d", h=n_heads) for t in (q, k, v)]

        # 정규화 + RoPE (Self-Attn에만 적용!)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb(q, rope_emb)
            k = apply_rotary_pos_emb(k, rope_emb)

        # Scaled Dot-Product Attention (파이토치 내장, Flash Attention 자동 활성화)
        attn_out = F.scaled_dot_product_attention(q, k, v)

        return self.output_proj(rearrange(attn_out, "b s h d -> b s (h d)"))
```

### Self-Attention vs Cross-Attention

```
Self-Attention (이미지 토큰 ↔ 이미지 토큰):
  Q, K, V 모두 같은 텐서(x)에서 나옴
  → "이미지의 각 패치가 다른 패치들과의 관계를 학습"
  → RoPE로 공간적 위치 인식

Cross-Attention (이미지 토큰 ← 텍스트 임베딩):
  Q는 이미지에서, K/V는 텍스트에서
  → "이미지의 각 패치가 텍스트의 어떤 부분에 주목할지 학습"
  → RoPE 없음 (텍스트 위치와 이미지 위치는 무관)
```

### `F.scaled_dot_product_attention`

```python
# 파이토치 2.0+ 에서 제공하는 통합 Attention 함수
# 내부적으로 Flash Attention / Memory Efficient Attention을 자동 선택
attn_output = F.scaled_dot_product_attention(q, k, v)
```

이 한 줄이 수동으로 구현하면 이런 코드를 대체합니다:

```python
# ❌ 수동 구현 (느리고, 메모리 많이 씀)
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
attn_weights = F.softmax(scores, dim=-1)
attn_output = torch.matmul(attn_weights, v)
```

> 💡 **팁:** `F.scaled_dot_product_attention`은 GPU에서 자동으로 Flash Attention v2를 사용합니다. 수동 구현 대비 메모리 $O(N^2) \to O(N)$, 속도 2~4배 향상.

---

## 5. AdaLN (Adaptive Layer Normalization) — 조건부 정규화

### 일반 LayerNorm의 한계

LayerNorm은 항상 같은 방식으로 정규화합니다. 하지만 디퓨전 모델은 **"지금 몇 번째 노이즈 제거 단계인지"**(timestep)에 따라 다르게 행동해야 합니다.

### AdaLN의 아이디어

timestep에 따라 **shift, scale, gate** 3개의 변조 파라미터를 동적으로 생성합니다:

```python
# 각 Block의 forward에서:

# 1. timestep embedding으로부터 변조값 생성
shift, scale, gate = self.adaln_modulation(t_emb).chunk(3, dim=-1)

# 2. LayerNorm 후 scale/shift 적용
x_normed = LayerNorm(x) * (1 + scale) + shift
#                         ^^^^^^^^^^^^^^^^^^^^^^^^
#                         이 부분이 timestep에 따라 달라짐!

# 3. Attention/MLP 통과 후 gate로 강도 조절
x = x + gate * sublayer_output
#       ^^^^
#       gate가 0에 가까우면 이 레이어의 영향이 약해짐
```

### AdaLN-LoRA — 메모리 절약 변형

Anima는 일반 AdaLN 대신 **Low-Rank** 버전을 사용합니다:

```python
# 일반 AdaLN:
#   Linear(D → 3D)  ← 파라미터: D × 3D = 3D²

# AdaLN-LoRA:
#   Linear(D → 256) → Linear(256 → 3D)  ← 파라미터: 256D + 256×3D ≈ 1024D
#   D=2048일 때, 3D²=12.6M → 1024D=2.1M (약 6배 절약)
```

또한 `TimestepEmbedding`이 전역 `adaln_lora` 텐서 `(B, T, 3D)`를 생성하여, **모든 블록에 공유**합니다. 각 블록의 개별 AdaLN에 이 공유 텐서를 더해서 최종 변조값을 만듭니다:

```python
# Block.forward() 안에서:
(shift, scale, gate) = (
    self.adaln_modulation_self_attn(t_emb)  # 블록 고유
    + adaln_lora                             # 전 블록 공유 (residual)
).chunk(3, dim=-1)
```

> 💡 **눈여겨볼 점:** 이것은 "전역 조건(timestep)" + "블록별 미세 조정"의 패턴입니다. 파라미터를 아끼면서도 블록마다 다른 행동을 가능하게 합니다.

---

## 6. Residual Connection + Gating — 안정적인 깊은 네트워크

### 기본 Residual Connection

```python
# ResNet 이후 거의 모든 딥러닝의 기본
x = x + sublayer(x)
```

입력을 그대로 더하는 "지름길"을 만들어서, 28층(Anima)처럼 깊은 네트워크도 안정적으로 학습합니다.

### Gated Residual (Anima 방식)

Anima의 DiT 블록은 단순 덧셈 대신 **gate**를 곱합니다:

```python
# Self-attention (gated)
x = x + gate_self_attn * self_attn(norm(x))

# Cross-attention (gated)
x = x + gate_cross_attn * cross_attn(norm(x), text_emb)

# MLP (gated)
x = x + gate_mlp * mlp(norm(x))
```

gate는 timestep에서 생성되는 실수 벡터(D차원)입니다. 학습 초반에는 `adaln_modulation`의 마지막 Linear가 **zero init**되어 있으므로:

```python
# init_weights()에서:
torch.nn.init.zeros_(self.adaln_modulation[2].weight)  # ← 마지막 레이어
```

이는 gate ≈ 0으로 시작한다는 뜻이고, 따라서:
- **학습 초기**: 모든 블록이 항등 함수 (x = x + 0)처럼 동작
- **학습 진행**: gate가 서서히 열리면서 각 레이어의 역할이 분화

> 💡 **이것이 중요한 이유:** 28개 블록을 한꺼번에 랜덤 초기화하면 gradient가 폭발하거나 소멸합니다. Zero-init gating은 "처음에는 아무것도 안 하다가 천천히 배우기 시작"하는 효과로, 학습 초기 안정성을 극적으로 높입니다.

---

## 7. 가중치 초기화 — 왜 init_weights()를 직접 작성할까

파이토치는 `nn.Linear`를 만들면 기본 초기화(Kaiming uniform)를 적용합니다. 하지만 Anima는 거의 모든 모듈에서 **직접 초기화**합니다.

### Truncated Normal 초기화

```python
def init_weights(self):
    # 표준편차 = 1 / sqrt(입력 차원)
    std = 1.0 / math.sqrt(self._dim)

    # 3σ 바깥의 극단값을 잘라냄 (truncated)
    torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3*std, b=3*std)
```

**왜 truncated?** 정규분포에서 극단적으로 큰/작은 값이 나오면 학습 초반에 gradient가 불안정해집니다. `a=-3σ, b=3σ`로 잘라내면 99.7%의 값은 유지하면서 극단값만 제거합니다.

### 깊이에 따른 스케일 조정

```python
# GPT2FeedForward에서:
std = 1.0 / math.sqrt(self._hidden_dim)
if self._layer_id is not None:
    std = std / math.sqrt(2 * (self._layer_id + 1))
    #                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
    #  깊은 레이어일수록 초기 가중치를 더 작게!
```

이것은 GPT-2 논문의 기법으로, 깊은 레이어의 초기 기여를 줄여서 gradient 안정성을 확보합니다.

### 특수 위치의 Zero Init

```python
# LLM Adapter의 output projection:
def init_weights(self):
    torch.nn.init.zeros_(self.o_proj.weight)    # ← 출력 0으로 시작

# AdaLN modulation의 마지막 레이어:
torch.nn.init.zeros_(self.adaln_modulation[-1].weight)  # gate ≈ 0
```

이 패턴은 "처음에는 이 모듈이 없는 것처럼 행동하라"는 의미입니다:
- **Cross-attention 출력 zero init**: 처음에 텍스트 조건이 영향을 주지 않음 → 서서히 학습
- **AdaLN gate zero init**: 위 섹션 6에서 설명한 gated residual의 핵심

> 💡 **팁:** 새 모듈을 기존 네트워크에 붙일 때, 출력을 zero init하면 "기존 네트워크의 동작을 해치지 않으면서 새 기능을 서서히 학습"할 수 있습니다. LoRA의 `lora_up` zero init도 같은 원리입니다.

---

## 8. Mixed Precision (bf16) — 메모리 절반, 속도 2배

### 부동소수점 종류

| 타입 | 비트 | 지수부 | 가수부 | 범위 | 정밀도 |
|---|---|---|---|---|---|
| float32 | 32 | 8 | 23 | ±3.4×10³⁸ | ~7자리 |
| float16 (fp16) | 16 | 5 | 10 | ±65504 | ~3자리 |
| **bfloat16 (bf16)** | 16 | **8** | 7 | ±3.4×10³⁸ | ~2자리 |

### bf16 vs fp16

```
fp16:  작은 범위, 높은 정밀도 → NaN 위험 (loss가 65504 넘으면 폭발)
bf16:  큰 범위,   낮은 정밀도 → 안전 (float32와 같은 범위, 정밀도만 약간 희생)
```

Anima 노트북에서 bf16을 강제하는 이유가 바로 이것입니다:

```python
# anima_lora_train_colab.ipynb 셀 1-1에서:
bf16_support = torch.cuda.is_bf16_supported()
if not bf16_support:
    print("⚠️ BF16 미지원 — TOML에서 dtype을 float16으로 수정 필요")
```

### torch.autocast — 자동 혼합 정밀도

```python
# 학습 시:
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(inputs)
    loss = criterion(output, target)
# autocast 블록 안에서는 대부분 bf16으로 연산
# 하지만 일부 연산(softmax, norm 등)은 자동으로 fp32 유지

# Anima의 RMSNorm에서 명시적 보호:
@torch.autocast('cuda', dtype=torch.float32)  # ← 이 함수는 항상 fp32!
def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight
```

> 💡 **핵심 원칙:** "연산은 bf16, 정규화/누적/손실 계산은 fp32." 이 규칙만 지키면 메모리를 절반으로 줄이면서 학습 안정성을 유지할 수 있습니다.

---

## 9. Gradient Checkpointing — VRAM을 시간으로 교환

### 문제: 역전파에는 중간 결과가 필요하다

```
Forward: x → [Block1] → a₁ → [Block2] → a₂ → ... → [Block28] → output
                         ↑              ↑
                     이 값들을 메모리에 보관해야 backward에서 쓸 수 있음
```

28개 블록의 중간 활성화를 전부 저장하면 수십 GB가 필요합니다.

### 해결: 중간 결과를 버리고, 필요할 때 다시 계산

```python
from torch.utils.checkpoint import checkpoint

# anima_train.py에서:
for block in model.blocks:
    def custom_forward(x, blk=block):
        return blk(x, t_embedding, cross, **block_kwargs)

    x = checkpoint(custom_forward, x, use_reentrant=False)
    #   ^^^^^^^^^^ 중간 활성화를 저장하지 않음
    #              backward 시 forward를 다시 실행하여 계산
```

**트레이드오프:**

| | 일반 학습 | Gradient Checkpointing |
|---|---|---|
| 메모리 | 모든 중간 활성화 저장 | 입/출력만 저장 |
| 속도 | 1× | ~1.3× (forward를 2번 실행) |
| VRAM | 매우 많음 | **~30-50% 절약** |

> 💡 **Colab T4 (16GB VRAM)** 같은 환경에서는 이것 없이는 2B 모델 학습이 불가능합니다. Anima 노트북에서 `activation_checkpointing = true`가 "변경 비권장" 고정값인 이유입니다.

---

## 10. Flow Matching vs DDPM — 확산 모델의 두 갈래

### DDPM (2020~2023의 주류)

```
Forward (노이즈 추가):
  x_t = √(ᾱₜ) · x₀ + √(1-ᾱₜ) · ε     ← 복잡한 스케줄

모델 예측:
  ε̂ = model(x_t, t)                     ← "노이즈"를 예측

추론:
  1000스텝에서 시작해서 1스텝씩 제거     ← 느림 (50~100스텝 필요)
```

### Flow Matching / Rectified Flow (Anima, Flux, SD3)

```
Forward (직선 보간):
  x_t = (1-t) · x₀ + t · ε               ← 단순! 직선 경로

모델 예측:
  v̂ = model(x_t, t)                      ← "속도"를 예측 (v = ε - x₀)

추론:
  ODE를 따라감                             ← 20~30스텝이면 충분
```

### 코드에서 보기

```python
# anima_train.py에서의 학습 루프 핵심:

# 1. 시간 샘플링 (logit-normal)
t = torch.sigmoid(torch.randn(batch_size, device=device))
t = (t * 3.0) / (1 + 2.0 * t)  # shift=3.0

# 2. 노이즈 혼합 (직선 보간)
noise = torch.randn_like(latents)
x_t = (1 - t) * latents + t * noise     # ← 이게 전부!

# 3. 속도 예측
target_velocity = noise - latents        # v = ε - x₀
predicted_velocity = model(x_t, t, text_emb)

# 4. 손실 = MSE
loss = F.mse_loss(predicted_velocity, target_velocity)
```

**Rectified Flow가 좋은 이유:**
- **직선 경로**: 곡선보다 학습하기 쉬움
- **적은 스텝**: 직선이므로 25스텝이면 충분 (DDPM은 50~1000)
- **단순한 수식**: 스케줄 설계가 필요 없음

> 💡 **눈여겨볼 점:** `t`의 분포가 균등(`uniform`)이 아니라 `logit_normal`인 이유는, 중간 시점(t≈0.5)에서 학습이 더 많이 일어나도록 하기 위해서입니다. 극단적으로 깨끗하거나(t≈0) 극단적으로 노이즈가 많은(t≈1) 구간은 학습 효과가 적습니다.

---

## 11. LoRA / LoKr — 거대 모델을 적은 파라미터로 미세조정

### 문제: 2B 모델을 전체 fine-tuning하려면?

```
파라미터 수: ~2,000,000,000개
메모리 (bf16): ~4GB (모델) + ~8GB (optimizer states) + ~4GB (gradients) = 16GB+
→ 일반 GPU로는 불가능
```

### LoRA의 아이디어

큰 행렬 $W$를 직접 수정하지 않고, **작은 행렬 두 개의 곱**을 더합니다:

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=32):
        # W: (out, in) — 원본은 건드리지 않음
        self.lora_down = nn.Linear(in_features, rank, bias=False)   # A: (rank, in)
        self.lora_up = nn.Linear(rank, out_features, bias=False)    # B: (out, rank)

        # A는 랜덤, B는 0으로 초기화 → 처음에는 LoRA 출력 = 0
        nn.init.kaiming_uniform_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.lora_up(self.lora_down(x)) * (alpha / rank)
```

```
원래: y = Wx                파라미터: out × in (예: 2048 × 2048 = 4M)
LoRA: y = Wx + BAx · α/r   파라미터: out × r + r × in (예: 2048×32 + 32×2048 = 131K)
                            → 원본의 3%만 학습!
```

### LoKr (Kronecker Product) — 더 효율적인 분해

```python
class LoKrLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=32, factor=8):
        # Kronecker Product: kron(W1, W2)
        self.lokr_w1 = nn.Parameter(torch.empty(factor, factor))
        self.lokr_w2 = nn.Parameter(torch.empty(out_features//factor, in_features//factor))

    def forward(self, x):
        weight = torch.kron(self.lokr_w1, self.lokr_w2)
        return F.linear(x, weight) * (alpha / rank)
```

**Kronecker Product란?** $W_1 \otimes W_2$는 $W_1$의 각 원소에 $W_2$ 전체를 곱해서 큰 행렬을 만듭니다:

```
W1 = [[a, b],     W2 = [[e, f],
      [c, d]]           [g, h]]

kron(W1, W2) = [[a·e, a·f, b·e, b·f],
                [a·g, a·h, b·g, b·h],
                [c·e, c·f, d·e, d·f],
                [c·g, c·h, d·g, d·h]]
```

factor=8일 때 파라미터: `8×8 + 256×256 = 65,600` (LoRA rank=32의 `131K`보다 적음!)

### 어디에 주입하나?

```python
# LoRAInjector의 기본 타겟:
DEFAULT_TARGETS = [
    "q_proj",       # attention query
    "k_proj",       # attention key
    "v_proj",       # attention value
    "output_proj",  # attention output
    "mlp.layer1",   # FFN 첫 번째 레이어
    "mlp.layer2",   # FFN 두 번째 레이어
]
```

원본 모델은 **완전히 frozen** (`requires_grad_(False)`), LoRA 파라미터만 학습합니다.

> 💡 **팁:** LoRA rank=32, alpha=32, LoKr factor=8은 Anima에서의 기본값입니다. rank를 높이면 표현력은 올라가지만 과적합 위험도 커지고, 메모리도 더 듭니다.

---

## 12. Block Swap — GPU 메모리 한계를 넘는 법

### 아이디어

28개 블록을 **전부 GPU에 올리지 않습니다.** 실행 중인 블록만 GPU에 두고, 나머지는 CPU RAM에 보관합니다.

```
┌─ GPU ──────────────────────────┐    ┌─ CPU RAM ───────────┐
│ Block 0  ← 현재 실행 중         │    │ Block 20            │
│ Block 1  ← 다음 실행 예정       │    │ Block 21            │
│ Block 2                        │    │ Block 22            │
│ ...                            │    │ ...                 │
│ Block 19                       │    │ Block 27            │
│                                │    │                     │
│ [CUDA Stream으로 비동기 전송]    │◄──►│                     │
└────────────────────────────────┘    └─────────────────────┘
```

**핵심 기법들:**

```python
# 1. 비동기 전송 (non-blocking)
block.to('cuda', non_blocking=True)  # CPU→GPU 전송을 요청만 하고 바로 리턴
torch.cuda.synchronize()              # 전송 완료를 기다림

# 2. CUDA Stream으로 연산과 전송을 겹침
#    Block N을 실행하는 동안, Block N+1을 미리 GPU로 전송
compute_stream = torch.cuda.Stream()
transfer_stream = torch.cuda.Stream()

with torch.cuda.stream(compute_stream):
    output = current_block(x)           # 현재 블록 연산

with torch.cuda.stream(transfer_stream):
    next_block.to('cuda', non_blocking=True)  # 다음 블록 미리 전송
```

### TOML에서의 설정

```toml
blocks_to_swap = 8  # 28블록 중 8개를 CPU에 오프로드
# → GPU에 20개만 상주, 메모리 ~30% 절약
# 값을 키울수록 메모리 절약 ↑, 속도 ↓
```

**주의사항:** LoRA 파라미터는 swap에서 **제외**됩니다. optimizer가 항상 GPU에서 접근해야 하기 때문입니다.

> 💡 **Colab T4(16GB)에서 2B 모델을 돌릴 수 있는 이유:** gradient checkpointing + block swap + bf16 + LoRA(소수 파라미터만 학습). 이 4가지의 조합이 없으면 최소 48GB+ GPU가 필요합니다.

---

## 13. register_buffer — 학습하지 않지만 모델에 속한 텐서

### 문제

RoPE의 주파수 벡터, 정규화 상수, 고정 마스크 등은:
- 학습 대상이 **아니지만** (gradient 불필요)
- 모델과 함께 **저장/로드** 되어야 하고
- `.to(device)` 호출 시 함께 **이동**해야 함

### 해결

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # ❌ 그냥 self.inv_freq = inv_freq 하면?
        #    → model.to('cuda') 해도 CPU에 남아있음
        #    → model.state_dict()에 포함 안 됨

        # ✅ register_buffer로 등록
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        #                                          ^^^^^^^^^^^^^^^^
        #                     persistent=False: state_dict에는 저장하지 않음
        #                     persistent=True:  state_dict에 저장 (기본값)
```

### nn.Parameter vs register_buffer vs 일반 속성

| 방식 | gradient 계산 | `.to(device)` 이동 | `state_dict` 저장 |
|---|---|---|---|
| `nn.Parameter(x)` | ✅ | ✅ | ✅ |
| `register_buffer(x, persistent=True)` | ❌ | ✅ | ✅ |
| `register_buffer(x, persistent=False)` | ❌ | ✅ | ❌ |
| `self.x = tensor` | ❌ | ❌ | ❌ |

> 💡 **팁:** "이 텐서가 학습되어야 하나?"를 먼저 판단하세요. 학습 대상이면 `Parameter`, 아니면 `register_buffer`. 절대 일반 속성으로 텐서를 저장하지 마세요 (device 불일치 버그의 주범).

---

## 14. torch.no_grad / inference_mode — 추론 시 메모리 절약

### 왜 필요한가?

파이토치는 기본적으로 **모든 연산의 계산 그래프를 기록**합니다 (역전파를 위해). 추론 시에는 이게 불필요한 메모리 낭비입니다.

### 세 가지 방법

```python
# 1. torch.no_grad() — 가장 흔히 쓰임
@torch.no_grad()
def sample_image(model, ...):
    output = model(x)  # 계산 그래프 기록 안 함 → 메모리 절약

# 2. torch.inference_mode() — no_grad보다 더 강력
with torch.inference_mode():
    hidden = qwen_model(input_ids, output_hidden_states=True)
    # 텐서가 "inference 전용"으로 표시됨
    # → in-place 연산 최적화 등 추가 이점

# 3. requires_grad_(False) — 특정 모델만 동결
model.requires_grad_(False)  # 이 모델의 모든 파라미터 gradient 비활성화
# 학습 중에도 VAE, 텍스트 인코더 등 frozen 모듈에 사용
```

### Anima에서의 사용 패턴

```python
# 모델 로드 시: 학습 대상이 아닌 것들 동결
qwen_model.eval().requires_grad_(False)  # 텍스트 인코더 동결
vae.requires_grad_(False)                # VAE 동결
model.requires_grad_(False)              # DiT 전체 동결
# → 이후 LoRA 파라미터만 requires_grad=True로 설정

# 추론(샘플링) 시: 전체를 no_grad로
@torch.no_grad()
def sample_image(model, vae, ...):
    model.eval()    # BatchNorm/Dropout 등을 평가 모드로
    # ... 이미지 생성 ...
    model.train()   # 다시 학습 모드로 복귀
```

**`.eval()` vs `.requires_grad_(False)` vs `torch.no_grad()` 차이:**

| | `.eval()` | `.requires_grad_(False)` | `torch.no_grad()` |
|---|---|---|---|
| 역할 | BatchNorm/Dropout 동작 변경 | gradient 계산 비활성화 | 계산 그래프 기록 중단 |
| 메모리 절약 | ❌ | 약간 | **✅ 크게** |
| 용도 | 추론 모드 전환 | 파라미터 동결 | 추론 시 메모리 절약 |
| 되돌리기 | `.train()` | `.requires_grad_(True)` | with 블록 종료 |

> 💡 **흔한 실수:** `.eval()`만 호출하면 메모리는 절약되지 않습니다. 반드시 `torch.no_grad()` 또는 `inference_mode()`를 **함께** 사용해야 합니다. Anima 코드에서도 두 가지를 항상 같이 씁니다.

---

## 마무리: 이 기법들이 어디서 다시 나타나는가

이 문서에서 다룬 기법들은 Anima뿐 아니라 현대 딥러닝의 **거의 모든 대형 모델**에서 동일하게 사용됩니다:

| 기법 | Anima | Stable Diffusion 3 | Flux | LLaMA 3 | GPT-4 계열 |
|---|---|---|---|---|---|
| RMSNorm | ✅ | ✅ | ✅ | ✅ | ✅ (추정) |
| RoPE | ✅ (3D) | ✅ (2D) | ✅ (2D) | ✅ (1D) | ✅ (1D) |
| AdaLN | ✅ | ✅ | ✅ | ❌ | ❌ |
| Flow Matching | ✅ | ✅ | ✅ | ❌ | ❌ |
| LoRA | ✅ | ✅ | ✅ | ✅ | ✅ |
| Gradient Checkpoint | ✅ | ✅ | ✅ | ✅ | ✅ |
| bf16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Flash Attention | ✅ | ✅ | ✅ | ✅ | ✅ |
| einops | ✅ | ✅ | ✅ | ❌ | ❌ |

Anima 코드를 읽으면서 이 기법들을 익히면, 다른 대형 모델의 코드도 자연스럽게 읽을 수 있게 됩니다.

---

*이 문서의 코드 예시는 `AnimaLoraToolkit/models/cosmos_predict2_modeling.py`, `anima_modeling.py`, `anima_train.py`에서 발췌·간소화한 것입니다.*
