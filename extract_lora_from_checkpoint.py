"""
DeepSpeed 체크포인트(global_step*)에서 LoRA safetensors 추출

DeepSpeed는 exclude_frozen_parameters=True로 저장하므로
layer_*-model_states.pt 파일들에 LoRA 파라미터만 분산 저장됨.
이 스크립트가 전부 합쳐서 ComfyUI 호환 safetensors로 변환합니다.

사용법:
  python extract_lora_from_checkpoint.py <global_step_폴더_경로> [출력경로.safetensors]

예시:
  python extract_lora_from_checkpoint.py ./global_step547
  python extract_lora_from_checkpoint.py ./global_step547 my_lora.safetensors
  python extract_lora_from_checkpoint.py --batch ./20260206_04-38-55

Colab에서 다운로드하는 법:
  훈련 실행 중에도 왼쪽 📁 파일브라우저에서 global_step* 폴더를
  통째로 다운로드 가능 (셀 실행 불필요)

GPU 불필요 — CPU + RAM만 사용합니다.
"""

import sys
import os
import re
import torch
from pathlib import Path

try:
    from safetensors.torch import save_file
except ImportError:
    print("❌ safetensors가 설치되어 있지 않습니다.")
    print("   pip install safetensors")
    sys.exit(1)


def extract_lora_from_checkpoint(checkpoint_dir: str, output_path: str = None, save_dtype: str = "bfloat16"):
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        print(f"❌ 폴더를 찾을 수 없습니다: {checkpoint_dir}")
        sys.exit(1)

    # ── 1. layer 파일 수집 ──────────────────────────────────
    # DeepSpeed pipeline 체크포인트: layer_*-model_states.pt (exclude_frozen=True → LoRA만)
    layer_files = sorted(
        checkpoint_dir.glob("layer_*-model_states.pt"),
        key=lambda p: int(re.search(r"layer_(\d+)", p.name).group(1))
    )
    # 단일 파일 방식 (pipeline_stages=1 등)
    mp_file = checkpoint_dir / "mp_rank_00_model_states.pt"

    if not layer_files and not mp_file.exists():
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다.")
        print(f"   폴더 내용: {[f.name for f in checkpoint_dir.iterdir()]}")
        sys.exit(1)

    # 출력 경로 기본값
    if output_path is None:
        step_name = checkpoint_dir.name  # e.g., "global_step547"
        output_path = checkpoint_dir.parent / f"lora_{step_name}.safetensors"
    output_path = Path(output_path)

    print(f"📂 체크포인트: {checkpoint_dir}")
    print(f"📦 출력 경로:  {output_path}")

    # ── 2. 파라미터 로드 (CPU only) ────────────────────────
    # Anima to_layers() 구조:
    #   layer 0: InitialLayer (pos_embedder, x_embedder, t_embedder 등)
    #   layer 1: LLMAdapterLayer (llm_adapter)
    #   layer 2 ~ N+1: TransformerLayer(blocks[0] ~ blocks[N-1])
    #   layer N+2: FinalLayer (final_layer)
    #
    # DeepSpeed는 각 layer의 파라미터를 로컬 이름으로 저장하므로
    # layer 번호에서 원래 transformer 경로를 복원해야 합니다.

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    target_dtype = dtype_map.get(save_dtype, torch.bfloat16)
    comfyui_sd = {}
    total_keys = 0

    if layer_files:
        print(f"⏳ layer 파일 {len(layer_files)}개 로딩 중... (CPU only)")
        for lf in layer_files:
            layer_num = int(re.search(r"layer_(\d+)", lf.name).group(1))
            layer_sd = torch.load(lf, map_location="cpu", weights_only=False)

            if len(layer_sd) == 0:
                continue

            total_keys += len(layer_sd)

            for key, value in layer_sd.items():
                # 원래 transformer 경로 복원
                if layer_num == 0:
                    # InitialLayer: pos_embedder.*, x_embedder.*, t_embedder.* 등
                    # 키 그대로 사용 (이미 올바른 이름)
                    original_key = key
                elif layer_num == 1:
                    # LLMAdapterLayer: llm_adapter.*
                    original_key = key  # 이미 llm_adapter.* 형태
                elif layer_num >= 2 and layer_num <= len(layer_files) - 1:
                    # TransformerLayer: layer_num → blocks[layer_num - 2]
                    block_idx = layer_num - 2
                    # 로컬 이름: "block.self_attn.q_proj.lora_A.default.weight"
                    # 원본 이름: "blocks.{block_idx}.self_attn.q_proj.lora_A.default.weight"
                    if key.startswith("block."):
                        original_key = f"blocks.{block_idx}." + key[len("block."):]
                    else:
                        original_key = f"blocks.{block_idx}.{key}"
                else:
                    # FinalLayer: final_layer.*
                    original_key = key

                # .default. 제거 (PEFT LoRA adapter suffix)
                clean_key = original_key.replace(".default.", ".")
                # .modules_to_save. 제거
                clean_key = clean_key.replace(".modules_to_save.", ".")
                # ComfyUI 포맷: diffusion_model. prefix
                if not clean_key.startswith("diffusion_model."):
                    clean_key = "diffusion_model." + clean_key

                comfyui_sd[clean_key] = value.to(target_dtype)

            lora_count = len(layer_sd)
            if lora_count > 0:
                print(f"   layer_{layer_num:02d}: {lora_count} params", end="")
                if layer_num == 0:
                    print(" (InitialLayer)")
                elif layer_num == 1:
                    print(" (LLMAdapter)")
                elif layer_num <= len(layer_files) - 1:
                    print(f" → blocks.{layer_num - 2}")
                else:
                    print(" (FinalLayer)")
    else:
        # 단일 파일 모드 (mp_rank_00)
        print(f"⏳ {mp_file.name} 로딩 중... (CPU only)")
        checkpoint = torch.load(mp_file, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("module", checkpoint)
        total_keys = len(state_dict)

        for key, value in state_dict.items():
            clean_key = key.replace(".default.", ".").replace(".modules_to_save.", ".")
            if not clean_key.startswith("diffusion_model."):
                clean_key = "diffusion_model." + clean_key
            comfyui_sd[clean_key] = value.to(target_dtype)

    print(f"\n   전체 로드 키: {total_keys}")
    print(f"   출력 키 수:  {len(comfyui_sd)}")

    if len(comfyui_sd) == 0:
        print("❌ 파라미터를 찾을 수 없습니다.")
        sys.exit(1)

    # 키 샘플 미리보기
    sample = list(comfyui_sd.keys())[:5]
    print(f"   키 샘플: {sample}")

    # ── 3. safetensors 저장 ────────────────────────────────
    print(f"\n💾 저장 중... ({save_dtype}, {len(comfyui_sd)} tensors)")
    os.makedirs(output_path.parent, exist_ok=True)
    save_file(comfyui_sd, str(output_path), metadata={"format": "pt"})

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"✅ 저장 완료: {output_path} ({file_size:.1f} MB)")

    # 키 샘플 출력
    sample = list(comfyui_sd.keys())[:5]
    print(f"\n🔑 키 샘플:")
    for k in sample:
        print(f"   {k} → {comfyui_sd[k].shape} ({comfyui_sd[k].dtype})")

    return output_path


def batch_extract(run_dir: str, save_dtype: str = "bfloat16"):
    """run 폴더 안의 모든 global_step*에서 LoRA 추출"""
    run_dir = Path(run_dir)
    step_dirs = sorted(
        [d for d in run_dir.glob("global_step*") if d.is_dir()],
        key=lambda p: int(re.search(r"(\d+)$", p.name).group(1))
    )

    if not step_dirs:
        print(f"❌ global_step* 폴더를 찾을 수 없습니다: {run_dir}")
        sys.exit(1)

    print(f"📂 Run 폴더: {run_dir}")
    print(f"🔄 {len(step_dirs)}개 체크포인트 발견\n")

    for step_dir in step_dirs:
        print(f"{'='*60}")
        try:
            extract_lora_from_checkpoint(str(step_dir), save_dtype=save_dtype)
        except Exception as e:
            print(f"⚠️ {step_dir.name} 실패: {e}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("사용 가능한 모드:")
        print("  1. 단일:  python extract_lora_from_checkpoint.py ./global_step547")
        print("  2. 일괄:  python extract_lora_from_checkpoint.py --batch ./run_폴더")
        print("  3. 지정:  python extract_lora_from_checkpoint.py ./global_step547 output.safetensors")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("❌ run 폴더 경로를 지정하세요.")
            sys.exit(1)
        batch_extract(sys.argv[2])
    else:
        checkpoint_path = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else None
        extract_lora_from_checkpoint(checkpoint_path, output)
