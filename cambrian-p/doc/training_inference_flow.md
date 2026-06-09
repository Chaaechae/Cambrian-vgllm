# Cambrian-P 학습/추론 Flow

> Cambrian-P가 어떤 데이터로, 어떻게 forward 되어 loss까지 계산되는지(학습) 그리고 어떻게 답변과 카메라 포즈를 동시에 생성하는지(추론) 코드 기준으로 정리한 문서입니다.

## 0. 한 줄 요약

Cambrian-S-7B(SigLIP2 + Qwen2.5-7B + MLP projector)를 베이스로, **프레임마다 "카메라 토큰" 1개를 끼워 넣고**, LLM의 hidden state에서 그 토큰을 뽑아 **VGGT 기반 pose head**로 보내 카메라 포즈(translation/rotation/FoV)를 회귀합니다. 한 번의 forward로 **VQA 정답(text)** 과 **per-frame 카메라 포즈**를 동시에 학습/생성합니다.

핵심 파일:

| 역할 | 파일 |
|---|---|
| 학습 진입 | `cambrianp/train/train.py`, `cambrianp/scripts/Cambrian-P-7B.sh` |
| 멀티모달 조립 | `cambrianp/model/llava_arch.py` |
| LLM forward + loss 결합 | `cambrianp/model/language_model/llava_qwen.py` |
| pose head | `cambrianp/model/reconstructor/vggt_reconstructor.py` |
| 실제 loss | `vggt/vggt/loss/loss.py` |
| pose 인코딩 변환 | `vggt/vggt/utils/pose_enc.py` |

---

## 1. 학습 (Training) Flow

### 1-1. 베이스 모델 & 무엇을 학습하나

`Cambrian-P-7B.sh` 기준:

- `--model_name_or_path nyu-visionx/Cambrian-S-7B-S3` 에서 **fine-tune** 시작
- `--mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model,mm_rec_head"` → 비전타워 + projector + LLM + **rec_head(pose head)** 전부 학습
- 학습률 분리: 본체 `1e-5`, 비전타워 `mm_vision_tower_lr 2e-6`, downstream(pose) head `downstream_head_lr 1e-4`
- 새로 추가되는 모듈 2가지:
  - **카메라 토큰**: `token_num 2` → 학습 가능한 임베딩 2개 (`vggt_reconstructor.py:120-130`, std=1e-6로 초기화)
  - **VGGTReconstructor (rec_head)**: projector + CameraHead (`load_rec_model True`)

이 스크립트에서는 `--enable_camera True --enable_depth False --enable_point False` → **카메라 포즈만** 회귀합니다. (depth/point head 코드는 존재하지만 꺼져 있음. `enable_depth=False`이면 `num_intermediate_layers`가 1로 강제됨 — `vggt_reconstructor.py:45-47`)

### 1-2. 데이터셋 (3종)

| 종류 | 소스 | 역할 |
|---|---|---|
| VSI-590K | `nyu-visionx/vsi-590k` (~236GB) | Spatial VQA + scene geometry |
| Cambrian-S-3M | `nyu-visionx/Cambrian-S-3M` | pose 라벨이 붙는 절반의 비디오 백본 |
| Cambrian-P-Data (ViPE pseudo pose) | `nyu-visionx/Cambrian-P-Data` | dense pose supervision |

`train.py`의 `LazySupervisedDataset`이 세 가지를 merge:

- `_load_main_jsonl_sources()`: VQA용 jsonl (`{"video", "conversations":[human/gpt], "image", "id"}`)
- `_load_vipe_cambrians()`: Cambrian-S 비디오에 ViPE로 만든 pseudo-pose(`pose_path`, `depth_path`, `intrinsics_path`)를 **attach** 모드로 붙임
- (옵션) MapAnything / ViPE-WSDG 등 scene 기반 rec 데이터

### 1-3. 한 샘플이 만들어지는 과정 (`_get_item`)

1. **프레임 샘플링**: `frames_upbound 128`, `force_sample True` → 비디오에서 `np.linspace`로 균일하게 128프레임 추출. rec 데이터는 `vipe_cambrians_rec_data_mode temporal`(랜덤 시작 + interval 10~100) 또는 `cut3r`/`unified` 방식.
2. **프레임 전처리**: SigLIP processor로 resize/normalize (`anyres_max_9`, `image_grid_pinpoints`).
3. **대화 토크나이즈**: `qwen_1_5` 템플릿(`<|im_start|>`). **user 입력 토큰은 라벨을 `-100`(IGNORE_INDEX)** 으로 마스킹 → assistant 답변 토큰만 CE loss에 기여.
4. **GT supervision(`rec_views`)** 부착:
   - `extrinsics` [N,3,4] (world→camera), `intrinsics` [N,3,3], `depths` [N,192,192], `cam_points`/`world_points`, `point_masks`, `is_metric_scale`(ViPE는 False=상대 스케일), `scale_by_points`
5. **loss 제어 플래그**:
   - `bp_vqa`: 진짜 VQA 샘플이면 True
   - `bp_rec`: pose supervision이 있으면 True

### 1-4. Interleaved training (VQA ↔ Rec 혼합)

`--interleaved_training True`. VQA 태스크와 pose-reconstruction 태스크를 한 배치 안에 섞습니다.

- 각 unique scene마다 **dummy rec-only 샘플**을 생성(`dummy_vqa=True`, `bp_vqa=False`, `bp_rec=True`). 질문은 패딩용이고 VQA 라벨은 전부 마스킹.
- `--force_bp_rec True`, `--interleaved_aug_rec_ratio 1.0`로 rec 신호 비중 조정.
- 효과: 같은 비디오 백본에서 "언어 답변"과 "3D 포즈"를 번갈아 학습.

### 1-5. DataCollator → 배치

`DataCollatorForSupervisedDataset`이:

- `input_ids`/`labels`를 패딩, `attention_mask` 생성
- 이미지 텐서는 프레임 수가 가변이라 stack하지 않고 flatten (list)
- `rec_views`는 batch 차원으로 stack
- `bp_vqa`, `bp_rec`, `has_rec_views_mask`를 함께 전달

모델 forward에 들어가는 키: `input_ids, labels, attention_mask, images, image_sizes, modalities, rec_views, bp_vqa, bp_rec, has_rec_views_mask`

### 1-6. Forward Pass

**(a) 멀티모달 조립** — `prepare_inputs_labels_for_multimodal` (`llava_arch.py:292`)

- 텍스트 토큰 임베딩 + 이미지 토큰 자리에 SigLIP→projector로 만든 비전 임베딩 삽입.
- 학습 중 `bp_rec`이 True인 샘플(또는 추론 시 `use_camera_tokens`)에 대해 **`inject_rec_tokens_to_llm`** 호출 (`llava_arch.py:556`).

**(b) 카메라 토큰 주입** — `inject_rec_tokens_to_llm` (`vggt_reconstructor.py:253`)

- `camera_tokens_mode="camera_tokens"`, `camera_tokens_place="append_to_frame"`.
- 프레임 임베딩을 `[num_frames, tokens_per_frame, dim]`로 보고, **각 프레임 끝에 카메라 토큰 1개를 append**:
  - `token_num=2`: **첫 프레임은 `camera_tokens[0]`, 나머지 프레임은 모두 `camera_tokens[1]`** (`vggt_reconstructor.py:276-282`) — 첫 프레임을 기준(canonical) 으로 두기 위함.
  - 카메라 토큰 라벨은 모두 `IGNORE_INDEX` (언어 loss에서 제외).
- `info_dict`에 토큰 위치/프레임 수 기록 → 나중에 hidden state에서 다시 뽑을 때 사용.

**(c) LLM forward** — `llava_qwen.py:142` `llm_forward`

- Qwen2.5로 forward, `output_hidden_states=True`(rec_head가 hidden state를 써야 하므로).
- **VQA loss**: `labels`(answer 토큰만 유효)로 cross-entropy (`llava_qwen.py:314-322`).

**(d) pose head forward + rec loss** — `llava_qwen.py:205-221`

- `bp_rec=False`인 샘플은 `torch.where`로 hidden state gradient를 끊음(`llava_qwen.py:196-203`).
- `rec_head(rec_outputs, info_dict_list, rec_views)` 호출:
  1. `forward_projector` → hidden state에서 프레임별 카메라 토큰 위치를 슬라이스 (`prepare_camera_tokens_for_per_frame`), `projector_list`로 `rec_embed_dim=2048`로 투영.
  2. `forward_rec_heads` → **CameraHead**(trunk depth 4, iterative refinement)가 프레임별 **pose encoding** 예측.
  3. `forward_loss` → `MultitaskLoss` 계산.

### 1-7. Loss 정의 (정확한 수식)

**Pose encoding 포맷 = `absT_quaR_FoV` (9차원)** (`vggt/utils/pose_enc.py`):

- `[0:3]` 절대 translation
- `[3:7]` 회전 quaternion
- `[7:9]` field-of-view (fov_h, fov_w)

**Camera loss** (`loss.py:82` `compute_camera_loss` → `camera_loss_single`):

- GT extrinsic/intrinsic → `extri_intri_to_pose_encoding`로 9D GT 인코딩 변환.
- 손실 타입 `l1` (코드 주석: 논문은 smooth-l1이지만 l1이 더 안정적이어서 l1 사용):
  - `loss_T = |pred[:3] - gt[:3]|` (clamp max=100 후 mean)
  - `loss_R = |pred[3:7] - gt[3:7]|`
  - `loss_FL = |pred[7:] - gt[7:]|`
- CameraHead가 **여러 stage(iterative)** 를 내놓고, 후반 stage에 더 큰 가중치(`gamma^(n-i-1)`, **gamma=0.6**)로 평균. 예: 4 stage면 `[0.216, 0.36, 0.6, 1.0]`.
- 컴포넌트 가중: `weight_trans=1.0`, `weight_rot=1.0`, `weight_focal=0.5`
  - `camera_loss = loss_T·1.0 + loss_R·1.0 + loss_FL·0.5`
- pose head 출력 activation: translation/quaternion은 `linear`, **FoV는 `relu`**(음수 시야각 방지).
- **유효 프레임 마스킹**: `point_masks` 기준 유효 3D 포인트가 100개 미만인 프레임은 loss에서 제외.
- **비-metric 데이터(`is_metric_scale=False`, ViPE)**: scale-only Sim(3) 정렬로 스케일 모호성 보정 (`compute_scale_only_alignment`, stop-gradient) — translation을 GT 스케일에 맞춤 (`loss.py:158-177`).

**MultitaskLoss 합산** (`loss.py:36`):

```
objective = camera_loss · rec_camera_loss_weight(5.0)   [+ depth_loss·1.0 if enable_depth]
```

(이 7B 스크립트는 depth/point off → camera만)

**최종 total loss** (`llava_qwen.py:221`):

```
total = VQA_CE_loss  +  rec_loss_weight(0.2) · objective
      = VQA_CE_loss  +  0.2 · (5.0 · camera_L1_loss)
```

- rec 데이터가 없는 샘플은 `dummy_rec_loss = sum(params)*0`로 그래프만 유지 (`llava_qwen.py:229-231`).

### 1-8. 최적화 설정

- DeepSpeed ZeRO-3 (`zero3_stage1.json`), `bf16`, `tf32`
- global batch 256, per-device 1 → grad accumulation 자동 계산
- cosine LR, warmup 0.03, **1 epoch**, gradient checkpointing, `torch_compile inductor`, `model_max_length 32768`

---

## 2. 추론 (Inference) Flow

진입: `llava_qwen.py:344` `generate()` (eval 시 `eval_config.json`의 `use_camera_tokens=true` 적용)

1. **입력 조립**: `prepare_inputs_labels_for_multimodal`로 텍스트 + 128프레임 비전 임베딩 조립. 학습과 동일하게 **`inject_rec_tokens_to_llm`이 프레임마다 카메라 토큰을 append** (`llava_arch.py:556`의 추론 분기 `use_camera_tokens`). 토큰 위치는 `self.info_dict_list`에 저장.
2. **텍스트 생성(VQA)**: `super().generate(...)`로 Qwen2.5가 답변 텍스트를 autoregressive 생성.
3. **포즈 예측(1회)**: forward 도중 `use_camera_tokens`가 켜져 있으면, hidden state에서 카메라 토큰을 뽑아 `rec_head`를 **한 번만** 실행해 pose encoding 예측 → `self.rec_head_preds`에 저장 (`llava_qwen.py:233-243`). 추론 모드에서는 loss 없이 `predictions`만 반환(`vggt_reconstructor.py:236-237`).
4. **반환**: `return generation_output, rec_head_preds` (`llava_qwen.py:380`)
   - `generation_output`: VQA 텍스트 답변
   - `rec_head_preds["pose_enc"]`: per-frame pose encoding → `pose_encoding_to_extri_intri`로 **카메라 extrinsic(translation+rotation) + intrinsic(FoV)** 복원 (principal point는 이미지 중심으로 가정) → **streaming camera pose estimation**.

즉, **단일 forward로 "spatial VQA 답변"과 "프레임별 카메라 궤적"을 동시에 산출**합니다. 포즈 평가는 `cambrianp/eval/relpose/`(evo 기반 relative pose error)로 수행됩니다.

---

## 3. 요약 다이어그램

```
[128 frames] ──SigLIP──projector──► frame embeds ──┐
                                                    ├─► [..., frame_i, CAM_token_i, ...] ──► Qwen2.5 LLM
[question text] ──embed──────────────────────────► ┘                                          │
                                                              ┌───────────────────────────────┤ hidden states
                                  VQA CE loss ◄── lm_head ◄───┘                                │
                                                                                              ▼
                                          camera tokens 위치 슬라이스 ──projector──► CameraHead
                                                                                              │
                                                              pose_enc (absT_quaR_FoV, 9D) ◄──┘
                                                                                              │
   train: L1 camera loss ×5.0 ×0.2 ─► total = VQA_CE + 0.2·(5.0·camera_loss)                  │
   infer: pose_encoding_to_extri_intri ─► per-frame camera trajectory ◄──────────────────────┘
```

---

## 4. config 의존 주의사항

코드에는 `VGGTAggProjector`(frame+global attention)와 depth/point loss 경로가 존재하지만, 공개된 `Cambrian-P-7B.sh`에서는:

- `--rec_projector_type mlp1x_gelu` → **VGGTAggProjector가 아니라 단순 MLP projector** 사용
- `--enable_depth False --enable_point False` → **depth/point head·loss 모두 비활성**, camera loss만 동작
- `enable_depth=False`라서 `num_intermediate_layers`가 1로 강제 → **마지막 layer hidden state 1개만** projector→CameraHead로 전달

즉, **이 릴리스 스크립트의 실효 학습 목표는**:

```
total = VQA_CE_loss + 0.2 · [ 5.0 · (|ΔT|·1.0 + |Δquat|·1.0 + |ΔFoV|·0.5) ]   (L1, multi-stage γ=0.6)
```

depth/point/aggregator 경로는 다른 config(예: depth 켜기, agg projector)에서 활성화되는 확장 옵션입니다.
