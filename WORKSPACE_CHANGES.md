# Workspace 代码改动说明（Seamless 双人对话适配）

**概览**
本次改动在现有 DIM 代码基础上加入了 Seamless 双人对话数据集适配，核心目标是支持 30 fps + 48 kHz 音频，并把 motion 维度改为 **51 维（exp50 + jaw1）**。同时将训练/评测入口改为可通过配置文件选择数据集与维度，避免 56 维的硬编码。

**新增文件**
- `seamless_preprocessing.py`  
  读取 `dataset_jsons/seamless_mini.json`，抽取 `exp50 + jaw1` 的 motion，提取 HuBERT 音频特征（16 kHz），并对齐到 30 fps 帧数；输出 `data/seamless_processed/manifest.json` 以及 `*_motion51.npy` 与 `*_audio_hubert.npy`。
- `code/dataset/seamless.py`  
  新增 Seamless 专用 Dataset 与 DataLoader：
  `SeamlessDyadicDataset`（dyadic 训练/评测）与 `SeamlessVQDataset`（VQ 训练），支持随机窗口采样、整段推理、speaker/listener 方向样本、speaker/listener ID 映射与 pad collate。
- `code/config_seamless.yaml`  
  Seamless 数据集配置模板，包含 `manifest_path`、`window_frames`、`fps`、`audio_dim`、`pose_dim/exp_dim` 与 VQ checkpoint 路径字段。

**训练/测试入口改动**
- `code/train_vq.py`  
  增加基于 `cfg.dataset` 的数据集选择逻辑，新增 Seamless VQ 数据加载路径。
- `code/train_s2s_pretrain.py`  
  新增 `--config` 解析；根据配置选择数据集（seamless/candor/vico/lm_listener）；支持在 config 中指定 speaker/listener VQ 模型配置；保存路径改为 `pretrain_ckpt` 可配置字段。
- `code/test_s2s_pretrain.py`  
  同步加入 `--config` 解析与 dataset 选择；支持从 config 指定 checkpoint；评测流程保持一致。

**模型与维度处理调整**
- `code/seq2seq_pretrain.py`  
  新增 `infer_motion_dims`，根据 `pose_dim/exp_dim` 或 `in_dim` 自动推断 motion 维度（兼容 56/51）。  
  `SLM` / `SLMFT` / `SpeakerSLMFT` 支持传入自定义 config 路径并读取 `speaker_vq_ckpt` / `listener_vq_ckpt`。  
  连续损失 `forward_continuous_loss` 改为按 `pose_dim` 与 `exp_dim` 计算，支持 51 维模式。
- `code/x_engine_pt.py`  
  所有 `src` 切分改为动态 `visual_dim = src.shape[2] - 768`，避免硬编码 56。
- `code/metrics/loss.py`  
  VQ AV loss 的视觉维度改为动态 `pred.shape[2] - 768`。

**评测与指标适配**
- `code/mymetrics.py`  
  新增维度自动推断；FID/PFID/MSE/VAR/PCC/STS 等指标支持 51 维（`jaw`）与 56 维（`pose`）两种形态。
- `code/metrics/eval_utils.py`  
  `calcuate_sid` 增加 `pose_dim` 参数，避免固定 6 维 pose。

**当前 workspace 中的未跟踪产物（非代码）**
以下文件/目录为运行产物或本地文件，未纳入版本控制：
- `best_model_pretrain.pt`
- `data/`
- `runs_vico_pretrain_listener_MAX/`
- `prompt.txt`

如需清理或加入 `.gitignore`，可以再告诉我你希望的处理方式。
