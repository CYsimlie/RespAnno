# RespAnno — Respiratory Sound Annotation Tool

基于 PyQt5 + pyqtgraph + librosa + scipy + LightGBM 的医学呼吸音标注软件。

## 工程化状态

当前处于**模块抽离阶段**。已从 legacy 单文件中提取以下纯逻辑模块：

| 模块 | 位置 | 测试数 |
|------|------|--------|
| 标签 IO (CSV/TXT/JSON) | `respanno/labels/annotation_io.py` | 43 |
| 音频预处理 (重采样/滤波) | `respanno/audio/preprocessing.py` | 26 |
| 频谱图 (STFT/着色/抽稀) | `respanno/dsp/spectrogram.py` | 19 |
| HSMM (Viterbi/转移/先验) | `respanno/ml/hsmm.py` | 18 |
| 短时特征 (56 features) | `respanno/dsp/features.py` | 15 |

共计 **121 个测试**，全部通过。GUI 集成阶段尚未开始。

## 功能概览

- **WAV 音频读取与回放** — 支持载入、播放、上一首/下一首
- **预处理** — 可选重采样（默认 4000 Hz）+ Butterworth 滤波
- **时频分析** — STFT 频谱图、FFT 频谱、50+ 短时特征（时域/频域/自相关）
- **手动区间标注** — 鼠标拖选 + 预设标签类型（哮鸣音、爆裂音等 9 种）
- **标注编辑** — 双击进入编辑模式，拖动边界，Enter 提交 / Esc 取消
- **撤销** — Ctrl+Z 撤销删除与编辑
- **标签导入导出** — CSV / TXT / JSON 格式，自动匹配 `_events` 同名文件
- **ML 辅助标注** — LightGBM 帧级二分类 + HSMM 相位后处理（Inspiration/Expiration/Pause）
- **Source 追踪** — 每次标注记录来源（manual / ml / auto_accepted / auto_edited / merged）
- **三轨道分轨** — 避免重叠标注互相遮挡

## 快速开始

```bash
# 安装依赖（conda 环境）
conda env create -f environment.yml
conda activate respanno

# 或使用 pip
pip install -r requirements.txt

# 启动应用
conda run -n respanno python -m respanno.main

# 运行测试
conda run -n respanno python -m pytest tests -q
```

## 项目结构

```
SoftwareX/
├── 1.6.6.py                 # 原始单文件（冻结，不再编辑）
├── legacy/
│   └── 1.6.6.py             # 冻结副本
├── respanno/                # 工程化包
│   ├── main.py              # 启动入口 → legacy GUI
│   ├── labels/
│   │   └── annotation_io.py # CSV/TXT/JSON 读写 + 行解析
│   ├── audio/
│   │   └── preprocessing.py # Butterworth 滤波 + 重采样 + 元数据
│   ├── dsp/
│   │   ├── spectrogram.py   # STFT + 显示抽稀 + 着色
│   │   └── features.py      # 56 短时特征（时域/频域/COR）
│   ├── ml/
│   │   └── hsmm.py          # HSMM Viterbi + 先验 + 转移矩阵
│   └── gui/                 # GUI 组件（待接入）
├── tests/                   # 121 tests, 0 skipped
├── docs/                    # 架构 & 测试文档
├── demo_data/               # 示例数据
├── examples/                # 使用示例
└── screenshots/             # 截图
```

## 依赖

- Python >= 3.9
- PyQt5, pyqtgraph
- numpy, scipy, librosa
- scikit-learn, lightgbm
- sounddevice

## 许可

MIT License
