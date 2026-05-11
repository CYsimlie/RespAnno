# RespAnno — Respiratory Sound Annotation Tool

基于 PyQt5 + pyqtgraph + librosa + scipy + LightGBM 的医学呼吸音标注软件。

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
# 安装依赖
pip install -r requirements.txt

# 启动（使用 legacy 主程序）
python legacy/1.6.6.py

# 或通过包入口启动
python -m respanno.main
```

## 项目结构

```
SoftwareX/
├── legacy/                 # 原始单文件程序（冻结，不再编辑）
├── respanno/               # 工程化包（逐步从 legacy 抽取模块）
│   ├── main.py             # 启动入口
│   ├── gui/                # GUI 组件（对话框、部件、视窗、标注条）
│   ├── audio/              # 音频加载与预处理
│   ├── dsp/                # 信号处理（STFT、FFT、短时特征）
│   ├── labels/             # 标签 IO（CSV/TXT/JSON 导入导出）
│   └── ml/                 # 机器学习（训练、推理、HSMM）
├── tests/                  # 测试
├── docs/                   # 文档
├── demo_data/              # 示例数据
├── examples/               # 使用示例
└── screenshots/            # 截图
```

## 依赖

- Python >= 3.9
- PyQt5, pyqtgraph
- numpy, scipy, librosa
- scikit-learn, lightgbm
- sounddevice

## 许可

MIT License
