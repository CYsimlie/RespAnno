# RespAnno 项目功能与测试大纲

> 用途：作为 SoftwareX 论文撰写的提示词素材，供其他 LLM 使用。
> 项目：RespAnno — Interactive Respiratory Sound Annotation Tool with ML-Assisted Labeling
> 版本：v1.6.6 | 语言：Python ≥ 3.9 | 许可：MIT | 代码量：~7,900 行（后端 5,500 + GUI 2,400）

---

## 一、项目总体架构

### 分层设计

```
1.6.6.py (主入口, 2443行, 1个类AudioViewer)
  │
  └─► respanno/  (纯Python后端, 33个.py文件, ~5500行)
        ├── audio/      — 音频预处理层
        ├── dsp/        — 数字信号处理层
        ├── ml/         — 机器学习层（6模块）
        ├── labels/     — 标注IO层
        └── gui/        — 可复用GUI组件（9个子模块）
```

### 模块清单与功能描述

#### 1. `respanno/audio/preprocessing.py` (308行)
音频预处理，零GUI依赖。

| 函数 | 功能 |
|------|------|
| `load_audio_file(path, target_sr)` | 加载WAV文件，可选重采样回目标采样率 |
| `get_original_sr(path)` | 读取WAV原始采样率（不加载数据） |
| `validate_preprocessing_config(cfg)` | 校验并规范化预处理配置字典 |
| `compute_target_sr(cfg, orig_sr)` | 计算有效目标采样率（关闭重采样时返回None） |
| `apply_butter_filter(audio, sr, cfg)` | Butterworth带通/低通/高通滤波，支持零相位(filtfilt) |
| `summarize_preprocessing(cfg)` | 生成预处理配置的人类可读摘要字符串 |

默认配置：4000 Hz重采样，滤波默认关闭。

#### 2. `respanno/dsp/spectrogram.py` (243行)
STFT频谱图计算与显示着色。

| 函数 | 功能 |
|------|------|
| `compute_stft_db(audio, sr, n_fft, hop_length, f_max)` | 计算STFT幅度谱(dB)，裁剪到 f ≤ f_max |
| `limit_frequency_range(S_db, freqs, f_max)` | 按频率上限裁剪频谱图 |
| `decimate_spec_for_display(spec, max_time_bins, max_freq_bins)` | 抽稀频谱图用于显示（不修改原始数据） |
| `get_palette_256(cmap)` | 返回256色查找表（Heatmap仿viridis / Grayscale） |
| `colorize_spectrogram(spec, cmap, vmin, vmax)` | 将2D频谱图映射为uint8 RGB图像 |
| `compute_spectrogram_display(audio, sr, config)` | 端到端：计算STFT→抽稀→着色，返回完整结果字典 |
| `compute_stft_frame_times(audio_length, sr, hop_length)` | 计算STFT帧中心时间戳 |

默认参数：n_fft=256, hop_length=64, f_max=2000Hz, cmap="Heatmap".

#### 3. `respanno/dsp/fft.py`
FFT幅度谱计算。

| 函数 | 功能 |
|------|------|
| `compute_fft(audio, sr, n_fft, f_max)` | 计算实数FFT幅度谱，裁剪到 f ≤ f_max |

#### 4. `respanno/dsp/features.py` (557行)
56维短时特征计算。

**7个时域特征：**
- RMS能量 (RMS Energy)
- 短时能量 (Short-Time Energy)
- 过零率 (Zero-Crossing Rate)
- Teager能量 (Teager Energy)
- Teager能量过零率 (Teager ZCR)
- 峰度 (Kurtosis)
- 偏度 (Skewness)

**30个频谱特征：**
- 谱质心 (Spectral Centroid)
- 谱带宽 (Spectral Bandwidth)
- 谱滚降 (Spectral Roll-off)
- 谱平坦度 (Spectral Flatness)
- 谱衰减 (Spectral Decrease)
- 谱斜率 (Spectral Slope)
- 谱峰度/偏度 (Spectral Kurtosis/Skewness)
- 谱熵 (Spectral Entropy)
- 谱通量 (Spectral Flux)
- 谱峰 (Spectral Crest)
- 能量集中度 (Energy Concentration)
- 子带能量比 (Sub-band Energy Ratios, 8个频带)
- F0估计与置信度 (F0 Estimate & Confidence)
- MFCC统计量 (均值/标准差, 2×5维)

**19个自相关特征(COR)：**
- 基于自相关曲线的波形周期性、峰值延迟、峰值间距离等

核心函数：

| 函数 | 功能 |
|------|------|
| `frame_signal(audio, sr, n_fft, hop_length)` | 反射填充帧分解（与librosa STFT语义对齐） |
| `compute_time_domain_features(audio, sr, n_fft, hop_length)` | 计算7个时域特征 |
| `compute_spectral_features(audio, sr, n_fft, hop_length, f_max)` | 计算30+19=49个频谱+COR特征 |
| `compute_short_time_features(audio, sr, n_fft, hop_length, f_max)` | 计算全部56个特征，返回时间轴+特征字典 |

#### 5. `respanno/labels/annotation_io.py` (517行)
标注数据IO（CSV/TXT/JSON读写）。

标准标注格式：
```python
{"start": float, "end": float, "label": str, "source": str}
```
source取值：manual / ml / auto_accepted / auto_edited / merged / merged_threshold

| 函数 | 功能 |
|------|------|
| `normalize_annotation(item)` | 将3/4元组旧格式转为标准dict |
| `parse_annotation_row(row, config)` | 按列映射解析CSV行 |
| `_detect_delimiter(line)` | 自动检测CSV分隔符（逗号/制表符/分号） |
| `read_annotations(path, config)` | 读取CSV/TXT标注文件 |
| `read_annotations_from_json(path, config)` | 读取JSON标注文件 |
| `write_annotations(path, ann_list)` | 导出标注到CSV文件 |

支持列映射：start_col/end_col/label_col/source_col（1-based），支持skip_header_lines。

#### 6. `respanno/labels/events_importer.py` (209行)
WAV同名_events文件自动导入。

类 `EventsFileIndexer`：
- 扫描目录，建立 `<wav_base>_events.(csv|txt|json)` 索引缓存
- 加载WAV时自动匹配并导入对应的事件标注文件
- 支持多种文件格式和列映射配置

#### 7. `respanno/ml/label_taxonomy.py` (101行)
标签分类体系。

将标签路由到三条ML管线之一：

| 管线 | 典型标签 | 处理方法 |
|------|---------|---------|
| `phase` (呼吸时相) | Inspiration, Expiration, Pause, 吸气, 呼气 | HSMM Viterbi后处理 |
| `abnormal_sound` (异常音) | Wheeze, Crackles, Rhonchi, Stridor, Pleural Rub | LightGBM二分类 |
| `other_event` (其他事件) | Speech, Cough, Noise, 语音, 咳嗽 | LightGBM二分类（与abnormal_sound共用逻辑） |

| 函数 | 功能 |
|------|------|
| `label_kind(label)` | 将标签字符串路由到管线类型 |
| `clear_ml_annotations(viewer, label)` | 删除指定标签的所有ML标注 |

#### 8. `respanno/ml/classifier.py` (380行)
LightGBM二分类器（适用于abnormal_sound和other_event管线）。

| 函数 | 功能 |
|------|------|
| `train_event_model(viewer, label, min_pos_frames, neg_pos_ratio, random_state, model_kind)` | 训练二分类器：构造正/负帧→特征选择(MI)→StandardScaler→LightGBM→阈值优化(F1-max)→存储到viewer.ml_models |
| `apply_event_model(viewer, label, min_dur_sec, expected_model_kinds)` | 在未审阅区域推理：加载模型→预测→合并连续帧→最小持续时间过滤→创建ML标注 |

训练指标：Accuracy, Specificity, Balanced Accuracy, MCC, AUROC, AUPRC, Brier Score, Confusion Matrix.

#### 9. `respanno/ml/phase_model.py` (430行)
呼吸时相模型（Inspiration/Expiration/Pause管线）。

| 函数 | 功能 |
|------|------|
| `train_phase_model(viewer, label, ...)` | 训练多类LightGBM分类器→HSMM Viterbi后处理→存储到viewer.ml_models |
| `apply_phase_model(viewer, label, min_dur_sec)` | 在未审阅区域推理：分类→HSMM Viterbi解码→生成时相片段 |

#### 10. `respanno/ml/hsmm.py` (362行)
HSMM（隐半马尔可夫模型）工具。

| 函数 | 功能 |
|------|------|
| `estimate_hop_sec(times, sr, hop_length)` | 估计帧步长（优先级：sr+hop > 中值时间差 > 0.05s） |
| `estimate_breath_cycle_sec(seg_I, seg_E)` | 从人工标注估计呼吸周期 |
| `build_hsmm_prior_from_prefix_labels(y_prefix, classes_, state_id_to_name, hop_sec, cycle_sec)` | 从已标注帧构建HSMM时长先验(dmin/dmax) |
| `build_hsmm_log_trans(state_names)` | 构建HSMM对数转移矩阵（2状态：互转+自环；3状态含Pause） |
| `hsmm_viterbi(log_emit, dmin, dmax, log_trans, log_pi)` | HSMM Viterbi解码器（显式时长建模） |
| `state_seq_to_segments(times, idx_unr, z_state_ids, target_state_id, min_dur_sec)` | 将Viterbi状态序列转为(start, end)片段 |

#### 11. `respanno/ml/frame_labels.py`
ML训练的帧级标签构造。

| 函数 | 功能 |
|------|------|
| `_iter_reviewed_annotations(annotations)` | 遍历已审阅标注（manual/auto_accepted/auto_edited/merged） |
| `get_manual_segments(annotations, label)` | 获取指定标签的所有已审阅片段 |
| `get_reviewed_prefix(annotations)` | 返回已审阅区域的最大结束时间 |
| `build_frame_labels(annotations, times, label, neg_segments, neg_margin)` | 构造帧级标签向量y（1=正样本, 0=安全负样本, -1=忽略） |

#### 12. `respanno/ml/service.py` (229行)
ML训练/推理调度器（管线分发入口）。

类 `MLService`：
- 维护标签分类常量（PHASE_LABELS / OTHER_EVENT_LABELS / ABNORMAL_SOUND_KIND等）
- `train_model_for_label(label, ...)` → 根据label_kind分发到对应管线
- `apply_model_for_label_on_unreviewed(label, ...)` → 根据label_kind分发到对应管线
- 所有HSMM辅助方法（_estimate_hop_sec等）也在此集中

#### 13-21. `respanno/gui/` (9个GUI子模块)

| 模块 | 类 | 功能 |
|------|-----|------|
| `dialogs/settings_dialog.py` | SettingsDialog | STFT/显示/预处理/特征选择/自动导入配置对话框 |
| `dialogs/annotation_label_dialog.py` | AnnotationLabelDialog | 标注标签选择对话框（9预设+自定义输入） |
| `dialogs/loop_player.py` | LoopPlayer | 片段循环播放对话框 |
| `spans/box_span.py` | BoxSpan | 标注可视化矩形条（pg.RectROI子类，支持拖拽/编辑/颜色） |
| `spans/span_label_item.py` | SpanLabelItem | BoxSpan上的文字标签项 |
| `views/annot_view_box.py` | AnnotViewBox | 标注面板ViewBox（点击选择+拖拽标记） |
| `views/wave_view_box.py` | WaveViewBox | 波形面板ViewBox |
| `widgets/clickable_slider.py` | ClickableSlider | 可点击跳转的QSlider |
| `widgets/color_bar.py` | ColorBarWidget + HistogramWidget | 配色条和STFT直方图 |
| `widgets/color_check_delegate.py` | ColorCheckDelegate | 带颜色的勾选框代理 |

---

## 二、GUI 功能清单（AudioViewer 主窗口）

### 菜单栏

#### File菜单
- Import Audio (Ctrl+O)：加载WAV文件
- Import Annotations (Ctrl+I)：导入标注文件
- Export Annotations (Ctrl+E)：导出标注到CSV/TXT/JSON
- Exit (Ctrl+Q)：退出

#### Settings菜单
- Settings (Ctrl+P)：打开配置对话框
- Auto-import matching _events annotations：切换同名事件文件自动导入开关

#### Help菜单
- About：关于信息

### 工具栏按钮

- Import WAV File：加载音频
- Previous (↑) / Next (↓)：切换目录内上一个/下一个WAV
- Play / Pause：音频播放/暂停
- Settings：打开配置
- Export Annotations：导出标注
- Import Annotations：导入标注
- Switch Spectrum：切换STFT/FFT/短时特征三个视图
- Annotation Legend：显示标注颜色图例
- 配色下拉框：Heatmap / Grayscale切换

### ML工具栏
- 标签下拉框：选择要训练/标注的标签（9个内置标签的英文名）
- Train Model：用已审阅标注训练当前标签的帧级分类器
- Auto-label Unreviewed：用训练好的模型自动标注未审阅区域
- Annotation Legend：颜色图例

### 键盘快捷键

| 快捷键 | 功能 |
|--------|------|
| Ctrl+O | 导入WAV |
| Ctrl+E | 导出标注 |
| Ctrl+I | 导入标注 |
| Ctrl+P | 打开Settings |
| Ctrl+Z | 撤销上次标注操作 |
| Space | 播放/暂停音频 |
| Left / Right | 快退/快进 1 秒 |
| Up / Down | 上一首/下一首 |
| Delete / Backspace | 删除选中标注 |
| Ctrl+A | 认可（接受）选中的 ML 标注 |
| Ctrl+T | 训练当前 ML 标签的模型 |
| Ctrl+M | 对当前 ML 标签自动标注未审阅区域 |
| Enter | 提交 span 编辑 |
| Esc | 取消 span 编辑 |
| F1 | 关于 |
| Ctrl+Q | 退出 |

### 交互方式
- **单击选中**：单击标注条选中（高亮），之后可用 Delete/Ctrl+A 操作
- **拖拽标注**：在标注面板拖拽鼠标创建标注区间→弹出标签对话框→选择预设或输入自定义标签
- **编辑标注**：双击BoxSpan进入编辑模式→拖拽手柄调整边界→Enter提交 / Esc取消
- **右键菜单**：播放片段、删除标注、修改source类型
- **撤销**：Ctrl+Z撤销删除/编辑/source修改

### 标注系统
- **9种内置标签**：哮鸣音(Wheeze)、爆裂音(Crackles)、摩擦音(Pleural Rub)、哼鸣音(Rhonchi)、喘息音(Stridor)、语音(Speech)、咳嗽(Cough)、呼气(Expiration)、吸气(Inspiration)
- **固定颜色映射**：每个内置标签有固定颜色（红/蓝/绿/紫/橙/棕/粉/灰/青绿）
- **自定义标签**：通过对话框输入任意文本，自动从色板分配颜色
- **三轨分轨**：最多3行防遮挡布局
- **Source溯源**：每条标注记录来源（manual/ml/auto_accepted/auto_edited/merged）

### 可视化面板
- **STFT频谱图**：Heatmap/Grayscale配色，可配置色阶上下限
- **FFT幅度谱**：单帧频谱视图
- **短时特征曲线**：最多显示5条特征曲线（0-1归一化叠加），从56个特征中选择
- **波形图**：抽稀显示，标注高亮
- **标注轨道**：彩色BoxSpan显示

### 音频预处理配置
- 重采样开关（默认4000 Hz）
- Butterworth滤波器（低通/高通/带通，可配置截止频率、阶数、零相位）
- 配置在下次load_audio时生效

### 标注导入/导出
- 支持CSV/TXT/JSON格式
- 自动检测分隔符
- 可配置列映射（start_col/end_col/label_col/source_col，1-based）
- 可配置跳过表头行数
- 自动导入同名_events文件（可开关）

### 性能优化
- 切换文件时仅加载波形+STFT，FFT和短时特征懒加载
- 长音频波形抽稀显示（max 50000点）
- 频谱图抽稀显示（max 2500时间bin × 512频率bin）

---

## 三、ML 管线详解

### 管线分发逻辑（MLService）

```
训练/推理请求
  │
  ├─► label_kind(label) ──► phase ──► HSMM Viterbi后处理
  │                        ├─► other_event ──► LightGBM二分类
  │                        └─► abnormal_sound ──► LightGBM二分类 (默认)
```

### 数据流（以Wheeze标签为例）

```
1. 用户人工标注Wheeze区域 → 存储在annotations中(source="manual")
2. 用户标注非Wheeze区域或删除误标 → 进入neg_segments硬负样本
3. 用户点击"Train Model"：
   a. 从annotations提取已审阅的Wheeze片段(正样本)
   b. 从neg_segments提取Wheeze的删除区(硬负样本)  
   c. 从STFT帧中采样安全负样本(neg_pos_ratio=5:1)
   d. 计算56维特征矩阵
   e. MI特征选择(SelectKBest, top 20)
   f. StandardScaler归一化
   g. LightGBM训练 + F1-max阈值优化
   h. 存储到viewer.ml_models[label]
4. 用户点击"Auto-label Unreviewed"：
   a. 加载已训练模型
   b. 在未审阅区域逐帧预测
   c. 合并连续正预测帧为片段
   d. 最小持续时间过滤(min_dur_sec=0.05)
   e. 去重（与已有标注合并）
   f. 添加ML标注(source="ml")
```

### 呼吸时相管线（特殊处理）

```
1. 训练时：多类LightGBM分类器(3类：Insp/Exp/Pause) + HSMM时长先验学习
2. 推理时：LightGBM预测 → 对数发射概率 → HSMM Viterbi解码(显式时长约束)
3. HSMM约束确保生理合理性：
   - 不允许Insp→Insp自跳（呼吸时相必须交替）
   - 时长先验从人工标注学习
   - 2状态(无Pause)或3状态(含Pause)自动检测
```

### ML模型存储结构

```python
viewer.ml_models = {
    "Wheeze": {
        "clf": Pipeline(...),          # LightGBM sklearn Pipeline
        "threshold": 0.42,              # F1最优阈值
        "feature_names": ["RMS", ...],  # 特征名列表
        "metrics": {"f1": 0.85, ...},  # 训练指标
        "kind": "abnormal_sound",       # 管线类型
        "random_state": 42,             # 复现种子
    },
    ...
}
```

---

## 四、测试覆盖情况（427个测试，22个测试文件）

### 测试文件清单

| 测试文件 | 测试数 | 覆盖范围 |
|---------|--------|---------|
| `test_module_imports.py` | 82 | 所有模块可导入性、公共符号验证（每个子模块一个测试类） |
| `test_annotation_roundtrip.py` | 43 | CSV/TXT/JSON标注IO往返保真度、分隔符检测、列映射、跳过表头 |
| `test_label_taxonomy_basic.py` | 48 | 标签→管线路由（phase/event/abnormal分类的正确性和边界情况） |
| `test_preprocessing_basic.py` | 26 | 滤波参数验证、配置校验、重采样、原始采样率保留 |
| `test_gui_static_integration.py` | 26 | AST级别GUI导入验证、静态分析检查 |
| `test_annotation_quality.py` | 21 | Source溯源正确性、去重逻辑、fixture完整性、导出后比对 |
| `test_frame_labels_basic.py` | 19 | 帧级训练标签构建：正/负/忽略标签的正确性 |
| `test_spectrogram_basic.py` | 19 | STFT计算、抽稀、着色、色板生成、f_max裁剪 |
| `test_hsmm_basic.py` | 18 | HSMM先验构建(2/3状态)、Viterbi解码正确性、hop估计 |
| `test_features_basic.py` | 15 | 56维短时特征计算（时域+频谱+COR）、输出维度验证 |
| `test_events_importer_basic.py` | 15 | WAV匹配_events文件自动导入、索引缓存、解析正确性 |
| `test_classifier_training_basic.py` | 13 | LightGBM二分类器训练、特征选择、阈值优化、指标输出 |
| `test_fft_basic.py` | 12 | FFT幅度谱计算、f_max裁剪 |
| `test_reproducibility.py` | 12 | 全管线确定性验证（相同输入+相同种子=相同输出） |
| `test_phase_model_basic.py` | 12 | HSMM时相模型训练（2/3状态自动检测） |
| `test_performance_baseline.py` | 9 | 吞吐量、延迟、内存基线（性能回归检测） |
| `test_e2e_ml_pipeline.py` | 8 | 端到端：音频→特征→训练→预测→结果验证 |
| `test_phase_apply_basic.py` | 8 | HSMM Viterbi解码 + 片段生成正确性 |
| `test_classifier_apply_basic.py` | 7 | ML推理：预测→合并→最小持续时间过滤→去重 |
| `test_icbhi_compatibility.py` | 6 | ICBHI 2017数据集格式与命名约定兼容性 |
| `test_roundtrip_workflow.py` | 4 | 完整工作流：WAV→预处理→标注→导出→重新导入验证 |

### 测试分类

| 类别 | 测试数 | 说明 |
|------|--------|------|
| **单元测试** | ~350 | 每个模块函数的独立测试 |
| **集成测试** | ~50 | 模块间交互（IO往返、ML管线端到端） |
| **性能测试** | 9 | 吞吐量/延迟/内存基线 |
| **兼容性测试** | 6 | ICBHI 2017格式兼容 |
| **静态测试** | 26(+82) | AST分析 + 模块导入性验证 |

---

## 五、依赖库

| 库 | 最低版本 | 用途 |
|----|---------|------|
| PyQt5 | 5.15 | GUI框架 |
| pyqtgraph | 0.13 | 交互式科学绘图 |
| numpy | 1.21 | 数值计算 |
| scipy | 1.7 | 信号处理、滤波器 |
| librosa | 0.9 | 音频IO、重采样、STFT |
| scikit-learn | 1.0 | StandardScaler、特征选择、评估指标 |
| lightgbm | 3.3 | 梯度提升树分类器 |
| sounddevice | 0.4 | 音频回放 |

---

## 六、与相关工作的区别

| 工具 | 定位 | 与RespAnno对比 |
|------|------|---------------|
| Audacity | 通用音频编辑器 | 有频谱图但无呼吸领域标注工作流、无ML辅助 |
| ELAN | 多媒体标注 | 支持分层标注但无呼吸领域特征、无ML集成 |
| LungPass | 深度学习分类 | 面向分类而非标注；RespAnno可作为其标注前端 |
| ICBHI 2017 | 标准数据集 | RespAnno的标注格式与ICBHI事件标注范式兼容 |
