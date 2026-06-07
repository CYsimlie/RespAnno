"""Final comprehensive Chinese->English for 1.6.6.py.

Handles ALL comment categories:
  - Pure comment lines (# ...)
  - Inline comments (code # CN)
  - Docstrings

Never touches data lines (labels, feature names, etc.).
"""
import sys, re

C = re.compile(r'[一-鿿　-〿＀-￯]')

# Load translate function from v2
exec(open('scripts/_translate_v2.py', encoding='utf-8').read().split("if __name__")[0])

TARGET = '1.6.6.py'

# ── Additional phrases for the remaining Chinese in 1.6.6.py ─────────────
MORE_PHRASES = [
    # Docstrings that were missed
    ("加载单个 WAV 文件。", "Load a single WAV file."),
    ("设计原则：", "Design principles:"),
    ("默认以 4000 Hz 重采样作为分析基础采样率；", "Resample to 4000 Hz by default as the analysis base sample rate;"),
    ("切换文件时仅完成必要操作：读取音频、绘制抽稀波形、绘制 STFT、导入标签；", "When switching files, perform only essential operations: read audio, draw decimated waveform, draw STFT, import labels;"),
    ("FFT 与短时特征不预先计算，改为切换到对应页面时懒加载以提升切换速度。", "FFT and short-time features are not pre-computed; they are lazy-loaded when switching to the corresponding page to improve switching speed."),

    # More docstrings
    ("交互式删除 BoxSpan 时：默认将该段加入 neg_segments（仅用于训练的硬负样本），", "When interactively deleting a BoxSpan: by default, add the segment to neg_segments (hard-negative samples for training only),"),
    ("且支持 Ctrl+Z 撤销。", "and support Ctrl+Z undo."),
    ("程序内部清理（如清除 ML 预测结果）可传 record_negative=False, push_undo=False。", "Internal cleanup (e.g., clearing ML predictions) can pass record_negative=False, push_undo=False."),

    ("统一遍历「可用于训练 / 已审阅」的标注区间：", 'Unified iteration over "trainable / reviewed" annotation intervals:'),
    ("兼容 (start, end, label) 和 (start, end, label, source) 两种格式", "Compatible with both (start, end, label) and (start, end, label, source) formats"),
    ("三元组一律视为人工标注（source='manual'）", "3-tuples are always treated as manual annotations (source='manual')"),
    ("四元组根据 source 是否属于「已审阅集合」判定", '4-tuples are judged based on whether source belongs to the "reviewed set"'),
    ("说明：为兼容旧代码保留函数名 _iter_manual_annotations，", "Note: the function name _iter_manual_annotations is retained for backward compatibility,"),
    ("但其语义已升级为「reviewed annotations」。", 'but its semantics have been upgraded to "reviewed annotations."'),

    # ── Phrases that v2 partially matched ──
    ("构造 Qt plugins 目录的绝对路径（兼容不同 PyQt5 安装布局）", "Construct the absolute path to the Qt plugins directory (compatible with different PyQt5 installation layouts)"),
    ("若 Qt5 层级不存在，回退至无 Qt5 层级的路径", "If the Qt5-level directory does not exist, fall back to the path without the Qt5 level"),
    ("设置 QT 插件环境变量，确保 Qt 运行时能找到 platform plugins", "Set the QT plugin environment variable to ensure the Qt runtime can locate platform plugins"),
    ("当存在快捷键歧义时，也连接 activatedAmbiguously 以确保撤销生效", "When shortcut ambiguity exists, also connect activatedAmbiguously to ensure Undo works"),
    ("额外注册 QAction 形式的 Undo（更贴近 Qt 标准动作系统，某些场景下比 QShortcut 更可靠）", "Additionally register an Undo QAction (closer to the Qt standard action system; more reliable than QShortcut in some scenarios)"),
    ("应用级事件过滤器：处理需要复杂逻辑的快捷键（Delete/Ctrl+A/编辑模式等）", "Application-level event filter: handles shortcuts requiring complex logic (Delete, Ctrl+A, edit mode, etc.)"),
    ("监听 STFT 图鼠标移动，通过 ViewBox 矩形判断是否在绘图区域内", "Listen for mouse movement on the STFT plot; use the ViewBox rectangle to determine if the cursor is inside the plotting area"),
    ("标注区 Y 轴固定为 3 行高度", "Annotation area Y-axis is fixed at 3-lane height"),
    ("仅允许 X 轴缩放/拖动，禁用 Y 轴", "Allow only X-axis zoom/pan; disable the Y-axis"),
    ("隐藏 Y 轴（坐标线、刻度、标签均不显示）", "Hide the Y-axis (axis line, ticks, and labels are all hidden)"),
    ("时间轴与 STFT 同步缩放/平移", "Synchronize time-axis zoom/pan with the STFT plot"),
    ("去除多余边距，避免隐藏轴后仍留白", "Remove extra margins to avoid whitespace after hiding the axis"),
    ("禁止各面板被完全折叠", "Prevent any panel from being collapsed entirely"),
    ("使用内置标签列表的英文名作为 ML 标签选项", "Use the English names from the built-in label list as ML label options"),
    ("初始化 current_ml_label（若已有值且在列表中则沿用，否则取首个）", "Initialize current_ml_label (keep the existing value if it is in the list; otherwise use the first)"),

    # File menu etc
    ("创建菜单栏：File / Settings / Help。", "Create the menu bar: File / Settings / Help."),
    ("初始化 Machine Learning 工具栏：标签选择 + 训练 + 自动标注 + 负样本管理。", "Initialize the Machine Learning toolbar: label selection + training + auto-labeling + negative sample management."),
    ("下拉框标签变化时更新当前 ML 操作目标。", "Update the current ML operation target when the combo-box label changes."),
    ("训练按钮：对当前选中标签训练帧级模型。", "Train button: train a frame-level model for the currently selected label."),
    ("自动标注按钮：用已训练模型对当前标签的未审阅区域进行自动标注。", "Auto-label button: use the trained model to auto-label unreviewed regions for the current label."),
    ("清除当前标签的全部硬负样本。", "Clear all hard negative samples for the current label."),
    ("更新清除负样本按钮的 tooltip，显示当前标签的负样本数量。", "Update the Clear Negatives button tooltip to show the current label's negative sample count."),
    ("根据内置标签和颜色映射生成标注颜色图例。", "Generate the annotation color legend from built-in labels and color mappings."),
    ("优先使用内置颜色映射；大小写不一致时做不区分大小写的回退匹配。", "Prefer the built-in color mapping; fall back to case-insensitive matching when casing differs."),
    ("清空所有标注：移除可视化对象、清空数据结构、复位视图范围。", "Clear all annotations: remove visual objects, clear data structures, reset view ranges."),
    ("快退/快进 delta_sec 秒（供 Left/Right 快捷键调用）。", "Seek backward/forward by delta_sec seconds (called by the Left/Right shortcut keys)."),
    ("根据当前 WAV 实时更新默认导出文件名。", "Update the default export filename in real time based on the current WAV file."),
    ("生成导出对话框默认路径：目录跟随上次导出，文件名跟随当前 WAV。", "Generate the default path for the export dialog: directory follows the last export, filename follows the current WAV."),
    ("导出标注数据到 CSV 文件。", "Export annotation data to a CSV file."),
    ("删除一个标注。", "Delete an annotation."),
    ("撤销最后一次编辑操作（删除、改 source、编辑几何位置）。", "Undo the last edit operation (delete, source change, geometry edit)."),
    ("撤销一次删除：恢复 annotations 条目 + 重建可视化 + 移除负样本。", "Undo a single deletion: restore the annotations entry + rebuild visualization + remove the negative sample."),
    ("撤销一次几何编辑：回滚位置、文本、样式和高亮。", "Undo a single geometry edit: roll back position, text, style, and highlight."),
    ("撤销 source 变更：恢复旧 source 并重绘视觉样式。", "Undo a source change: restore the old source and redraw the visual style."),
    ("将机器标注标记为「已认可」，默认 source=auto_accepted，支持撤销。", "Mark a machine annotation as accepted; default source=auto_accepted; undoable."),
    ("记录当前进入编辑态的 BoxSpan（若已有其他 span 在编辑则先提交退出）。", "Record the BoxSpan currently entering edit mode (if another span is already in edit mode, commit and exit it first)."),
    ("退出编辑模式并清空编辑状态。", "Exit edit mode and clear the editing state."),
    ("编辑态：在状态栏显示当前选中 span 的起止时间。", "Edit mode: display the start and end times of the currently selected span in the status bar."),
    ("根据标签文本返回稳定的颜色：内置标签使用固定颜色，自定义标签使用自动配色。", "Return a stable color for a label text: built-in labels use fixed colors; custom labels use auto-assigned colors."),
    ("根据当前可见 BoxSpan 动态选择空闲轨道；优先使用编号最小的空轨。", "Dynamically select a free lane based on the currently visible BoxSpan instances; prefer the lane with the smallest index."),
    ("鼠标在 STFT 图上移动时显示 t/f 坐标；移出绘图区域则还原标题。", "Display time/frequency coordinates when the mouse moves over the STFT plot; restore the title when the mouse leaves the plotting area."),
    ("根据 self.selected_features 计算并绘制短时特征曲线（0-1 归一化叠加显示）。", "Compute and plot short-time feature curves based on self.selected_features (0-1 normalized, overlaid display)."),
    ("说明：该函数在主线程计算，仅在切换到特征页时懒加载；", "Note: this function computes on the main thread and is only lazy-loaded when switching to the features page;"),
    ("load_audio / 上一首 / 下一首不主动调用。", "it is not called proactively during load_audio / previous / next."),
    ("切换配色方案：不重新计算 STFT，仅对显示用谱图重着色。", "Switch the color scheme: do not recompute the STFT; only recolor the display spectrogram."),
    ("按 selected_names 顺序为每条特征曲线分配不重复的颜色。", "Assign a unique color to each feature curve in the order of selected_names."),
    ("应用级事件过滤器：处理需要复杂逻辑的快捷键。", "Application-level event filter: handles shortcuts that require complex logic."),
    ("包括：Ctrl+Z 撤销（兜底）、编辑模式 Enter/Esc、", "Includes: Ctrl+Z undo (fallback), edit-mode Enter/Esc,"),
    ("Delete 删除选中标注、Ctrl+A 认可 ML 标注。", "Delete to remove the selected annotation, Ctrl+A to accept an ML annotation."),
    ("只更新文件名，不决定目录。目录在导出时优先沿用上一次导出标注的目录。", "Only the filename is updated; the directory is not determined here. On export, the directory defaults to the last export directory."),
    ("命名规则：<当前wav文件名不含扩展名>_events.csv", "Naming convention: <current_wav_basename_without_extension>_events.csv"),
    ("直接按照代码中的预设标签顺序生成图例，不依赖当前是否已有标注。", "Generate the legend directly from the preset label order in the code, without depending on whether annotations currently exist."),
    ("补充当前文件中出现过的自定义标签。", "Supplement with any Custom labels that appear in the current file."),
    # ── Specific 1.6.6.py only ──
    ("—— 机器学习硬负样本管理器（仅用于训练，不导出，与标注显示无关）——", "Machine-learning hard-negative sample manager (training only; not exported; unrelated to annotation display)"),
    ("—— 撤销栈（支持删除、认可等编辑操作的回退）——", "Undo stack (supports rollback of delete, accept, and other edit operations)"),
    ("—— 多轨标注管理（最多 3 轨，仅用于标注视图）——", "Multi-lane annotation management (max 3 lanes; annotation view only)"),
    ("—— 编辑模式（双击 BoxSpan 进入）——", "Edit mode (double-click a BoxSpan to enter)"),
    ("—— STFT 显示参数（配色方案与显示窗口上下限）——", "STFT display parameters (color scheme and display window bounds)"),
    ("—— 短时特征曲线配色（预定义不重复色板，最多支持 10 条曲线）——", "Short-time feature curve color palette (predefined, non-repeating; supports up to 10 curves)"),
    ("—— 标注类型与颜色管理（内置呼吸音标签及对应颜色）——", "Annotation type and color management (built-in respiratory sound labels with corresponding colors)"),
    ("—— 音频加载预处理参数（默认启用 4000 Hz 重采样，滤波默认关闭以保持原始音频内容不变）——", "Audio loading preprocessing parameters (4000 Hz resampling enabled by default; filtering disabled by default to preserve original audio content)"),
    ("—— 同名事件文件自动导入（可开关）——", "Auto-import of matching events files (toggleable)"),
    ("—— 短时特征默认选择（可在 Settings 中修改）——", "Default short-time feature selection (configurable in Settings)"),
    ("—— 性能优化：切换文件时仅加载波形与 STFT，FFT 与短时特征改为懒加载——", "Performance optimization: when switching files, only load waveform and STFT; FFT and short-time features are lazy-loaded on demand"),
    ("—— 机器学习相关状态 ——", "Machine-learning state"),
    ("—— 短时特征页（仅负责显示，计算逻辑由独立模块处理）——", "Short-time features page (display only; computation logic is handled by a separate module)"),
    ("—— 页面栈：STFT / FFT / 短时特征 ——", "Page stack: STFT / FFT / Short-Time Features"),
    ("—— 垂直分割器：上（页面栈）| 中（波形）| 下（标注）——", "Vertical splitter: top (page stack) | middle (waveform) | bottom (annotations)"),
    # ── Section headers ──
    ("文件选择框", "file selection dialog"),
    ("备忘录小白框 (显示在条上方)", "Note label box (displayed above the bar)"),
    ("根据 label_color & text 更新 HTML", "Update HTML based on label_color & text"),
    ('标注条不直接显示文字，标签含义统一通过工具栏"Annotation Legend"查看', 'Annotation bar does not display text directly; label meanings are viewed via the toolbar "Annotation Legend"'),
    ("视觉编码：根据 source(manual/ml) 统一设置 pen/brush/label 样式", "Visual encoding: unify pen/brush/label styles based on source (manual/ml)"),
    ("主色 (边框/强调)", "Primary color (border / emphasis)"),
    ("应用到 ROI 本体", "Apply to ROI body"),
    ("同步 label 的视觉 (不改文字内容)", "Sync label visuals (not text content)"),
    ("同步频谱红条", "Sync spectrogram red bar"),
    ("同步导出缓存 (兼容 3/4 元组；三元组视为人工标注)", "Sync export cache (compatible with 3/4-tuples; 3-tuples treated as manual annotations)"),
    ('编辑态：保持"编辑态虚线"提示，不要被 source 样式覆盖', 'Edit state: keep "edit-mode dashed" notice; do not let source style override'),
    # ── Data annotation comments (safe - they explain colors, not are data) ──
    ("内置标签的固定颜色映射", "Fixed color mapping for built-in labels"),
    ("自定义标签的自动配色色板", "Auto-coloring palette for custom labels"),
    # Inline comment patterns
    ("单轨高度", "Single-lane height"),
    ("轨道间距", "Lane gap"),
    ("每轨占用区间 [(s, e), ...]", "Occupied intervals per lane [(s, e), ...]"),
    ("当前所有 BoxSpan", "All current BoxSpan instances"),
    ("BoxSpan → 频谱 LinearRegionItem", "BoxSpan to spectrogram LinearRegionItem mapping"),
    ("BoxSpan → annotations 索引", "BoxSpan to annotations index mapping"),
    ("当前处于编辑态的 BoxSpan", "BoxSpan currently in edit mode"),
    ("当前被点击选中的 BoxSpan（用于 Delete 等快捷键）", "Currently selected BoxSpan (used by Delete and other shortcuts)"),
    ('可选："Heatmap"、"Grayscale"', 'Options: "Heatmap", "Grayscale"'),
    ("None 表示自动取 1%~99% 分位数", "None means auto-compute from the 1st-99th percentile"),
    ("最近一次 STFT 原始数据 (freq×time)，用于直方图统计与重着色", "Most recent STFT raw data (freq x time), used for histogram statistics and recoloring"),
    ("{特征名: QColor}，按所选特征顺序依次分配", "{feature_name: QColor}, assigned sequentially in selection order"),
    ("{标签文本: QColor}", "{label_text: QColor}"),
    ("当前 ML 操作的目标标签（由 init_ml_toolbar 中的下拉框赋值）", "Target label for the current ML operation (assigned by the combo box in init_ml_toolbar)"),
    ("上次导出的 CSV 路径（用于下次导出时沿用同一目录）", "Last export CSV path (reused as the default directory for the next export)"),
    ("当前 WAV 对应的默认导出文件名", "Default export filename for the current WAV"),
    ("波形显示范围 (ymin, ymax)", "Waveform display range (ymin, ymax)"),
    ("上次打开的 Settings 标签页索引", "Index of the last-opened Settings tab"),
    ("初始隐藏 Y 轴", "Initially hide the Y-axis"),
    ("短时特征曲线缓存 {特征名: pg.PlotDataItem}", "Short-time feature curve cache {feature_name: pg.PlotDataItem}"),
    ("帧时间戳 1D array (T,)", "Frame timestamps, 1D array (T,)"),
    ("帧特征 2D array (T, D)", "Frame features, 2D array (T, D)"),
    ("特征名列表 list[str]", "Feature name list, list[str]"),
    ("已训练模型 {label: {clf, threshold, feature_names, ...}}", "Trained models {label: {clf, threshold, feature_names, ...}}"),
    ("ML 训练/推理调度器", "ML training/inference dispatcher"),
    # ── Method-level inline comments ──
    ("基础标题", "Base title"),
    ("监听 STFT 图的鼠标移动", "Listen for mouse movement on the STFT plot"),
    ("第 0 页：STFT", "Page 0: STFT"),
    ("第 1 页：FFT", "Page 1: FFT"),
    ("第 2 页：短时特征", "Page 2: Short-Time Features"),
    ("默认显示 STFT", "Default to STFT"),
    ("上：页面栈", "Top: page stack"),
    ("中：波形", "Middle: waveform"),
    ("下：标注", "Bottom: annotations"),
    ("拉伸比例（可按实际需求调整）", "Stretch factors (adjustable according to actual needs)"),
    ("把 STFT 相关显示数据传进去", "Pass STFT-related display data in"),
    ("Settings上次使用的标签页", "Last-used Settings tab"),
    ("新增", "New"),
    ('新增：保存"读取时预处理"和"自动标签读取参数"。不立即重读当前音频，避免影响现有编辑状态；下次 load_audio 生效。', 'New: save "load-time preprocessing" and "auto label import" parameters. Do not immediately reload the current audio to avoid disrupting the current editing state; takes effect on the next load_audio call.'),
    ("取回 STFT 显示设置 (配色 + 上下限)，先保存再重绘，避免使用旧色阶。", "Retrieve STFT display settings (color scheme + bounds); save before redrawing to avoid using old color levels."),
    ("Settings 改变后只刷新波形/STFT，并标记 FFT/特征失效；不立即计算短时特征。", "After Settings changes, only refresh the waveform/STFT and mark FFT/features as dirty; do not recompute short-time features immediately."),
    ("若只改了色阶而未重算，也可以直接按当前谱图重着色。", "If only the color levels were changed without recomputing, simply recolor the current spectrogram in-place."),
    ("默认路径：目录沿用上次导出位置，文件名跟随当前 WAV", "Default path: directory reuses the last export location; filename follows the current WAV"),
    ("例如：当前音频 1008.wav → 默认文件名 1008_events.csv", "Example: current audio 1008.wav -> default filename 1008_events.csv"),
    ("验证：打印导出 source 分布统计（可在控制台查看）", "Verification: print exported source-distribution statistics (viewable in the console)"),
    ("记住路径", "Remember the path"),
    ("删除 BoxSpan 对象", "Remove the BoxSpan object"),
    ("记录硬负样本（按被删除标注的原始标签，而非当前训练类型）", "Record a hard negative sample (using the deleted annotation's original label, not the current training label)"),
    ("移除 UI 组件", "Remove UI components"),
    ("记录撤销信息", "Record undo information"),
    ("兼容：旧 LinearRegionItem 删除路径 (如果你代码中还有)", "Backward compatibility: old LinearRegionItem deletion path (if still present in your code)"),
    ("若传入的是 STFT 高亮区域 (LinearRegionItem)，尝试反查对应的 BoxSpan，", "If the input is an STFT highlight region (LinearRegionItem), attempt to reverse-lookup the corresponding BoxSpan,"),
    ("统一走 BoxSpan 删除逻辑，以便支持硬负样本与 Ctrl+Z 撤销。", "and route through the unified BoxSpan deletion logic to support hard-negative samples and Ctrl+Z undo."),
    ("复用 BoxSpan 删除逻辑", "Reuse the BoxSpan deletion logic"),
    ("若已经有同 idx 的 span (理论上不该发生)，直接返回避免重复", "If a span with the same index already exists (should not happen in theory), return immediately to prevent duplication"),
    ("非 manual source 使用红色文字", "Non-manual source annotations use red text"),
    ("STFT 高亮", "STFT highlight"),
    ("用于视图与数据不同步时的重建兜底。", "Used as a rebuild fallback when the view and data are out of sync."),
    ("兜底：至少从图中移除", "Fallback: at least remove it from the plot"),
    ("STFT 高亮区域", "STFT highlight regions"),
    ("索引映射", "Index mapping"),
    ("兼容旧格式残留", "Backward compatibility with legacy format remnants"),
    ("重置轨道缓存（不影响数据）", "Reset lane cache (does not affect data)"),
    ("清空视图", "Clear the view"),
    ("复位 Y 轴显示范围（避免渲染后不可见）", "Reset the Y-axis display range (to avoid invisibility after rendering)"),
    ("重建", "Rebuild"),
    ("archived 不显示也不导出 (合并/删除前的旧 span)", "Archived spans are neither displayed nor exported (old spans before merge/delete)"),
    ("避免异常中断主流程", "Avoid interrupting the main flow with exceptions"),
    ("兜底：恢复失败则重建整个视图", "Fallback: if restoration fails, rebuild the entire view"),
    ("span 丢失则重建整个视图", "If the span is lost, rebuild the entire view"),
    ("STFT 高亮同步", "Sync STFT highlight"),
    ("兜底：回滚失败则重建视图", "Fallback: if rollback fails, rebuild the view"),
    ("BoxSpan 编辑模式：双击进入；Enter 提交；Esc 取消", "BoxSpan edit mode: double-click to enter; Enter to commit; Esc to cancel"),
    ("退出编辑后清理状态栏提示", "Clean the status bar notice after exiting edit mode"),
    ("已有映射直接返回", "Return directly if already mapped"),
    ("预设类型 (英文名)", "Preset type (English name)"),
    ("其他任意文本：从调色板顺序取色", "Any other text: pick a color sequentially from the palette"),
    ("若未提供文本 (正常交互标注)，则弹出带预设类型的对话框", "If no text is provided (normal interactive annotation), show the dialog with preset types"),
    ("渲染标注可视化 + 频谱高亮", "Render annotation visualization + spectrogram highlight"),
    ("注册到 annotations 数据", "Register into the annotations data store"),
    ("自动导入同名 _events 标注 (可开关)", "Auto-import matching _events annotations (toggleable)"),
    ("STFT → FFT：触发懒加载", "STFT -> FFT: trigger lazy loading"),
    ("FFT → Short-Time Features：懒加载短时特征", "FFT -> Short-Time Features: lazy-load short-time features"),
    ("只要有时间交集即视为重叠", "Any temporal intersection is considered an overlap"),
    ("收集现有每一行的区间", "Collect the intervals of every existing lane"),
    ("找第一个不重叠的行", "Find the first non-overlapping lane"),
    ("三行都冲突，则放最后一行", "If all three lanes conflict, place it in the last lane"),
    ("仅在绘图区内触发 (不包含轴刻度/边距)", "Trigger only inside the plotting area (excluding axis ticks/margins)"),
    ("标题右侧显示坐标 (可调样式)", "Display coordinates on the right side of the title (style adjustable)"),
    ("画最多 5 items", "Plot at most 5 items"),
    ("选择颜色", "Assign colors"),
    ("时间轴范围与 STFT 对齐", "Align the time-axis range with STFT"),
    ("与主图方向一致：time × freq", "Match the main plot orientation: time x freq"),
    ("规则：与 WAV 同目录下的 <wav_base>_events.(csv|txt)", "Rule: look for <wav_base>_events.(csv|txt) in the same directory as the WAV file"),
    ("同时注册系统标准 Undo 快捷键（Windows/Linux: Ctrl+Z, macOS: Cmd+Z）", "Also register the system-standard Undo shortcut (Windows/Linux: Ctrl+Z, macOS: Cmd+Z)"),

    # ── Load audio block inline comments ──
    ("切换文件前停止当前播放，避免旧音频仍占用声卡。", "0. Stop playback before switching files to prevent the old audio from still occupying the sound card."),
    ("手动导入路径为空时，弹出文件选择框", "1. If no path is provided, show the file selection dialog"),
    ("建立同目录 WAV 列表，用于上一首/下一首。", "2. Build a WAV-file list from the same directory for previous/next navigation."),
    ("读取音频。默认使用 Settings 中的重采样采样率；关闭重采样时保留原始采样率。", "3. Read the audio. Use the resample rate from Settings by default; preserve the original sample rate when resampling is disabled."),
    ("切换文件时，先清空旧缓存和旧标注，避免旧图/旧特征误用。", "4. When switching files, clear old caches and old annotations first to avoid reusing stale plots/features."),
    ("新音频默认重置自动色阶，避免上一条文件的色阶影响当前文件。", "Reset auto-levels so the previous file does not affect the current one."),
    ("先清旧标注，再重绘新图，防止旧 STFT 高亮残留。", "Clear old annotations first, then redraw new plots to prevent residual old STFT highlights."),
    ("只做必要显示：抽稀波形 + STFT。FFT/特征延后到用户切换页面时计算。", "5. Show only essentials: decimated waveform + STFT. FFT and features are deferred until the user switches pages."),
    ("自动导入标签。标签导入很轻，仍保留在 load_audio 里。", "6. Auto-import labels. Label import is lightweight, so it stays inside load_audio."),
    ("坐标范围与页面状态。", "7. Coordinate ranges and page state."),
    ("保存完整谱值用于 Settings 直方图；显示用谱图可以抽稀。", "Save full spectrogram values for the Settings histogram; the display spectrogram may be decimated."),
    ("抽稀后的图像仍映射到完整音频时长，保证标注时间轴一致。", "The decimated image still maps to the full audio duration, ensuring consistent annotation time axes."),
    ("STFT/FMAX 变化会影响频域特征，但不在 STFT 刷新时立即计算特征。", "Changes to STFT/f_max affect frequency-domain features, but features are not recomputed immediately during STFT refresh."),
    ("移除可视化对象 —— BoxSpan 及其上方文字标签", "1) Remove visual objects -- BoxSpan instances and their text labels"),
    ("从 annot_plot 移除自身及文字标签", "Remove itself and its text label from annot_plot"),
    ("移除 STFT 频谱图上的高亮区域", "2) Remove highlight regions from the STFT spectrogram"),
    ("兼容旧格式：清除 annotation_items 中保存的区域和文字", "3) Backward compatibility: clear regions and text stored in annotation_items"),
    ("清除拖拽过程中的临时选区", "4) Clear temporary selection regions from an in-progress drag"),
    ("清空数据结构", "5) Clear data structures"),
    ("复位标注视图范围", "6) Reset the annotation view range"),
    ("取消波形高亮", "7) Cancel waveform highlight"),
    ("BoxSpan 及其文字标签", "1) BoxSpan instances and their text labels"),
    ("临时拖拽选区", "5) Temporary drag-selection region"),
    ("分轨", "Lane assignment"),
    ("颜色与笔刷", "Color and pen"),
    ("创建并渲染一个标注的 BoxSpan + STFT 高亮区域。", "Create and render a BoxSpan plus STFT highlight region for an annotation."),
    ("共用，消除重复的分轨/颜色/笔刷/高亮逻辑。", "Shared by both, eliminating duplicated lane-assignment / color / pen / highlight logic."),
    ("仅清空可视化对象（BoxSpan、STFT 高亮等），不清空 annotations/neg/undo 等数据。", "Clear only visual objects (BoxSpan, STFT highlights, etc.); do not clear annotations/neg/undo data."),
    ("从 self.annotations 重新渲染所有 BoxSpan（不改动 annotations 数据）。", "Rebuild all BoxSpan visualizations from self.annotations (does not modify annotation data)."),
    ("根据 annotations[idx] 重建 BoxSpan 可视化（仅供撤销使用），不修改 annotations。", "Rebuild the BoxSpan visualization from annotations[idx] (for undo only); does not modify annotations."),
    ("将指定标注的 source 修改为 new_source 并同步 UI，支持撤销。", "Change the source of a specified annotation to new_source, sync the UI, and support undo."),
    # Data
    ("先搭\"source 状态机\"的基础：后续会逐步引入更多状态。", 'Lay the foundation for the "source state machine": more states will be introduced gradually in the future.'),
    ('reviewed: 参与"已审阅前缀"统计；trainable: 可进入下一轮训练的正样本标注。', 'reviewed: participates in the "reviewed prefix" statistic; trainable: a positive-sample annotation eligible for the next training round.'),
    ("兼容三元组 / 四元组格式，其他长度直接跳过", "Support 3-tuple / 4-tuple formats; skip other lengths"),
    ("三元组直接视为人工标记", "3-tuples are treated directly as manual labels"),
    ("四元组用自带的 source", "4-tuples use their own source field"),
    ("过滤 None 项，兼容 3/4 元组格式；排除 source='archived'（已归档/隐藏）的标注", "Filter out None entries; support 3-/4-tuple formats; exclude annotations whose source is 'archived' (archived/hidden)"),

    # Ctrl shortcuts inline
    ("Ctrl+Z 撤销", "Ctrl+Z undo"),
    ("Ctrl+A 认可选中的 ML 标注", "Ctrl+A accept the selected ML annotation"),
    ("Delete 键删除选中的标注", "Delete key removes the selected annotation"),
    ("编辑模式键盘：Enter 提交 / Esc 取消", "Edit-mode keyboard: Enter to commit / Esc to cancel"),

    # Section headers
    ("播放/暂停 (Space)", "Play / Pause (Space)"),
    ("快退/快进 (Left/Right)", "Seek backward / forward (Left / Right)"),
    ("ML 训练/自动标注：使用当前 ML 下拉框选中的标签", "ML training / auto-labeling: uses the label selected in the ML combo box"),
    ("撤销 (Ctrl+Z)", "Undo (Ctrl+Z)"),
    ("File 菜单", "File menu"),
    ("Settings 菜单", "Settings menu"),
    ("Help 菜单", "Help menu"),
    ("下拉框：选择待训练/自动标注的标签", "Combo box: select the label to train/auto-label"),
    ("训练按钮", "Train button"),
    ("自动标注按钮", "Auto-label button"),
    ("清除负样本按钮", "Clear negatives button"),
    ("标注颜色图例按钮", "Annotation color legend button"),
    ("工具栏槽函数", "Toolbar slot functions"),

    # Color label in-code comments (NOT data, just comments next to data)
    ("红", "Red"),
    ("蓝", "Blue"),
    ("绿", "Green"),
    ("紫", "Purple"),
    ("橙", "Orange"),
    ("棕", "Brown"),
    ("粉", "Pink"),
    ("灰", "Gray"),
    ("青绿", "Teal"),
]

def translate_text(text):
    """Apply both phrase lists in order (MORE first, then v2 list as fallback)."""
    result = text
    for cn, en in MORE_PHRASES + PHRASES:
        if cn in result:
            result = result.replace(cn, en)
    return result

# ── Process ─────────────────────────────────────────────────────────────
dry_run = '--write' not in sys.argv
with open(TARGET, 'r', encoding='utf-8') as f:
    lines = f.readlines()

changed = 0
data_keywords = ['("', "('", 'FEATURE_NAMES', 'PHASE_LABELS', 'OTHER_EVENT',
                 'ABNORMAL_SOUND', 'annotation_builtin_labels',
                 'annotation_color_builtin', '哮鸣音', '爆裂音', '摩擦音',
                 '哼鸣音', '喘息音', '语音', '咳嗽', '呼气', '吸气',
                 'Wheeze":', 'Crackles":', 'Pleural Rub":', 'Rhonchi":',
                 'Stridor":', 'Speech":', 'Cough":', 'Expiration":',
                 'Inspiration":', '"wheeze"', 'feature_palette']

for i, line in enumerate(lines):
    stripped = line.strip()
    if not C.search(line):
        continue

    # Skip data lines
    if any(kw in stripped for kw in data_keywords):
        continue

    indent = line[:len(line) - len(line.lstrip())]

    if stripped.startswith('"""') or stripped.startswith("'''"):
        # Docstring — translate the whole line
        new_line = translate_text(line)
        if new_line != line:
            lines[i] = new_line
            changed += 1

    elif stripped.startswith('#'):
        # Pure comment: translate everything after '# '
        hash_pos = line.index('#')
        prefix = line[:hash_pos + 1]
        comment = line[hash_pos + 1:]
        translated = translate_text(comment)
        new_line = prefix + translated
        if new_line != line:
            lines[i] = new_line
            changed += 1

    elif '#' in line:
        # Inline comment: translate only the comment part
        hash_pos = line.index('#')
        code_part = line[:hash_pos]
        comment = line[hash_pos:]
        translated = translate_text(comment)
        new_line = code_part + translated
        if new_line != line:
            lines[i] = new_line
            changed += 1

if not dry_run:
    with open(TARGET, 'w', encoding='utf-8') as f:
        f.writelines(lines)

action = '[DRY RUN]' if dry_run else '[WRITTEN]'
print(f"{action} {TARGET}: {changed} changes")
if dry_run:
    print("Run with --write to apply.")
