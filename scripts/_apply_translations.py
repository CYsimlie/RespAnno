"""Apply Chinese→English comment translations from all four translation agents.

Reads translation mappings from JSON files produced by the agents,
then performs exact-string replacement in each source file.

Usage:
    python scripts/_apply_translations.py          # dry-run
    python scripts/_apply_translations.py --write  # actually write
"""
import os, sys, json, re

ROOT = r'd:\SoftwareX_win_test\New folder2\SoftwareX'

# ── Translation mappings ────────────────────────────────────────────────

GUI_TRANSLATIONS = {
    "respanno/gui/dialogs/settings_dialog.py": {
        "来源: 1.6.6.py 行号 79-762 (class SettingsDialog)": "Source: 1.6.6.py lines 79-762 (class SettingsDialog)",
        "# —— 新增：读取音频时的预处理设置。默认开启 4000 Hz 重采样；滤波默认关闭，避免改变旧行为。——":
            "# — New: preprocessing settings when loading audio. 4000 Hz resampling on by default; filtering off by default to preserve old behavior. —",
        "# —— 新增：自动读取同名标签文件的解析设置。默认保持旧逻辑：auto/csv/txt、_events 后缀、自动分隔符、前3列为 start/end/label。——":
            "# — New: parsing settings for auto-import of matching label files. Preserves old defaults: auto/csv/txt, _events suffix, auto delimiter, columns 1-3 as start/end/label. —",
        "# ==== STFT 标签页 ====": "# ==== STFT tab ====",
        "# ==== STFT Display (直方图 + colorbar + 上下限 + Reset Defaults)====":
            "# ==== STFT Display (histogram + colorbar + limits + Reset Defaults) ====",
        "# 顶部：直方图 + 渐变色条 (HistogramLUTWidget)": "# Top: histogram + gradient color bar (HistogramLUTWidget)",
        "# 给 HistogramLUT 提供一份图像数据，便于统计直方图": "# Provide image data to HistogramLUT for histogram statistics",
        "# 与主图方向一致：time × freq": "# Consistent with main plot orientation: time x freq",
        "# 生成与主图风格一致的 ColorMap": "# Generate a ColorMap consistent with the main plot style",
        "# 初始化 colorbar 配色 (保留可编辑三角控件)": "# Initialize colorbar color scheme (preserve editable triangle handles)",
        "# 某些 PyQt5 + 老 pyqtgraph 需要 save/restore 一次才能正确刷新": "# Some PyQt5 + old pyqtgraph require a save/restore cycle to refresh correctly",
        "# 中部：配色 + 上下限输入框": "# Middle: color scheme + limit input boxes",
        "# 初始化上下限：优先用传入值，否则用 1%~99% 分位": "# Initialize limits: prefer passed-in values, fall back to 1%-99% percentiles",
        "# Settings到 HistogramLUTItem：这一步决定开窗区间 (与主图共享)": "# Settings to HistogramLUTItem: this step determines the window range (shared with main plot)",
        "# 同步到输入框": "# Sync to input boxes",
        "# 底部：Reset Defaults (回到 1%~99% 分位)": "# Bottom: Reset Defaults (return to 1%-99% percentiles)",
        "# 组装到 group 里": "# Assemble into group",
        "# 配色": "# Color scheme",
        "# 下限/上限": "# Lower / upper limit",
        "# 组装进 STFT 页": "# Assemble into STFT page",
        "# ========= 联动逻辑：levels ↔ 文本框，顺带保证与主图一致 =========":
            "# ========= Linkage logic: levels <-> text boxes, also ensuring consistency with main plot =========",
        "# 1)拖动 HistogramLUT items (或编辑 colorbar) → 更新输入框": "# 1) Dragging HistogramLUT items (or editing colorbar) -> update input boxes",
        "# 2)改输入框 → 回写到 HistogramLUT (开窗)，colorbar 的条和三角会一起动": "# 2) Changing input boxes -> write back to HistogramLUT (windowing), bar and triangles move together",
        "# 3)切换配色 → 更新 colorbar (保留颜色编辑功能)": "# 3) Switch color scheme -> update colorbar (preserve color editing)",
        "# 再 save/restore 一次，防止 Qt5 下首次不刷新": "# Save/restore once more to prevent initial non-refresh under Qt5",
        "# 4)Reset Defaults (上下限回到 1%~99% 分位)": "# 4) Reset Defaults (limits return to 1%-99% percentiles)",
        "# ==== Display标签页 ====": "# ==== Display tab ====",
        "# 缩放滑条 + 输入框": "# Zoom slider + input box",
        "# 上下限": "# Upper and lower limits",
        "# Reset Defaults按钮": "# Reset Defaults button",
        "# ==== Preprocessing 标签页：读取 WAV 时的重采样与可选滤波 ====":
            "# ==== Preprocessing tab: resampling and optional filtering when loading WAV ====",
        "# ==== Auto Label Import 标签页：配置自动读取同名标签文件的解析规则 ====":
            "# ==== Auto Label Import tab: configure parsing rules for auto-importing matching label files ====",
        "# 1-based 列号；source_col=0 表示不读取 source，统一 manual。":
            "# 1-based column numbers; source_col=0 means skip source column, default to manual.",
        "# 按钮": "# Buttons",
        "# ==== Short-Time Features 标签页 ====": "# ==== Short-Time Features tab ====",
        "# 行间距更舒服": "# More comfortable row spacing",
        "# 可选：统一放大每一项的 sizeHint (若你想更高)": "# Optional: uniformly enlarge each item's sizeHint (if you want taller rows)",
        "# —— 频域/时频统计特征 (基于 STFT Amplitude谱)——":
            "# — Frequency-domain / time-frequency statistical features (based on STFT amplitude spectrum) —",
        "# —— 窄带/少峰检测增强特征 ——": "# — Narrowband / sparse-peak detection enhancement features —",
        "# —— 频移自相关 (cor)特征：100–1200 Hz 子带，基于每帧谱的\"向下频移自相关\"曲线 ——":
            '# — Frequency-shifted autocorrelation (cor) features: 100-1200 Hz sub-band, based on per-frame "downward frequency-shifted autocorrelation" curve —',
        "# 应用有色对勾委托": "# Apply colored checkmark delegate",
        "# —— 让勾选颜色与曲线一致 ——": "# — Make checkmark colors consistent with curves —",
        "# 颜色分配规则与主窗口一致：按\"已选特征\"的顺序分配":
            '# Color assignment matches main window: assigned by order of selected features',
        "# 构建一个 {特征名: QColor} 映射": "# Build a {feature_name: QColor} mapping",
        "# 把颜色写到每个条目的 UserRole，并顺带给未选的分配备用颜色 (避免后续点选颜色重复)":
            "# Write color to each item's UserRole, and assign fallback colors to unselected ones (avoid duplicate colors on later selection)",
        "# 预勾选 (来自主窗口)": "# Pre-check (from main window)",
        "# 取消最新一次勾选 (把它设回未选)": "# Uncheck the most recent check (set it back to unchecked)",
        "# 注意：此槽在单个 item 改变时触发，找到那个 item": "# Note: this slot fires on a single item change; find that item",
        "# 兼容性占位": "# Compatibility placeholder",
        "# 简单策略：从末尾开始把第6个之后的设回未选": "# Simple strategy: from the end, set items beyond the 5th back to unchecked",
        '"\\u6ed1\\u6761\\u6539\\u53d8\\u65f6\\u66f4\\u65b0\\u500d\\u6570\\u8f93\\u5165\\u6846\\u548c\\u4e0a\\u4e0b\\u9650"':
            '"""Update zoom factor input box and limits when slider changes."""',
        '"\\u624b\\u52a8\\u8f93\\u5165\\u7f29\\u653e\\u500d\\u6570\\u65f6\\u66f4\\u65b0\\u6ed1\\u6761\\u548c\\u4e0a\\u4e0b\\u9650"':
            '"""Update slider and limits when zoom factor is entered manually."""',
        '"\\u6839\\u636e\\u500d\\u6570\\u7f29\\u653e\\u539f\\u59cb\\u7684 min/max \\u5e76\\u66f4\\u65b0Display\\u8303\\u56f4"':
            '"""Scale original min/max by zoom factor and update display range."""',
        '"""Reset Defaults\\u7684\\u81ea\\u52a8\\u8ba1\\u7b97\\u7684\\u8303\\u56f4"""':
            '"""Auto-calculated range for Reset Defaults."""',
        '"\\u8fd4\\u56de (cmap_name, vmin, vmax)\\n        \\u4f18\\u5148\\u4ece HistogramLUT \\u7684 levels \\u8bfb (\\u4fdd\\u8bc1\\u548c colorbar \\u5bf9\\u9f50)"':
            'Return (cmap_name, vmin, vmax).\n        Prefer reading from HistogramLUT levels (ensures alignment with colorbar).',
    },
    "respanno/gui/dialogs/annotation_label_dialog.py": {
        '"\\u6807\\u6ce8\\u6807\\u7b7e\\u9009\\u62e9\\u5bf9\\u8bdd\\u6846\\uff1a\\u652f\\u6301\\u9884\\u8bbe\\u7c7b\\u578b + Custom\\u6587\\u672c"':
            '"""Annotation label selection dialog: supports preset types + Custom text."""',
        "# 顶部Notice：Time段": "# Top notice: time segment",
        "# 预设类型下拉框：如 哮鸣音(Wheeze)、爆裂音(Crackles) 等": "# Preset type dropdown: e.g. Wheeze, Crackles, etc.",
        "# 文本输入框：最终用于保存的标签文本 (通常是英文)": "# Text input box: final label text used for saving (usually English)",
        "# 选择预设时，自动把英文名写入文本框": "# When selecting a preset, automatically fill the English name into the text box",
        "# 底部按钮": "# Bottom buttons",
        "# 不输入则视为取消": "# Treat empty input as cancel",
        '"\\u6a21\\u6001\\u6267\\u884c\\u5e76\\u8fd4\\u56de\\u6700\\u7ec8\\u6587\\u672c (\\u53ef\\u80fd\\u4e3a None)\\u3002"':
            '"""Execute modally and return the final text (may be None)."""',
    },
    "respanno/gui/spans/box_span.py": {
        "# default\"只读\"：不允许拖动/改boundary；仅在editmode下开启": '# Default "read-only": no dragging or boundary changes; enabled only in edit mode',
        "# Disable corner handles": "# Disable corner handles",
        "# —— 视觉encoding：不再强制纯白填充 (后面统一由 _apply_visual_style() 控制)——": "# — Visual encoding: no longer force solid white fill (unified by _apply_visual_style() later) —",
        "# Transparent first, to avoid white block occlusion": "# Transparent first, to avoid white block occlusion",
        "# Lock vertical position": "# Lock vertical position",
        "# Lock height": "# Lock height",
        "# 独立填充层：RectROI 在部分 pyqtgraph version中只明显Display边框，": "# Independent fill layer: RectROI only clearly shows border in some pyqtgraph versions,",
        "# 因此额外用一个 QGraphicsRectItem 负责稳定Display类别填充色。": "# so use an extra QGraphicsRectItem to reliably display the category fill color.",
        "# —— editmode门控 (default关闭，双击进入)——": "# — Edit mode gate (off by default, double-click to enter) —",
        "# data层快照 (用于editcommit后可undo)": "# Data-layer snapshot (for undo after edit commit)",
        "# ★ New：label文字color (不传则default黑色)": "# * New: label text color (defaults to black if not passed)",
        "# default黑字": "# Default black text",
        "# 只保留左右把手 (defaulthide+disable；仅在editmodeenable)": "# Keep only left/right handles (hidden+disabled by default; enabled only in edit mode)",
        "# disable把手的鼠标交互 (避免与\"mark拖拽\"冲突)": '# Disable handle mouse interaction (avoid conflict with "mark drag")',
        "# 备注小白框 (Display在条上方)": "# Note label box (displayed above the bar)",
        "# Create an empty one first": "# Create an empty one first",
        "# 根据 label_color &amp; text update HTML": "# Update HTML based on label_color & text",
        "# annotation条不直接Display文字，label含义统一通过toolbar\"Annotation Legend\"查看": '# Annotation bar does not display text directly; label meanings are viewed via the toolbar "Annotation Legend"',
        "# —— 视觉encoding：根据 source(manual/ml) 统一Settings pen/brush/label style ——": "# — Visual encoding: unify pen/brush/label styles based on source (manual/ml) —",
        "# editmode：仅在该mode下允许拖动/改boundary": "# Edit mode: dragging and boundary changes allowed only in this mode",
        "# 记录 data 层快照 (用于commit后可undo)": "# Record data-layer snapshot (for undo after commit)",
        "# Allow overall translation": "# Allow overall translation",
        "# enable并Display左右把手 (仅用于水平改boundary)": "# Enable and display left/right handles (for horizontal boundary changes only)",
        "# 视觉Notice：edit态用虚线 (不改变 source 的语义；Exit后restore)": "# Visual notice: dashed line during edit (does not change source semantics; restored on exit)",
        "# 通知 owner：进入edit态 (用于键盘 Enter/Esc)": "# Notify owner: entering edit state (for keyboard Enter/Esc)",
        "# 初次刷新status栏": "# Initial status bar refresh",
        "# cancel：restore原interval (不入 undo)": "# Cancel: restore original interval (not pushed to undo)",
        "# disable移动": "# Disable movement",
        "# hide并disable把手": "# Hide and disable handles",
        "# 强制synchronize一次 (update annotations/spec/label + restore source 对应style)": "# Force sync once (update annotations/spec/label + restore source-corresponding style)",
        "# commit：若interval发生变化，则入栈 undo，support Ctrl+Z": "# Commit: if interval changed, push to undo stack for Ctrl+Z support",
        "# new_item based on data layer": "# new_item based on data layer",
        "# 若edit的是机器/认可annotation，则将 source 升级为 auto_edited (便于进入train与export追踪)": "# If editing a machine/accepted annotation, upgrade source to auto_edited (for training inclusion and export tracking)",
        "# Clean up snapshots": "# Clean up snapshots",
        "# 视觉encoding：New的辅助函数": "# Visual encoding: new helper functions",
        "# 找不到就default manual。仅用于视觉encoding，不改变逻辑。": "# Default to manual if not found. For visual encoding only; does not change logic.",
        "# Box 基色：按label类别稳定映射 (不要用 label_color；label_color 仅用于文字color)": "# Box base color: stable mapping by label category (not label_color; label_color is only for text color)",
        "# Primary color (border / emphasis)": "# Primary color (border / emphasis)",
        "# Visual conventions:": "# Visual conventions:",
        "# - 未确认的机器annotation：虚线 (ml/auto/pred 等)或 merged_*global*": "# - Unconfirmed machine annotations: dashed line (ml/auto/pred etc.) or merged_*global*",
        "# - 已认可/已edit/局部合并后的annotation：实线 (auto_accepted/auto_edited/merged 等)": "# - Accepted / edited / locally merged annotations: solid line (auto_accepted/auto_edited/merged etc.)",
        "# Apply to ROI body": "# Apply to ROI body",
        "# RectROI itself retains faint fill; the independent fill layer handles the main visible color block.": "# RectROI itself retains faint fill; the independent fill layer handles the main visible color block.",
        "# synchronize label 的视觉 (不改文字内容)": "# Sync label visuals (not text content)",
        "# Note box follows (slightly above the bar)": "# Note box follows (slightly above the bar)",
        "# synchronizespectrumRed条": "# Sync spectrogram red bar",
        "# synchronizeexport缓存 (兼容 3/4 元组；三元组视为人工annotation)": "# Sync export cache (compatible with 3/4-tuples; 3-tuples treated as manual annotations)",
        "# —— 视觉encoding：": "# — Visual encoding:",
        "# 非edit态：按 source synchronizestyle": "# Non-edit state: sync style by source",
        "# edit态：保持\"edit态虚线\"Notice，不要被 source style覆盖": '# Edit state: keep "edit-mode dashed" notice; do not let source style override',
        "# edit态：实时在status栏Display起止Time": "# Edit state: real-time display of start/end time in status bar",
        "# 双击switcheditmode：default进入；edit态再双击commitExit": "# Double-click toggles edit mode: enter by default; double-click again in edit state to commit and exit",
        "# 兜底：如果出错，仍允许Play": "# Fallback: if error, still allow playback",
        "# 非editmode下：吞掉拖拽 (不移动/不改boundary)，避免与\"mark拖拽\"冲突": '# Non-edit mode: swallow drag events (no move or boundary change), avoid conflict with "mark drag"',
        "# editmode下：交给 ROI default实现 (support平移/改boundary)": "# Edit mode: delegate to ROI default implementation (supports translation / boundary changes)",
        "# —— 机器annotation\"认可\"：将 source 置为 auto_accepted (进入train/计入已reviewprefix)——": '# — Machine annotation "Accept": set source to auto_accepted (include in training / count as reviewed prefix) —',
    },
    "respanno/gui/spans/span_label_item.py": {
        '# \\"\\"\\"annotation条上的文字label：把右键/双击event转发给所属 BoxSpan，避免被mark拖拽逻辑抢走。\\"\\"\\"':
            '# Text label on annotation bar: forward right-click / double-click events to the owning BoxSpan, preventing them from being hijacked by mark-drag logic.',
    },
    "respanno/gui/widgets/color_bar.py": {
        '"\\u53f3\\u4fa7\\u7684\\u7eafDisplay\\u8272\\u6761\\uff0c\\u4e0d\\u53ef\\u70b9\\u51fb\\uff0c\\u4e0d\\u4f1a\\u751f\\u6210\\u4e09\\u89d2\\u63a7\\u4ef6\\u3002\\n    \\u76f4\\u63a5\\u7528 AudioViewer._get_palette_256 \\u753b\\u51fa 0~1 \\u7684\\u6e10\\u53d8\\u3002\\n    "':
            'Pure display color bar on the right side; non-clickable, no triangle handles.\n    Draws a 0-1 gradient directly using AudioViewer._get_palette_256.\n    ',
        "# 瘦瘦的一条，类似 colorbar": "# A thin strip, similar to a colorbar",
        "# 兜底：自己构造一个简单 LUT": "# Fallback: construct a simple LUT ourselves",
        "# 粗略 viridis 风格": "# Rough viridis style",
        "# 低值在底部": "# Low values at bottom",
        "# 上lower limit指示线": "# Upper / lower limit indicator lines",
    },
    "respanno/gui/widgets/color_check_delegate.py": {
        '"\\u52fe\\u9009\\u540e\\u624d\\u4e0a\\u8272\\uff1b\\u672a\\u52fe\\u9009Display\\u7a7a\\u6846\\uff1b\\u6574\\u884c\\u53ef\\u70b9\\u51fb\\u5207\\u6362\\uff1b\\u65b9\\u5757\\u5c3a\\u5bf8\\u66f4\\u5927"':
            '"""Color only when checked; empty box when unchecked; entire row clickable to toggle; larger box size."""',
        "# 放大方块 (原来16)": "# Enlarged box (was 16)",
        "# 背景": "# Background",
        "# 勾选状态": "# Check state",
        "# 颜色 (来自 UserRole；未勾选也可以先拿着，但不填充)": "# Color (from UserRole; can be held even when unchecked, but not filled)",
        "# 方块区域 (更大)": "# Box area (larger)",
        "# 边框": "# Border",
        "# 填充：仅已勾选才填充彩色；未勾选Display空框": "# Fill: only fill with color when checked; empty box when unchecked",
        "# 勾选对号：仅已勾选绘制": "# Checkmark: only drawn when checked",
        "# 文本": "# Text",
        "# 放大行高，增大点击目标": "# Increase row height, enlarge click target",
        "# 写回 CheckStateRole": "# Write back to CheckStateRole",
    },
}

MAIN_TRANSLATIONS = {
    '# 构造 Qt plugins 目录的绝对路径（兼容不同 PyQt5 安装布局）':
        'Construct the absolute path to the Qt plugins directory (compatible with different PyQt5 installation layouts)',
    '# 若 Qt5 层级不存在，回退至无 Qt5 层级的路径':
        'If the Qt5-level directory does not exist, fall back to the path without the Qt5 level',
    '# 设置 QT 插件环境变量，确保 Qt 运行时能找到 platform plugins':
        'Set the QT plugin environment variable to ensure the Qt runtime can locate platform plugins',
    '# —— 机器学习硬负样本管理器（仅用于训练，不导出，与标注显示无关）——':
        'Machine-learning hard-negative sample manager (training only; not exported; unrelated to annotation display)',
    '# —— 撤销栈（支持删除、认可等编辑操作的回退）——':
        'Undo stack (supports rollback of delete, accept, and other edit operations)',
    '# —— 多轨标注管理（最多 3 轨，仅用于标注视图）——':
        'Multi-lane annotation management (max 3 lanes; annotation view only)',
    'self.LANE_H = 0.35          # 单轨高度':
        'self.LANE_H = 0.35          # Single-lane height',
    'self.LANE_GAP = 0.25        # 轨道间距':
        'self.LANE_GAP = 0.25        # Lane gap',
    'self._lanes = [[] for _ in range(self.MAX_LANES)]    # 每轨占用区间 [(s, e), ...]':
        'self._lanes = [[] for _ in range(self.MAX_LANES)]    # Occupied intervals per lane [(s, e), ...]',
    'self._spans = []                                      # 当前所有 BoxSpan':
        'self._spans = []                                      # All current BoxSpan instances',
    'self._span2spec = {}                                  # BoxSpan → 频谱 LinearRegionItem':
        'self._span2spec = {}                                  # BoxSpan to spectrogram LinearRegionItem mapping',
    'self._span2idx = {}                                   # BoxSpan → annotations 索引':
        'self._span2idx = {}                                   # BoxSpan to annotations index mapping',
    '# —— 编辑模式（双击 BoxSpan 进入）——':
        'Edit mode (double-click a BoxSpan to enter)',
    'self._editing_span = None   # 当前处于编辑态的 BoxSpan':
        'self._editing_span = None   # BoxSpan currently in edit mode',
    'self._selected_span = None  # 当前被点击选中的 BoxSpan（用于 Delete 等快捷键）':
        'self._selected_span = None  # Currently selected BoxSpan (used by Delete and other shortcuts)',
    '# —— STFT 显示参数（配色方案与显示窗口上下限）——':
        'STFT display parameters (color scheme and display window bounds)',
    'self.stft_cmap = "Heatmap"  # 可选："Heatmap"、"Grayscale"':
        'self.stft_cmap = "Heatmap"  # Options: "Heatmap", "Grayscale"',
    'self.stft_vmin = None       # None 表示自动取 1%~99% 分位数':
        'self.stft_vmin = None       # None means auto-compute from the 1st-99th percentile',
    'self._last_spec_vals = None  # 最近一次 STFT 原始数据 (freq×time)，用于直方图统计与重着色':
        'self._last_spec_vals = None  # Most recent STFT raw data (freq x time), used for histogram statistics and recoloring',
    '# —— 短时特征曲线配色（预定义不重复色板，最多支持 10 条曲线）——':
        'Short-time feature curve color palette (predefined, non-repeating; supports up to 10 curves)',
    'self.feature_color_map = {}  # {特征名: QColor}，按所选特征顺序依次分配':
        'self.feature_color_map = {}  # {feature_name: QColor}, assigned sequentially in selection order',
    '# —— 标注类型与颜色管理（内置呼吸音标签及对应颜色）——':
        'Annotation type and color management (built-in respiratory sound labels with corresponding colors)',
    '# 内置标签的固定颜色映射':
        'Fixed color mapping for built-in labels',
    '"Wheeze": QColor("#e41a1c"),       # 红':
        '"Wheeze": QColor("#e41a1c"),       # Red',
    '"Crackles": QColor("#377eb8"),      # 蓝':
        '"Crackles": QColor("#377eb8"),      # Blue',
    '"Pleural Rub": QColor("#4daf4a"),   # 绿':
        '"Pleural Rub": QColor("#4daf4a"),   # Green',
    '"Rhonchi": QColor("#984ea3"),        # 紫':
        '"Rhonchi": QColor("#984ea3"),        # Purple',
    '"Stridor": QColor("#ff7f00"),        # 橙':
        '"Stridor": QColor("#ff7f00"),        # Orange',
    '"Speech": QColor("#a65628"),          # 棕':
        '"Speech": QColor("#a65628"),          # Brown',
    '"Cough": QColor("#f781bf"),           # 粉':
        '"Cough": QColor("#f781bf"),           # Pink',
    '"Expiration": QColor("#999999"),      # 灰':
        '"Expiration": QColor("#999999"),      # Gray',
    '"Inspiration": QColor("#66c2a5"),     # 青绿':
        '"Inspiration": QColor("#66c2a5"),     # Teal',
    '# 自定义标签的自动配色色板':
        'Auto-coloring palette for custom labels',
    '# {标签文本: QColor}':
        '# {label_text: QColor}',
    '# 当前 ML 操作的目标标签（由 init_ml_toolbar 中的下拉框赋值）':
        'Target label for the current ML operation (assigned by the combo box in init_ml_toolbar)',
    "self.last_export_path = None  # 上次导出的 CSV 路径（用于下次导出时沿用同一目录）":
        "self.last_export_path = None  # Last export CSV path (reused as the default directory for the next export)",
    'self.default_export_annotation_name = "annotations_events.csv"  # 当前 WAV 对应的默认导出文件名':
        'self.default_export_annotation_name = "annotations_events.csv"  # Default export filename for the current WAV file',
    'self.wave_y_range = None  # 波形显示范围 (ymin, ymax)':
        'self.wave_y_range = None  # Waveform display range (ymin, ymax)',
    'self.last_settings_tab_index = 0  # 上次打开的 Settings 标签页索引':
        'self.last_settings_tab_index = 0  # Index of the last-opened Settings tab',
    '# —— 音频加载预处理参数（默认启用 4000 Hz 重采样，滤波默认关闭以保持原始音频内容不变）——':
        'Audio loading preprocessing parameters (4000 Hz resampling enabled by default; filtering disabled by default to preserve original audio content)',
    '# —— 同名事件文件自动导入（可开关）——':
        'Auto-import of matching events files (toggleable)',
    '# 规则：与 WAV 同目录下的 <wav_base>_events.(csv|txt)':
        'Rule: look for <wav_base>_events.(csv|txt) in the same directory as the WAV file',
    'self.show_y_axis = False  # 初始隐藏 Y 轴':
        'self.show_y_axis = False  # Initially hide the Y-axis',
    '# —— 短时特征默认选择（可在 Settings 中修改）——':
        'Default short-time feature selection (configurable in Settings)',
    '# 短时特征曲线缓存 {特征名: pg.PlotDataItem}':
        'Short-time feature curve cache {feature_name: pg.PlotDataItem}',
    '# —— 性能优化：切换文件时仅加载波形与 STFT，FFT 与短时特征改为懒加载——':
        'Performance optimization: when switching files, only load waveform and STFT; FFT and short-time features are lazy-loaded on demand',
    '# —— 机器学习相关状态 ——':
        'Machine-learning state',
    'self.stft_frame_times = None    # 帧时间戳 1D array (T,)':
        'self.stft_frame_times = None    # Frame timestamps, 1D array (T,)',
    'self.stft_features = None       # 帧特征 2D array (T, D)':
        'self.stft_features = None       # Frame features, 2D array (T, D)',
    'self.stft_feature_names = None  # 特征名列表 list[str]':
        'self.stft_feature_names = None  # Feature name list, list[str]',
    'self.ml_models = {}             # 已训练模型 {label: {clf, threshold, feature_names, ...}}':
        'self.ml_models = {}             # Trained models {label: {clf, threshold, feature_names, ...}}',
    'self.ml_service = MLService(self)  # ML 训练/推理调度器':
        'self.ml_service = MLService(self)  # ML training/inference dispatcher',
    '# ========= 播放/暂停 (Space) =========':
        '# ========= Play / Pause (Space) =========',
    '# ========= 快退/快进 (Left/Right) =========':
        '# ========= Seek backward / forward (Left / Right) =========',
    '# ========= ML 训练/自动标注：使用当前 ML 下拉框选中的标签 =========':
        '# ========= ML training / auto-labeling: uses the label selected in the ML combo box =========',
    '# ========= 撤销 (Ctrl+Z) =========':
        '# ========= Undo (Ctrl+Z) =========',
    '# 同时注册系统标准 Undo 快捷键（Windows/Linux: Ctrl+Z, macOS: Cmd+Z）':
        'Also register the system-standard Undo shortcut (Windows/Linux: Ctrl+Z, macOS: Cmd+Z)',
    '# 当存在快捷键歧义时，也连接 activatedAmbiguously 以确保撤销生效':
        'When shortcut ambiguity exists, also connect activatedAmbiguously to ensure Undo works',
    '# 额外注册 QAction 形式的 Undo（更贴近 Qt 标准动作系统，某些场景下比 QShortcut 更可靠）':
        'Additionally register an Undo QAction (closer to the Qt standard action system; more reliable than QShortcut in some scenarios)',
    '# 应用级事件过滤器：处理需要复杂逻辑的快捷键（Delete/Ctrl+A/编辑模式等）':
        'Application-level event filter: handles shortcuts requiring complex logic (Delete, Ctrl+A, edit mode, etc.)',
    '# 监听 STFT 图鼠标移动，通过 ViewBox 矩形判断是否在绘图区域内':
        'Listen for mouse movement on the STFT plot; use the ViewBox rectangle to determine if the cursor is inside the plotting area',
    'self.spec_title_base = "STFT Spectrogram"  # 基础标题':
        'self.spec_title_base = "STFT Spectrogram"  # Base title',
    'self._stft_proxy = pg.SignalProxy(  # 监听 STFT 图的鼠标移动':
        'self._stft_proxy = pg.SignalProxy(  # Listen for mouse movement on the STFT plot',
    '# 标注区 Y 轴固定为 3 行高度':
        'Annotation area Y-axis is fixed at 3-lane height',
    '# —— 短时特征页（仅负责显示，计算逻辑由独立模块处理）——':
        'Short-time features page (display only; computation logic is handled by a separate module)',
    '# 仅允许 X 轴缩放/拖动，禁用 Y 轴':
        'Allow only X-axis zoom/pan; disable the Y-axis',
    '# 隐藏 Y 轴（坐标线、刻度、标签均不显示）':
        'Hide the Y-axis (axis line, ticks, and labels are all hidden)',
    '# 关闭网格':
        'Turn off the grid',
    '# 时间轴与 STFT 同步缩放/平移':
        'Synchronize time-axis zoom/pan with the STFT plot',
    '# 去除多余边距，避免隐藏轴后仍留白':
        'Remove extra margins to avoid whitespace after hiding the axis',
    '# —— 页面栈：STFT / FFT / 短时特征 ——':
        'Page stack: STFT / FFT / Short-Time Features',
    'self.spec_stack.addWidget(self.spec_stft_plot)  # 第 0 页：STFT':
        'self.spec_stack.addWidget(self.spec_stft_plot)  # Page 0: STFT',
    'self.spec_stack.addWidget(self.spec_fft_plot)   # 第 1 页：FFT':
        'self.spec_stack.addWidget(self.spec_fft_plot)   # Page 1: FFT',
    'self.spec_stack.addWidget(self.feat_page)        # 第 2 页：短时特征':
        'self.spec_stack.addWidget(self.feat_page)        # Page 2: Short-Time Features',
    'self.spec_stack.setCurrentWidget(self.spec_stft_plot)  # 默认显示 STFT':
        'self.spec_stack.setCurrentWidget(self.spec_stft_plot)  # Default to STFT',
    '# —— 垂直分割器：上（页面栈）| 中（波形）| 下（标注）——':
        'Vertical splitter: top (page stack) | middle (waveform) | bottom (annotations)',
    'self.main_splitter.addWidget(self.spec_stack)   # 上：页面栈':
        'self.main_splitter.addWidget(self.spec_stack)   # Top: page stack',
    'self.main_splitter.addWidget(self.wave_plot)    # 中：波形':
        'self.main_splitter.addWidget(self.wave_plot)    # Middle: waveform',
    'self.main_splitter.addWidget(self.annot_plot)   # 下：标注':
        'self.main_splitter.addWidget(self.annot_plot)   # Bottom: annotations',
    '# 拉伸比例（可按实际需求调整）':
        'Stretch factors (adjustable according to actual needs)',
    '# 禁止各面板被完全折叠':
        'Prevent any panel from being collapsed entirely',
    '"""Create the menu bar: File / Settings / Help."""':
        '"""Create the menu bar: File / Settings / Help."""',
    '# ===== File 菜单 =====':
        '# ===== File menu =====',
    '# ===== Settings 菜单 =====':
        '# ===== Settings menu =====',
    '# ===== Help 菜单 =====':
        '# ===== Help menu =====',
    '"""Initialize the Machine Learning toolbar: label selection + training + auto-labeling + negative sample management."""':
        '"""Initialize the Machine Learning toolbar: label selection + training + auto-labeling + negative sample management."""',
    '# ===== 下拉框：选择待训练/自动标注的标签 =====':
        '# ===== Combo box: select the label to train/auto-label =====',
    '# 使用内置标签列表的英文名作为 ML 标签选项':
        'Use the English names from the built-in label list as ML label options',
    '# 初始化 current_ml_label（若已有值且在列表中则沿用，否则取首个）':
        'Initialize current_ml_label (keep the existing value if it is in the list; otherwise use the first)',
    '# ===== 训练按钮 =====':
        '# ===== Train button =====',
    '# ===== 自动标注按钮 =====':
        '# ===== Auto-label button =====',
    '# ===== 清除负样本按钮 =====':
        '# ===== Clear negatives button =====',
    '# ===== 标注颜色图例按钮 =====':
        '# ===== Annotation color legend button =====',
    '# ===== 工具栏槽函数 =====':
        '# ===== Toolbar slot functions =====',
    '"""comboboxlabel变化时updatecurrent ML 操作目标。"""':
        '"""Update the current ML operation target when the combo-box label changes."""',
    '"""Train button：对current选中labeltrain帧级model。"""':
        '"""Train button: train a frame-level model for the currently selected label."""',
    '"""Auto-label button：用已trainmodel对currentlabel的未reviewregion进行autoannotation。"""':
        '"""Auto-label button: use the trained model to auto-label unreviewed regions for the current label."""',
    '"""清除currentlabel的all硬negative sample。"""':
        '"""Clear all hard negative samples for the current label."""',
    '"""updateClear negatives button的 tooltip，displaycurrentlabel的negative sample数量。"""':
        '"""Update the Clear Negatives button tooltip to show the current label\'s negative sample count."""',
    '"""根据内置label和color映射生成annotationcolor图例。"""':
        '"""Generate the annotation color legend from built-in labels and color mappings."""',
    '"""优先使用内置color映射；size写不一致时做不区分size写的回退匹配。"""':
        '"""Prefer the built-in color mapping; fall back to case-insensitive matching when casing differs."""',
    '# 直接按照代码中的预设标签顺序生成图例，不依赖当前是否已有标注。':
        'Generate the legend directly from the preset label order in the code, without depending on whether annotations currently exist.',
    '# 补充当前File中出现过的Custom标签。':
        'Supplement with any Custom labels that appear in the current file.',
    '"""loadsingle WAV file。\n\n    设计原则：\n    1) default以 4000 Hz resampling作为分析基础sample rate；\n    2) switchfile时仅完成必要操作：读取audio、绘制抽稀waveform、绘制 STFT、importlabel；\n    3) FFT 与短时feature不预先compute，改为switch到对应页面时懒load以提升switch速度。\n    """':
        '"""Load a single WAV file.\n\n    Design principles:\n    1) Resample to 4000 Hz by default as the analysis base sample rate;\n    2) When switching files, perform only essential operations: read audio, draw decimated waveform, draw STFT, import labels;\n    3) FFT and short-time features are not pre-computed; they are lazy-loaded when switching to the corresponding page to improve switching speed.\n    """',
    '# 0. 切换文件前停止当前播放，避免旧音频仍占用声卡。':
        '# 0. Stop playback before switching files to prevent the old audio from still occupying the sound card.',
    '# 1. 手动导入路径为空时，弹出File选择框':
        '# 1. If no path is provided, show the file selection dialog',
    '# 2. 建立同目录 WAV 列表，用于上一首/下一首。':
        '# 2. Build a WAV-file list from the same directory for previous/next navigation.',
    '# 3. 读取音频。默认使用 Settings 中的重采样采样率；关闭重采样时保留原始采样率。':
        '# 3. Read the audio. Use the resample rate from Settings by default; preserve the original sample rate when resampling is disabled.',
    '# 4. 切换文件时，先清空旧缓存和旧标注，避免旧图/旧特征误用。':
        '# 4. When switching files, clear old caches and old annotations first to avoid reusing stale plots/features.',
    '# 新音频默认重置自动色阶，避免上一条文件的色阶影响当前文件。':
        'Reset auto-levels by default for the new audio to prevent the previous file\'s color levels from affecting the current file.',
    '# 先清旧标注，再重绘新图，防止旧 STFT 高亮残留。':
        'Clear old annotations first, then redraw new plots to prevent residual old STFT highlights.',
    '# 5. 只做必要显示：抽稀波形 + STFT。FFT/特征延后到用户切换页面时计算。':
        '# 5. Show only essentials: decimated waveform + STFT. FFT and features are deferred until the user switches pages.',
    '# 6. 自动导入标签。标签导入很轻，仍保留在 load_audio 里。':
        '# 6. Auto-import labels. Label import is lightweight, so it stays inside load_audio.',
    '# 7. 坐标范围与页面状态。':
        '# 7. Coordinate ranges and page state.',
    '# 保存完整谱值用于 Settings 直方图；显示用谱图可以抽稀。':
        'Save full spectrogram values for the Settings histogram; the display spectrogram may be decimated.',
    '# 抽稀后的图像仍映射到完整音频时长，保证标注时间轴一致。':
        'The decimated image still maps to the full audio duration, ensuring consistent annotation time axes.',
    '# STFT/FMAX 变化会影响频域特征，但不在 STFT 刷新时立即计算特征。':
        'Changes to STFT/f_max affect frequency-domain features, but features are not recomputed immediately during STFT refresh.',
    '"""清空所有annotation：移除可视化对象、清空data结构、复位视图range。"""':
        '"""Clear all annotations: remove visual objects, clear data structures, reset view ranges."""',
    '# 1) 移除可视化对象 —— BoxSpan 及其上方文字标签':
        '# 1) Remove visual objects -- BoxSpan instances and their text labels',
    'sp.cleanup()  # 从 annot_plot 移除自身及文字标签':
        'sp.cleanup()  # Remove itself and its text label from annot_plot',
    '# 2) 移除 STFT 频谱图上的高亮区域':
        '# 2) Remove highlight regions from the STFT spectrogram',
    '# 3) 兼容旧格式：清除 annotation_items 中保存的区域和文字':
        '# 3) Backward compatibility: clear regions and text stored in annotation_items',
    '# 4) 清除拖拽过程中的临时选区':
        '# 4) Clear temporary selection regions from an in-progress drag',
    '# 5) 清空数据结构':
        '# 5) Clear data structures',
    '# 6) 复位标注视图范围':
        '# 6) Reset the annotation view range',
    '# 7) 取消波形高亮':
        '# 7) Cancel waveform highlight',
    '"""seek backward/fast forward delta_sec 秒（供 Left/Right 快捷键调用）。"""':
        '"""Seek backward/forward by delta_sec seconds (called by the Left/Right shortcut keys)."""',
    '# ↓↓↓ 新增：把 STFT 相关Display数据传进去':
        '# New: pass STFT-related display data in',
    'dlg.set_current_tab(self.last_settings_tab_index)  # 💡 Settings上次使用的标签页':
        'dlg.set_current_tab(self.last_settings_tab_index)  # Last-used Settings tab',
    'self.selected_features = dlg.get_selected_features()  # 新增':
        'self.selected_features = dlg.get_selected_features()  # New',
    '# 新增：保存"读取时预处理"和"自动标签读取参数"。不立即重读当前音频，避免影响现有编辑状态；下次 load_audio 生效。':
        '# New: save "load-time preprocessing" and "auto label import" parameters. Do not immediately reload the current audio to avoid disrupting the current editing state; takes effect on the next load_audio call.',
    '# 取回 STFT DisplaySettings (配色 + 上下限)，先保存再重绘，避免使用旧色阶。':
        'Retrieve STFT display settings (color scheme + bounds); save before redrawing to avoid using old color levels.',
    '# Settings 改变后只刷新波形/STFT，并标记 FFT/特征失效；不立即计算短时特征。':
        'After Settings changes, only refresh the waveform/STFT and mark FFT/features as dirty; do not recompute short-time features immediately.',
    '# 若只改了色阶而未重算，也可以直接按当前谱图重着色。':
        'If only the color levels were changed without recomputing, simply recolor the current spectrogram in-place.',
    '"""根据current WAV 实时updatedefaultexportFile名。\n    只updateFile名，不决定directory。directory在export时优先沿用previous次Export Annotations的directory。\n    命名规则：<currentwavFile名不含扩展名>_events.csv\n    """':
        '"""Update the default export filename in real time based on the current WAV file.\n    Only the filename is updated; the directory is not determined here. On export, the directory defaults to the last Export Annotations directory.\n    Naming convention: <current_wav_basename_without_extension>_events.csv\n    """',
    '"""生成exportdialogdefaultpath：directory跟随上次export，File名跟随current WAV。"""':
        '"""Generate the default path for the export dialog: directory follows the last export, filename follows the current WAV."""',
    '"""exportannotationdata到 CSV file。"""':
        '"""Export annotation data to a CSV file."""',
    "# 过滤 None 项，兼容 3/4 元组格式；排除 source='archived'（已归档/隐藏）的标注":
        "# Filter out None entries; support 3-/4-tuple formats; exclude annotations whose source is 'archived' (archived/hidden)",
    '# 默认路径：目录沿用上次导出位置，文件名跟随当前 WAV':
        'Default path: directory reuses the last export location; filename follows the current WAV',
    '# 例如：当前音频 1008.wav → 默认文件名 1008_events.csv':
        'Example: current audio 1008.wav -> default filename 1008_events.csv',
    '# === 验证：打印导出 source 分布统计（可在控制台查看）===':
        '# === Verification: print exported source-distribution statistics (viewable in the console) ===',
    'self.last_export_path = path  # 记住路径':
        'self.last_export_path = path  # Remember the path',
    '"""delete一个annotation。\n    - 交互式delete BoxSpan 时：default将该段加入 neg_segments（仅用于train的硬negative sample），\n    且support Ctrl+Z undo。\n    - 程序内部清理（如清除 ML predict结果）可传 record_negative=False, push_undo=False。\n    """':
        '"""Delete an annotation.\n    - When interactively deleting a BoxSpan: by default, add the segment to neg_segments (hard-negative samples for training only),\n    and support Ctrl+Z undo.\n    - Internal cleanup (e.g., clearing ML predictions) can pass record_negative=False, push_undo=False.\n    """',
    '# 删除 BoxSpan 对象':
        '# Remove the BoxSpan object',
    '# 记录硬负样本（按被删除标注的原始标签，而非当前训练类型）':
        'Record a hard negative sample (using the deleted annotation\'s original label, not the current training label)',
    '# 移除 UI 组件':
        '# Remove UI components',
    '# 记录撤销信息':
        '# Record undo information',
    '# 兼容：旧 LinearRegionItem 删除路径 (如果你代码中还有)':
        'Backward compatibility: old LinearRegionItem deletion path (if still present in your code)',
    '# 若传入的是 STFT 高亮区域 (LinearRegionItem)，尝试反查对应的 BoxSpan，':
        'If the input is an STFT highlight region (LinearRegionItem), attempt to reverse-lookup the corresponding BoxSpan,',
    '# 统一走 BoxSpan 删除逻辑，以便支持硬负样本与 Ctrl+Z 撤销。':
        'and route through the unified BoxSpan deletion logic to support hard-negative samples and Ctrl+Z undo.',
    '# 复用 BoxSpan 删除逻辑':
        'Reuse the BoxSpan deletion logic',
    '"""根据 annotations[idx] Rebuild BoxSpan 可视化（仅供undo使用），不修改 annotations。"""':
        '"""Rebuild the BoxSpan visualization from annotations[idx] (for undo only); does not modify annotations."""',
    '# 若已经有同 idx 的 span (理论上不该发生)，直接返回避免重复':
        'If a span with the same index already exists (should not happen in theory), return immediately to prevent duplication',
    '"""创建并渲染一个annotation的 BoxSpan + STFT highlightregion。\n    共用，消除重复的Lane assignment/color/笔刷/高亮逻辑。\n    """':
        '"""Create and render a BoxSpan plus STFT highlight region for an annotation.\n    Shared by both, eliminating duplicated lane-assignment / color / pen / highlight logic.\n    """',
    '# 分轨':
        '# Lane assignment',
    '# 颜色与笔刷':
        '# Color and pen',
    '# 非 manual source 使用红色文字':
        'Non-manual source annotations use red text',
    '# STFT 高亮':
        '# STFT highlight',
    '"""仅清空可视化对象（BoxSpan、STFT highlight等），不清空 annotations/neg/undo 等data。\n    用于视图与data不synchronize时的Rebuild兜底。\n    """':
        '"""Clear only visual objects (BoxSpan, STFT highlights, etc.); do not clear annotations/neg/undo data.\n    Used as a rebuild fallback when the view and data are out of sync.\n    """',
    '# 1) BoxSpan 及其文字标签':
        '# 1) BoxSpan instances and their text labels',
    '# 兜底：至少从图中移除':
        'Fallback: at least remove it from the plot',
    '# 2) STFT 高亮区域':
        '# 2) STFT highlight regions',
    '# 3) 索引映射':
        '# 3) Index mapping',
    '# 4) 兼容旧格式残留':
        '# 4) Backward compatibility with legacy format remnants',
    '# 5) 临时拖拽选区':
        '# 5) Temporary drag-selection region',
    '# 6) 重置轨道缓存（不影响数据）':
        '# 6) Reset lane cache (does not affect data)',
    '"""从 self.annotations 重新渲染所有 BoxSpan（不改动 annotations data）。"""':
        '"""Rebuild all BoxSpan visualizations from self.annotations (does not modify annotation data)."""',
    '# 清空视图':
        '# Clear the view',
    '# 复位 Y 轴显示范围（避免渲染后不可见）':
        'Reset the Y-axis display range (to avoid invisibility after rendering)',
    '# 重建':
        '# Rebuild',
    '# archived 不Display也不导出 (合并/删除前的旧 span)':
        'Archived spans are neither displayed nor exported (old spans before merge/delete)',
    '# 避免异常中断主流程':
        'Avoid interrupting the main flow with exceptions',
    '"""将指定annotation的 source 修改为 new_source 并synchronize UI，supportundo。"""':
        '"""Change the source of a specified annotation to new_source, sync the UI, and support undo."""',
    '"""undo最后一次edit操作（delete、改 source、edit几何position）。"""':
        '"""Undo the last edit operation (delete, source change, geometry edit)."""',
    '"""undo一次delete：restore annotations 条目 + Rebuild可视化 + 移除negative sample。"""':
        '"""Undo a single deletion: restore the annotations entry + rebuild visualization + remove the negative sample."""',
    '# 兜底：恢复失败则重建整个视图':
        'Fallback: if restoration fails, rebuild the entire view',
    '"""undo一次几何edit：回滚position、text、style和高亮。"""':
        '"""Undo a single geometry edit: roll back position, text, style, and highlight."""',
    '# span 丢失则重建整个视图':
        'If the span is lost, rebuild the entire view',
    '# STFT 高亮同步':
        '# Sync STFT highlight',
    '# 兜底：回滚失败则重建视图':
        'Fallback: if rollback fails, rebuild the view',
    '"""undo source 变更：restore旧 source 并重绘视觉style。"""':
        '"""Undo a source change: restore the old source and redraw the visual style."""',
    '"""将机器annotationmark为「已认可」，default source=auto_accepted，supportundo。"""':
        '"""Mark a machine annotation as accepted; default source=auto_accepted; undoable."""',
    '# BoxSpan 编辑模式：双击进入；Enter 提交；Esc 取消':
        'BoxSpan edit mode: double-click to enter; Enter to commit; Esc to cancel',
    '"""记录current进入edit态的 BoxSpan（若已有其他 span 在edit则先commit退出）。"""':
        '"""Record the BoxSpan currently entering edit mode (if another span is already in edit mode, commit and exit it first)."""',
    '"""退出editmode并清空editstatus。"""':
        '"""Exit edit mode and clear the editing state."""',
    '# Exit编辑后清理状态栏Notice':
        'Clean the status bar notice after exiting edit mode',
    '"""edit态：在status栏displaycurrent选中 span 的起止时间。"""':
        '"""Edit mode: display the start and end times of the currently selected span in the status bar."""',
    '"""根据labeltext返回稳定的color：内置label使用固定color，自定义label使用autoColor scheme。"""':
        '"""Return a stable color for a label text: built-in labels use fixed colors; custom labels use auto-assigned colors."""',
    '# 已有映射直接返回':
        'Return directly if already mapped',
    '# 预设类型 (英文名)':
        'Preset type (English name)',
    '# 其他任意文本：从调色板顺序取色':
        'Any other text: pick a color sequentially from the palette',
    '# 若未提供文本 (正常交互标注)，则弹出带预设类型的对话框':
        'If no text is provided (normal interactive annotation), show the dialog with preset types',
    '# —— 渲染标注可视化 + 频谱高亮 ——':
        'Render annotation visualization + spectrogram highlight',
    '# —— 注册到 annotations 数据 ——':
        'Register into the annotations data store',
    '# 自动导入同名 _events 标注 (可开关)':
        'Auto-import matching _events annotations (toggleable)',
    '# STFT → FFT：触发懒加载':
        'STFT -> FFT: trigger lazy loading',
    '# FFT -&gt; Short-Time Features：懒加载短时特征':
        'FFT -> Short-Time Features: lazy-load short-time features',
    '# 只要有时间交集即视为重叠':
        'Any temporal intersection is considered an overlap',
    '"""根据current可见 BoxSpan 动态select空闲轨道；优先使用编号最小的空轨。"""':
        '"""Dynamically select a free lane based on the currently visible BoxSpan instances; prefer the lane with the smallest index."""',
    '# 收集现有每一行的区间':
        'Collect the intervals of every existing lane',
    '# 找第一个不重叠的行':
        'Find the first non-overlapping lane',
    '# 三行都冲突，则放最后一行':
        'If all three lanes conflict, place it in the last lane',
    '"""鼠标在 STFT 图上移动时display t/f 坐标；移出绘图region则还原标题。"""':
        '"""Display time/frequency coordinates when the mouse moves over the STFT plot; restore the title when the mouse leaves the plotting area."""',
    '# 仅在绘图区内触发 (不包含轴刻度/边距)':
        'Trigger only inside the plotting area (excluding axis ticks/margins)',
    '# 标题右侧Display坐标 (可调样式)':
        'Display coordinates on the right side of the title (style adjustable)',
    '"""根据 self.selected_features compute并绘制短时feature曲线（0-1 归一化叠加display）。\n    说明：该函数在主线程compute，仅在switch到feature页时懒load；\n    load_audio / previous首 / next首不主动调用。\n    """':
        '"""Compute and plot short-time feature curves based on self.selected_features (0-1 normalized, overlaid display).\n    Note: this function computes on the main thread and is only lazy-loaded when switching to the features page;\n    it is not called proactively during load_audio / previous / next.\n    """',
    '# 画最多 5 items':
        'Plot at most 5 items',
    'self._assign_feature_colors(names)  # 选择颜色':
        'self._assign_feature_colors(names)  # Assign colors',
    '# Time轴范围与 STFT 对齐':
        'Align the time-axis range with STFT',
    '"""switchColor scheme方案：不重新compute STFT，仅对display用谱图重着色。"""':
        '"""Switch the color scheme: do not recompute the STFT; only recolor the display spectrogram."""',
    'disp = spec_disp.T  # 与主图方向一致：time × freq':
        'disp = spec_disp.T  # Match the main plot orientation: time x freq',
    '"""按 selected_names 顺序为每条feature曲线分配不重复的color。"""':
        '"""Assign a unique color to each feature curve in the order of selected_names."""',
    '统一遍历「可用于训练 / 已审阅」的标注区间：\n    - 兼容 (start, end, label) 和 (start, end, label, source) 两种格式\n    - 三元组一律视为人工标注（source=\'manual\'）\n    - 四元组根据 source 是否属于「已审阅集合」判定\n    说明：为兼容旧代码保留函数名 _iter_manual_annotations，\n    但其语义已升级为「reviewed annotations」。':
        'Unified iteration over annotation intervals that are "trainable / reviewed":\n    - Compatible with both (start, end, label) and (start, end, label, source) formats\n    - 3-tuples are always treated as manual annotations (source=\'manual\')\n    - 4-tuples are judged based on whether source belongs to the "reviewed set"\n    Note: the function name _iter_manual_annotations is retained for backward compatibility,\n    but its semantics have been upgraded to "reviewed annotations."',
    '# 先搭"source 状态机"的基础：后续会逐步引入更多状态。':
        'Lay the foundation for the "source state machine": more states will be introduced gradually in the future.',
    '# reviewed: 参与"已审阅前缀"统计；trainable: 可进入下一轮训练的正样本标注。':
        'reviewed: participates in the "reviewed prefix" statistic; trainable: a positive-sample annotation eligible for the next training round.',
    '# 兼容三元组 / 四元组格式，其他长度直接跳过':
        'Support 3-tuple / 4-tuple formats; skip other lengths',
    'src = "manual"  # 三元组直接视为人工标记':
        'src = "manual"  # 3-tuples are treated directly as manual labels',
    's, e, t, src = item[:4]  # 四元组用自带的 source':
        's, e, t, src = item[:4]  # 4-tuples use their own source field',
    '"""应用级eventfilter器：handle需要复杂逻辑的快捷键。\n    包括：Ctrl+Z undo（兜底）、editmode Enter/Esc、\n    Delete delete选中annotation、Ctrl+A 认可 ML annotation。\n    """':
        '"""Application-level event filter: handles shortcuts that require complex logic.\n    Includes: Ctrl+Z undo (fallback), edit-mode Enter/Esc,\n    Delete to remove the selected annotation, Ctrl+A to accept an ML annotation.\n    """',
    '# Ctrl+Z 撤销':
        '# Ctrl+Z undo',
    '# Ctrl+A 认可选中的 ML 标注':
        '# Ctrl+A accept the selected ML annotation',
    '# Delete 键删除选中的标注':
        '# Delete key removes the selected annotation',
    '# 编辑模式键盘：Enter 提交 / Esc 取消':
        '# Edit-mode keyboard: Enter to commit / Esc to cancel',
}

TEST_TRANSLATIONS = {
    # Common patterns repeated across many test files
    "验证空输入或 None 输入时的行为。": "Verify behavior with empty or None input.",
    "验证默认参数值符合预期。": "Verify default parameter values match expectations.",
    "验证标注的 source 溯源信息正确保留。": "Verify annotation source provenance is correctly preserved.",
    "验证 butter filter 的确定性：相同输入产生逐位相同的输出。": "Verify butter filter determinism: same input produces bitwise-identical output.",
    "验证 config validation 的确定性：相同输入产生逐位相同的输出。": "Verify config validation determinism: same input produces bitwise-identical output.",
    "验证 fft 的确定性：相同输入产生逐位相同的输出。": "Verify FFT determinism: same input produces bitwise-identical output.",
    "验证 stft 的确定性：相同输入产生逐位相同的输出。": "Verify STFT determinism: same input produces bitwise-identical output.",
    "验证 short time features 的确定性：相同输入产生逐位相同的输出。": "Verify short-time features determinism: same input produces bitwise-identical output.",
    "验证 feature matrix 的确定性：相同输入产生逐位相同的输出。": "Verify feature matrix determinism: same input produces bitwise-identical output.",
    "验证 build frame labels 的确定性：相同输入产生逐位相同的输出。": "Verify build_frame_labels determinism: same input produces bitwise-identical output.",
    "验证 log trans 的确定性：相同输入产生逐位相同的输出。": "Verify log_trans determinism: same input produces bitwise-identical output.",
    "验证 hsmm viterbi 的确定性：相同输入产生逐位相同的输出。": "Verify HSMM Viterbi determinism: same input produces bitwise-identical output.",
    "验证 hsmm prior 的确定性：相同输入产生逐位相同的输出。": "Verify HSMM prior determinism: same input produces bitwise-identical output.",
    "验证：np.allclose(a1, a2)。": "Verify np.allclose(a1, a2).",
    "验证：not np.allclose(a1, a2)。": "Verify not np.allclose(a1, a2).",
    "验证完整 ML 管线（合成音频→特征→训练→推理）运行无异常。": "Verify the complete ML pipeline (synthetic audio -> features -> train -> inference) runs without error.",
    "验证 full pipeline 的确定性：相同输入产生逐位相同的输出。": "Verify full-pipeline determinism: same input produces bitwise-identical output.",
    "验证帧标签构建与实际特征矩阵的维度对齐。": "Verify frame label construction aligns dimensionally with actual feature matrix.",
    "验证计算延迟在可接受范围内。": "Verify compute latency is within acceptable range.",
    "测量完整频谱显示管线（STFT→抽稀→着色）的延迟。": "Measure full spectrogram display pipeline (STFT -> decimation -> colorization) latency.",
    "验证 legacy/1.6.6.py 不包含对 respanno 模块的引用（保持原始快照）。": "Verify legacy/1.6.6.py contains no references to respanno modules (keep original snapshot).",
    "验证 1.6.6.py 的 AST 中包含对 respanno.annotation.io 的 import 语句。": "Verify 1.6.6.py AST contains import statements for respanno.annotation.io.",
    "生成 WAV → load+preprocess → annotate → export CSV → re-import": "Generate WAV -> load+preprocess -> annotate -> export CSV -> re-import",
}

# ── Apply translations ─────────────────────────────────────────────────

def apply_translations(filepath, translations, dry_run=True):
    """Apply exact-string replacements to a file."""
    if not os.path.exists(filepath):
        print(f"  [MISSING] {filepath}")
        return 0

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    changed = 0
    new_content = content
    for old, new in translations.items():
        if old in new_content:
            new_content = new_content.replace(old, new)
            changed += 1
        # also try with CRLF line endings
        elif old.replace('\n', '\r\n') in new_content:
            new_content = new_content.replace(old.replace('\n', '\r\n'), new)
            changed += 1

    if changed > 0:
        if not dry_run:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
        print(f"  {'[DRY RUN]' if dry_run else '[WRITTEN]'} {os.path.relpath(filepath, ROOT)} ({changed} changes)")
    else:
        print(f"  [SKIP] {os.path.relpath(filepath, ROOT)} (no matches)")

    return changed


def main():
    import sys
    dry_run = '--write' not in sys.argv
    mode = "DRY RUN" if dry_run else "WRITE MODE"
    print(f"=== Apply Chinese->English Translations ({mode}) ===\n")

    total_files = 0
    total_changes = 0

    # 1. Main file
    print("--- 1.6.6.py ---")
    fpath = os.path.join(ROOT, '1.6.6.py')
    n = apply_translations(fpath, MAIN_TRANSLATIONS, dry_run)
    total_files += 1
    total_changes += n

    # 2. GUI files
    print("\n--- GUI files ---")
    for relpath, translations in GUI_TRANSLATIONS.items():
        fpath = os.path.join(ROOT, relpath)
        n = apply_translations(fpath, translations, dry_run)
        total_files += 1
        total_changes += n

    # 3. Test files
    print("\n--- Test files ---")
    test_dir = os.path.join(ROOT, 'tests')
    for root, dirs, files in os.walk(test_dir):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', 'fixtures')]
        for fname in files:
            if not fname.endswith('.py'):
                continue
            fpath = os.path.join(root, fname)
            n = apply_translations(fpath, TEST_TRANSLATIONS, dry_run)
            total_files += 1
            total_changes += n

    print(f"\n=== Done: {total_changes} replacements across {total_files} files ===")
    if dry_run:
        print("Run with --write to apply changes.")


if __name__ == '__main__':
    main()
