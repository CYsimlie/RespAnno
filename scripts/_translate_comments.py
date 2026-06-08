"""Translate Chinese comments and docstrings to English in Python files.

Focuses on:
  1. # comment lines containing Chinese
  2. Docstring lines containing Chinese
  3. Inline Chinese in non-string, non-variable-name positions

Preserves:
  - Chinese *data*: feature names like "谱centroid", built-in labels like "哮鸣音"
  - String literals used as keys or labels (these are content, not comments)

Dry-run mode (default): prints what WOULD change without writing.
"""
import os, re, json

ROOT = r'd:\SoftwareX_win_test\New folder2\SoftwareX'
FILES_TO_PROCESS = [
    '1.0.0.py',
    'respanno/gui/dialogs/settings_dialog.py',
    'respanno/gui/spans/box_span.py',
    'respanno/dsp/features.py',
    'respanno/gui/widgets/color_check_delegate.py',
    'respanno/gui/views/annot_view_box.py',
    'respanno/gui/dialogs/annotation_label_dialog.py',
    'respanno/gui/widgets/color_bar.py',
    'respanno/gui/spans/span_label_item.py',
    'respanno/gui/views/wave_view_box.py',
    'respanno/gui/widgets/clickable_slider.py',
    'respanno/ml/label_taxonomy.py',
    'respanno/ml/service.py',
    'tests/test_annotation_roundtrip.py',
    'tests/test_preprocessing_basic.py',
    'tests/test_features_basic.py',
    'tests/test_spectrogram_basic.py',
    'tests/test_gui_static_integration.py',
    'tests/test_negatives_basic.py',
    'tests/test_annotation_quality.py',
    'tests/test_hsmm_basic.py',
    'tests/test_events_importer_basic.py',
    'tests/test_classifier_training_basic.py',
    'tests/test_frame_labels_basic.py',
    'tests/test_phase_model_basic.py',
    'tests/test_reproducibility.py',
    'tests/test_label_taxonomy_basic.py',
    'tests/test_phase_apply_basic.py',
    'tests/test_classifier_apply_basic.py',
    'tests/test_module_imports.py',
    'tests/test_e2e_ml_pipeline.py',
    'tests/test_ml_service_basic.py',
    'tests/test_gui_widgets_headless.py',
    'tests/test_icbhi_compatibility.py',
    'tests/test_roundtrip_workflow.py',
]

# ── Chinese→English phrase translations ──────────────────────────────────
# Each entry: (chinese_pattern, english_replacement)
PHRASES = [
    # ---- Common patterns ----
    ("验证", "verify"),
    ("测试", "test"),
    ("检查", "check"),
    ("确保", "ensure"),
    ("计算", "compute"),
    ("返回", "return"),
    ("创建", "create"),
    ("加载", "load"),
    ("保存", "save"),
    ("读取", "read"),
    ("写入", "write"),
    ("导出", "export"),
    ("导入", "import"),
    ("删除", "delete"),
    ("更新", "update"),
    ("选择", "select"),
    ("取消", "cancel"),
    ("提交", "commit"),
    ("撤销", "undo"),
    ("配置", "configure"),
    ("初始化", "initialize"),
    ("显示", "display"),
    ("隐藏", "hide"),
    ("启用", "enable"),
    ("禁用", "disable"),
    ("支持", "support"),
    ("忽略", "ignore"),
    ("处理", "handle"),
    ("包含", "contain"),
    ("设置", "set"),
    ("获取", "get"),
    ("监听", "listen"),
    ("触发", "trigger"),
    ("切换", "switch"),
    ("恢复", "restore"),
    ("缩放", "zoom"),
    ("重置", "reset"),
    ("过滤", "filter"),
    ("同步", "synchronize"),
    ("默认", "default"),
    ("自动", "auto"),
    ("手动", "manual"),
    ("参数", "parameter"),
    ("输入", "input"),
    ("输出", "output"),
    ("错误", "error"),
    ("警告", "warning"),
    ("信息", "info"),
    ("消息", "message"),
    ("状态", "status"),
    ("进度", "progress"),
    ("范围", "range"),
    ("大小", "size"),
    ("位置", "position"),
    ("颜色", "color"),
    ("字体", "font"),
    ("文本", "text"),
    ("标签", "label"),
    ("按钮", "button"),
    ("菜单", "menu"),
    ("对话框", "dialog"),
    ("窗口", "window"),
    ("面板", "panel"),
    ("工具栏", "toolbar"),
    ("标签页", "tab"),
    ("复选框", "checkbox"),
    ("下拉框", "combobox"),
    ("滑块", "slider"),
    ("输入框", "input box"),
    ("文本框", "text field"),
    ("列表", "list"),
    ("表格", "table"),
    ("树形", "tree"),
    ("图标", "icon"),
    ("信号", "signal"),
    ("槽", "slot"),
    ("事件", "event"),
    ("回调", "callback"),
    ("线程", "thread"),
    ("进程", "process"),
    ("文件", "file"),
    ("目录", "directory"),
    ("路径", "path"),
    ("名称", "name"),
    ("标题", "title"),
    ("内容", "content"),
    ("格式", "format"),
    ("版本", "version"),
    ("数据", "data"),
    ("类型", "type"),
    ("字段", "field"),
    ("列", "column"),
    ("行", "row"),
    ("键", "key"),
    ("值", "value"),
    ("选项", "option"),
    ("模式", "mode"),
    ("布局", "layout"),
    ("样式", "style"),
    ("主题", "theme"),
    ("语言", "language"),
    ("区域", "region"),
    ("区间", "interval"),
    ("边界", "boundary"),
    ("上限", "upper limit"),
    ("下限", "lower limit"),
    ("当前", "current"),
    ("上一", "previous"),
    ("下一", "next"),
    ("全部", "all"),
    ("部分", "partial"),
    ("单个", "single"),
    ("多个", "multiple"),
    ("第一个", "first"),
    ("最后一个", "last"),
    ("中间", "middle"),
    ("开始", "start"),
    ("结束", "end"),
    ("暂停", "pause"),
    ("播放", "play"),
    ("停止", "stop"),
    ("前进", "forward"),
    ("后退", "backward"),
    ("快进", "fast forward"),
    ("快退", "fast backward"),
    ("上一首", "previous track"),
    ("下一首", "next track"),
    ("音频", "audio"),
    ("视频", "video"),
    ("图像", "image"),
    ("波形", "waveform"),
    ("频谱", "spectrum"),
    ("频谱图", "spectrogram"),
    ("特征", "feature"),
    ("分类器", "classifier"),
    ("模型", "model"),
    ("训练", "train"),
    ("推理", "inference"),
    ("预测", "predict"),
    ("标注", "annotation"),
    ("标记", "mark"),
    ("选择区", "selection"),
    ("编辑", "edit"),
    ("编辑器", "editor"),
    ("编辑态", "edit mode"),
    ("非编辑态", "non-edit mode"),
    ("未编辑", "unedited"),
    ("已编辑", "edited"),
    ("审阅", "review"),
    ("未审阅", "unreviewed"),
    ("已审阅", "reviewed"),
    ("前缀", "prefix"),
    ("后缀", "suffix"),
    ("采样率", "sample rate"),
    ("重采样", "resampling"),
    ("滤波", "filtering"),
    ("低通", "lowpass"),
    ("高通", "highpass"),
    ("带通", "bandpass"),
    ("带阻", "bandstop"),
    ("截止频率", "cutoff frequency"),
    ("阶数", "order"),
    ("零相位", "zero-phase"),
    ("帧", "frame"),
    ("帧数", "frame count"),
    ("帧索引", "frame index"),
    ("帧时刻", "frame time"),
    ("时域", "time domain"),
    ("频域", "frequency domain"),
    ("幅度", "magnitude"),
    ("相位", "phase"),
    ("能量", "energy"),
    ("功率", "power"),
    ("过零率", "zero-crossing rate"),
    ("峰度", "kurtosis"),
    ("偏度", "skewness"),
    ("熵", "entropy"),
    ("通量", "flux"),
    ("质心", "centroid"),
    ("带宽", "bandwidth"),
    ("滚降", "roll-off"),
    ("平坦度", "flatness"),
    ("衰减", "decrease"),
    ("斜度", "slope"),
    ("峰", "crest"),
    ("子带", "sub-band"),
    ("自相关", "autocorrelation"),
    ("互相关", "cross-correlation"),
    ("周期性", "periodicity"),
    ("延迟", "latency"),
    ("吞吐量", "throughput"),
    ("内存", "memory"),
    ("占用", "footprint"),
    ("基线", "baseline"),
    ("性能", "performance"),
    ("回归", "regression"),
    ("分类", "classification"),
    ("评分", "score"),
    ("阈值", "threshold"),
    ("准确率", "accuracy"),
    ("精确率", "precision"),
    ("召回率", "recall"),
    ("F1 分数", "F1 score"),
    ("AUROC", "AUROC"),
    ("AUPRC", "AUPRC"),
    ("Brier 分数", "Brier score"),
    ("混淆矩阵", "confusion matrix"),
    ("混淆", "confusion"),
    ("矩阵", "matrix"),
    ("正样本", "positive sample"),
    ("负样本", "negative sample"),
    ("硬负样本", "hard negative sample"),
    ("软负样本", "soft negative sample"),
    ("误标", "mislabel"),
    ("误报", "false positive"),
    ("漏报", "false negative"),
    ("管线", "pipeline"),
    ("调度", "dispatch"),
    ("解析", "parse"),
    ("分隔符", "delimiter"),
    ("检测", "detect"),
    ("换行符", "newline"),
    ("编码", "encoding"),
    ("解码", "decode"),
    ("编码器", "encoder"),
    ("解码器", "decoder"),
    ("掩码", "mask"),
    ("序列", "sequence"),
    ("状态", "state"),
    ("转移", "transition"),
    ("转移矩阵", "transition matrix"),
    ("持续时间", "duration"),
    ("先验", "prior"),
    ("后验", "posterior"),
    ("对数", "log"),
    ("概率", "probability"),
    ("发射", "emission"),
    ("发射概率", "emission probability"),
    ("初始分布", "initial distribution"),
    ("Viterbi 解码", "Viterbi decode"),
    ("Viterbi", "Viterbi"),
    ("HSMM", "HSMM"),
    ("LightGBM", "LightGBM"),
    ("STFT", "STFT"),
    ("FFT", "FFT"),
    ("dB", "dB"),
    ("Hz", "Hz"),
    ("kHz", "kHz"),
    ("秒", "seconds"),
    ("毫秒", "milliseconds"),
    ("分", "minute"),
    ("小时", "hour"),
    ("天", "day"),
    ("周", "week"),
    ("月", "month"),
    ("年", "year"),
    # ---- Phrases / sentences ----
    ("相同输入产生逐位相同的输出", "same input produces bitwise-identical output"),
    ("相同输入 + 相同种子 => 相同输出", "same input + same seed => identical output"),
    ("计算延迟在可接受范围内", "compute latency is within acceptable range"),
    ("标注的 source 溯源信息正确保留", "annotation source provenance is correctly preserved"),
    ("生成 WAV → load+preprocess → annotate → export CSV → re-import", "generate WAV → load+preprocess → annotate → export CSV → re-import"),
    ("标注溯源信息不变（手工/导入/ML 生成）", "annotation provenance is invariant (manual/imported/ML-generated)"),
    ("显示在条上方", "displayed above the bar"),
    ("不直接显示文字，标签含义统一通过工具栏查看", "label text is not shown directly on bar; use toolbar legend instead"),
    ("视觉编码", "visual encoding"),
    ("只改变视觉样式", "only changes visual style"),
    ("来源", "source"),
    ("溯源", "provenance"),
]

# ── detect Chinese ───────────────────────────────────────────────────────
CHINESE_RE = re.compile(r'[一-鿿　-〿＀-￯]')

def has_chinese(s):
    return bool(CHINESE_RE.search(s))

# ── translate a single line ──────────────────────────────────────────────
def translate_line(line):
    """Replace Chinese phrases in `line` with English equivalents."""
    result = line
    for cn, en in PHRASES:
        if cn in result:
            result = result.replace(cn, en)
    return result

# ── process one file ─────────────────────────────────────────────────────
def process_file(fpath, dry_run=True):
    with open(fpath, 'r', encoding='utf-8') as f:
        original = f.read()

    lines = original.split('\n')
    changed = []
    modified = False

    for i, line in enumerate(lines):
        if has_chinese(line):
            # Skip Chinese data: feature names, built-in labels in lists/tuples/dicts
            stripped = line.strip()
            # Keep lines that are purely data (variable assignments with Chinese in string literals)
            # Only translate comment lines and docstrings
            is_comment = stripped.startswith('#')
            is_docstring_line = ('"""' in stripped or "'''" in stripped)
            is_feature_data = any(kw in stripped for kw in ['FEATURE_NAMES', 'builtin_labels', 'BUILTIN_LABELS', '("', "('"])

            if is_comment or is_docstring_line:
                new_line = translate_line(line)
                if new_line != line:
                    changed.append((i+1, line.strip()[:100], new_line.strip()[:100]))
                    lines[i] = new_line
                    modified = True
            elif is_feature_data:
                # Don't translate Chinese data values
                pass
        # lines that look like: respanno\gui\... (file paths in comments) are already English

    if modified:
        new_content = '\n'.join(lines)
        if not dry_run:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(new_content)
        print(f"\n{'[DRY RUN]' if dry_run else '[WRITTEN]'}  {os.path.relpath(fpath, ROOT)}  ({len(changed)} changes)")
        for lineno, old, new in changed:
            print(f"  L{lineno}: {old}")
            print(f"      -> {new}")
    else:
        print(f"  [SKIP] {os.path.relpath(fpath, ROOT)} (no Chinese comments found)")

    return modified

# ── main ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    dry_run = '--write' not in sys.argv
    mode = "DRY RUN" if dry_run else "WRITE MODE"
    print(f"=== Chinese Comment Translation Tool ({mode}) ===\n")
    total = total_changed = 0
    for rel in FILES_TO_PROCESS:
        fpath = os.path.join(ROOT, rel)
        if not os.path.isfile(fpath):
            print(f"  [MISSING] {rel}")
            continue
        modified = process_file(fpath, dry_run=dry_run)
        total += 1
        if modified:
            total_changed += 1

    print(f"\n=== Done: {total_changed}/{total} files would be modified ===")
    if dry_run:
        print("Run with --write to apply changes.")
