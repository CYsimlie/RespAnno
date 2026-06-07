"""Spot-check 9 sample translations from _translate_v2."""
import sys
sys.path.insert(0, '.')
# Load translate function from v2 script
with open('scripts/_translate_v2.py', encoding='utf-8') as f:
    code = f.read()
code = code.split("if __name__")[0]
exec(code)

tests = [
    '# 构造 Qt plugins directory的绝对path（兼容不同 PyQt5 安装layout）',
    '# —— Machine-learning hard-negative sample manager (training only; not exported; unrelated to annotation display)——',
    '    self.LANE_H = 0.35          # Single-lane height',
    '"""清空所有标注：移除可视化对象、清空数据结构、复位视图范围。"""',
    '"""comboboxlabel变化时updatecurrent ML 操作目标。"""',
    '"""验证完整 ML 管线（合成音频→特征→训练→推理）运行无异常。"""',
    '"""Verify butter filter 的确定性：相同input产生逐位相同的output。"""',
    '# 直接按照代码中的预设label顺序生成图例，不依赖current是否已有annotation。',
    '# ===== File menu =====',
]
for t in tests:
    print('IN :', t)
    print('OUT:', translate_chinese_in_text(t))
    print()
