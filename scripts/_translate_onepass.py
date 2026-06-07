"""One-pass complete translation of ALL Chinese in comments/docstrings in 1.6.6.py.

Combines ALL phrase lists from v2, final, and extra missing words.
Only touches # comments, docstrings, and inline comments. Never data lines.
"""
import sys, re

C = re.compile(r'[一-鿿]')

# Load v2 PHRASES
with open('scripts/_translate_v2.py', encoding='utf-8') as f:
    code = f.read()
exec(code.split("if __name__")[0])

# Load MORE_PHRASES from final
with open('scripts/_translate_final.py', encoding='utf-8') as f:
    fcode = f.read()
exec(fcode.split("# ── Process")[0])

all_pairs = list(MORE_PHRASES) + list(PHRASES)

# ── Extra missing words ─────────────────────────────────────────────────
extra = [
    ("构造","Construct "),("绝对","absolute "),("兼容","compatible with"),
    ("不同","different "),("安装","installation "),("若 ","If "),
    ("不存在","does not exist"),("回退至","fall back to"),("无 ","without "),
    ("层级","level"),("插件","plugin"),("环境变量","environment variable"),
    ("运行时","runtime"),("能找到","can locate"),("平台","platform"),
    ("当存在","When "),("快捷键","shortcut"),("歧义","ambiguity"),
    ("也连接","also connect"),("以 ","to "),("生效","take effect"),
    ("应用级","Application-level"),("过滤器","filter"),("复杂","complex"),
    ("逻辑","logic"),("等","etc."),("图","plot"),("鼠标","mouse"),
    ("移动","movement"),("通过","via"),("矩形","rectangle"),
    ("判断","determine"),("是否","whether"),("在","in"),
    ("绘图","plotting"),("固定为","fixed at"),("行","lane"),
    ("高度","height"),("仅允许","Allow only"),("拖动","drag"),
    ("坐标线","axis line"),("刻度","tick"),
    ("均不","all"),("网格","grid"),("同步","sync"),
    ("平移","pan"),("多余","extra"),("边距","margin"),
    ("避免","avoid"),("后仍留白","leaving whitespace after"),
    ("禁止","Prevent"),("各","each"),
    ("被完全折叠","from being collapsed entirely"),
    ("使用","Use"),("内置","built-in"),
    ("英文名","English names"),("作为","as"),
    ("若已有值且在列表中则沿用","keep existing value if in list"),
    ("否则取首个","otherwise use first"),
    ("按照","according to"),("预设","preset"),
    ("顺序","order"),("生成","generate"),
    ("图例","legend"),("依赖","depending on"),
    ("是否已有","whether there are existing"),
    ("中出现的","that appear in"),("出现过","appeared"),
    ("停止","stop"),("播放","playback"),
    ("避免旧","prevent old"),("仍占用","from still occupying"),
    ("声卡","sound card"),("为空","is empty"),
    ("弹出","show"),("框","dialog"),
    ("建立","Build"),("同","same"),
    ("中的","in"),("关闭","off"),
    ("保留","preserve"),("原始","original"),
    ("先清空旧缓存和旧","clear old caches and old"),
    ("误用","misuse"),
    ("新音频默认重置自动色阶","Reset auto-levels for new audio"),
    ("避免上一条文件的色阶影响当前文件","prevent previous file levels from affecting current"),
    ("先清旧","Clear old"),("再重绘新图","then redraw new plots"),
    ("防止旧","prevent residual old"),("残留","residue"),
    ("只做必要","Show only essential"),("抽稀","decimated"),
    ("延后到用户切换页面时","deferred until user switches pages"),
    ("很轻","is lightweight"),("仍保留","so stays"),
    ("里","inside"),("坐标","Coordinate"),("页面","page"),
    ("保存完整谱值用于","Save full spectrum values for"),
    ("直方图","histogram"),
    ("用谱图可以","the display spectrogram may be"),
    ("后的图像仍映射到","image still maps to"),
    ("完整","full"),("时长","duration"),
    ("保证","ensuring"),("一致","consistent"),
    ("变化会影响","changes affect"),
    ("但不在","but not during"),
    ("刷新时立即","immediately on refresh"),
    ("移除可视化对象","Remove visual objects"),
    ("及其上方","and their"),("文字","text"),
    ("从","from"),("移除自身及","Remove itself and"),
    ("图上","plot"),("高亮","highlight"),
    ("兼容旧","Backward compatible with old"),
    ("中保存","stored in"),("和","and"),
    ("拖拽","drag"),("过程中的临时","in-progress temporary"),
    ("数据结构","data structures"),("复位","Reset"),
    ("视图","view"),("快退","Seek backward"),
    ("快进","seek forward"),("供","called by"),
    ("调用","key"),("传进去","pass in"),
    ("上次","Last"),("使用的","used"),
    ("读取时","load-time"),
    ("不立即重读","Do not immediately reload"),
    ("避免影响现有","avoid disrupting current"),
    ("下次","next"),("取回","Retrieve"),
    ("先","first"),("再重绘","then redraw"),
    ("使用旧","using old"),
    ("改变后只刷新","change only refresh"),
    ("并标记","and mark"),("失效","invalid"),
    ("不立即","not immediately"),
    ("若只改了","If only changed"),
    ("而未重算","without recomputing"),
    ("也可以直接按当前","can also directly use current"),
    ("重着色","recolor"),
    ("过滤","Filter out"),("项","entries"),
    ("排除","exclude"),("已归档","archived"),
    ("隐藏","hidden"),
    ("沿用上次导出位置","reuses last export location"),
    ("跟随当前","follows current"),("例如","Example"),
    ("当前音频","current audio"),("文件名","filename"),
    ("打印导出","print exported"),
    ("分布统计","distribution statistics"),
    ("可在控制台查看","viewable in console"),
    ("记住","Remember"),("对象","object"),
    ("按被删除","using deleted"),
    ("而非当前","not current"),
    ("代码中还有","still present in code"),
    ("若传入的是","If input is"),
    ("尝试反查对应的","attempt reverse-lookup"),
    ("统一走","route through unified"),
    ("以便","to"),("复用","Reuse"),
    ("若已经有同","If a span with same"),
    ("理论上不该发生","should not happen"),
    ("直接返回","return directly"),
    ("避免重复","prevent duplication"),
    ("创建并渲染一个","Create and render a"),
    ("共用","Shared"),("消除重复","eliminating duplicated"),
    ("与笔刷","and pen"),("非","Non"),
    ("使用红色文字","use red text"),
    ("仅清空可视化对象","Clear only visual objects"),
    ("不清空","do not clear"),("等数据","data"),
    ("用于视图与数据不同步时的重建兜底","used as rebuild fallback when view and data desync"),
    ("重置轨道缓存","Reset lane cache"),
    ("不影响","does not affect"),
    ("重新渲染所有","rebuild all"),
    ("不改动","does not modify"),
    ("显示范围","display range"),
    ("渲染后不可见","invisible after rendering"),
    ("不显示也不导出","neither displayed nor exported"),
    ("合并","merge"),("前的旧","old before"),
    ("异常中断主流程","exceptions from interrupting main flow"),
    ("将指定","Change specified"),("修改为","to"),
    ("并同步","sync"),
    ("撤销最后一次编辑操作","Undo last edit operation"),
    ("编辑几何位置","geometry edit"),
    ("撤销一次删除","Undo a deletion"),("恢复","restore"),
    ("条目","entry"),("重建可视化","rebuild visualization"),
    ("移除负样本","remove negative sample"),
    ("兜底","Fallback"),
    ("恢复失败则重建整个视图","if restore fails rebuild entire view"),
    ("撤销一次几何编辑","Undo a geometry edit"),
    ("回滚位置","roll back position"),
    ("样式和高亮","style and highlight"),
    ("丢失则重建整个视图","lost then rebuild entire view"),
    ("回滚失败则重建视图","if rollback fails rebuild view"),
    ("变更","change"),("恢复旧","restore old"),
    ("重绘视觉样式","redraw visual style"),
    ("将机器标注标记为","Mark machine annotation as"),
    ("已认可","accepted"),("编辑模式","edit mode"),
    ("进入","enter"),
    ("记录当前进入编辑态的","Record currently entering edit mode"),
    ("若已有其他","if another"),
    ("在编辑则先提交退出","in edit commit and exit first"),
    ("退出编辑模式并清空编辑状态","Exit edit mode and clear editing state"),
    ("退出编辑后清理状态栏提示","Clean status bar notice after exiting edit mode"),
    ("编辑态","Edit mode"),
    ("在状态栏显示","display in status bar"),
    ("选中","selected"),("起止时间","start and end times"),
    ("返回稳定的颜色","Return stable color"),
    ("文本","text"),("固定颜色","fixed colors"),
    ("自动配色","auto-assigned colors"),
    ("已有映射直接返回","Return directly if already mapped"),
    ("英文名","English name"),
    ("其他任意文本","Any other text"),
    ("从调色板顺序取色","pick color sequentially from palette"),
    ("若未提供","If not provided"),
    ("正常交互","normal interactive"),
    ("则弹出带预设类型的","show dialog with preset types"),
    ("渲染","Render"),("可视化","visualization"),
    ("注册到","Register into"),("同名","matching"),
    ("可开关","toggleable"),("懒加载","lazy loading"),
    ("懒加载短时特征","lazy-load short-time features"),
    ("只要有时间交集即视为重叠","Any temporal intersection is considered overlap"),
    ("动态选择空闲轨道","Dynamically select free lane"),
    ("优先使用编号最小的空轨","prefer lane with smallest index"),
    ("收集现有每一行的","Collect intervals of each existing"),
    ("找第一个不重叠的","Find first non-overlapping"),
    ("三行都冲突则放最后一行","If all three lanes conflict place in last lane"),
    ("坐标","coordinates"),
    ("移出绘图区域则还原标题","restore title when leaving plot area"),
    ("仅在绘图区内","Only within plotting area"),
    ("不包含轴刻度","excluding axis ticks"),
    ("标题右侧","Right side of title"),("可调","adjustable"),
    ("计算并绘制短时特征曲线","Compute and plot short-time feature curves"),
    ("归一化叠加显示","normalized overlaid display"),
    ("该函数在主线程计算","function computes on main thread"),
    ("仅在切换到特征页时","only when switching to features page"),
    ("不主动调用","not called proactively"),
    ("画最多","Plot at most"),
    ("时间轴范围与","Time axis range aligned with"),
    ("切换配色方案","Switch color scheme"),
    ("不重新计算","do not recompute"),
    ("仅对显示用谱图","only recolor display spectrogram"),
    ("与主图方向一致","Match main plot orientation"),
    ("每条特征曲线","each feature curve"),
    ("不重复","unique"),("先搭","Lay foundation for"),
    ("状态机","state machine"),("基础","foundation"),
    ("后续会逐步引入更多","more will be introduced gradually"),
    ("参与","participates in"),("统计","statistic"),
    ("可进入下一轮训练的正样本","positive sample eligible for next training round"),
    ("三元组","3-tuple"),("四元组","4-tuple"),
    ("其他长度直接跳过","skip other lengths"),
    ("直接视为人工标记","treated as manual label"),
    ("用自带的","uses its own"),
    ("包括","Includes"),
    ("认可","accept"),
    ("键删除选中的","key removes selected"),
    ("键盘","keyboard"),
    ("过滤器","filter"),
    ("的",""),  # possessive particle - remove
]
all_pairs = extra + all_pairs
all_pairs.sort(key=lambda x: -len(x[0]))

DATA_KW = ['哮鸣音','爆裂音','摩擦音','哼鸣音','喘息音','吸气','呼气','咳嗽','语音',
           'feature_palette','annotation_builtin_labels','annotation_color_builtin',
           'Wheeze":','Crackles":','Pleural Rub":','Rhonchi":','Stridor":',
           'Speech":','Cough":','Expiration":','Inspiration":',
           '("', "('"]

# ── Process ─────────────────────────────────────────────────────────────
with open('1.6.6.py', encoding='utf-8') as f:
    lines = f.readlines()

changed = 0
for i, line in enumerate(lines):
    s = line.strip()
    if not C.search(line):
        continue
    if any(k in s for k in DATA_KW):
        continue

    is_comment = s.startswith('#')
    is_docstring = s.startswith('"""') or s.startswith("'''")
    has_inline = '#' in line and not is_comment

    if not (is_comment or is_docstring or has_inline):
        continue

    new = line
    for cn, en in all_pairs:
        if cn in new:
            new = new.replace(cn, en)
    if new != line:
        lines[i] = new
        changed += 1

with open('1.6.6.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print(f'{changed} comment/docstring lines translated')
