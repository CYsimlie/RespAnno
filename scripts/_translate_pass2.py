"""Pass 2: word-level re-translation for 1.6.6.py remaining Chinese.

The file already has partial translations (CN+EN mixed). This pass applies
individual Chinese → English word replacements aggressively to clean up
all remaining Chinese in comments and docstrings.
"""
import sys, re

C = re.compile(r'[一-鿿　-〿＀-￯]')
TARGET = '1.6.6.py'

# Load v2 PHRASES
exec(open('scripts/_translate_v2.py', encoding='utf-8').read().split("if __name__")[0])

# Also add the MORE_PHRASES from final
exec(open('scripts/_translate_final.py', encoding='utf-8').read().split("# ── Process")[0])

ALL = MORE_PHRASES + PHRASES

DATA_KEYWORDS = ['("', "('", 'FEATURE_NAMES', 'PHASE_LABELS', 'OTHER_EVENT',
                 'ABNORMAL_SOUND', 'annotation_builtin_labels',
                 'Wheeze":', 'Crackles":', 'Pleural Rub":', 'Rhonchi":',
                 'Stridor":', 'Speech":', 'Cough":', 'Expiration":',
                 'Inspiration":', '"wheeze"', 'feature_palette']

# ── Process ─────────────────────────────────────────────────────────────
dry_run = '--write' not in sys.argv
with open(TARGET, 'r', encoding='utf-8') as f:
    lines = f.readlines()

changed = 0
for i, line in enumerate(lines):
    stripped = line.strip()
    if not C.search(line):
        continue

    # Skip data lines (Chinese labels are UI content, not comments)
    if any(kw in stripped for kw in DATA_KEYWORDS if kw.startswith('("') or kw.startswith("('")):
        continue
    # Also skip lines that are purely data assignments with Chinese in dict keys
    if any(kw in stripped for kw in ['哮鸣音', '爆裂音', '摩擦音', '哼鸣音',
                                      '喘息音', '吸气', '呼气', '咳嗽', '语音']):
        continue

    new_line = line
    in_comment = False
    if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
        in_comment = True
    elif '#' in line:
        # Inline comment - only translate the comment part
        hash_pos = line.index('#')
        code_part = line[:hash_pos]
        comment = line[hash_pos:]
        new_comment = comment
        for cn, en in ALL:
            if cn in new_comment:
                new_comment = new_comment.replace(cn, en)
        new_line = code_part + new_comment
        if new_line != line:
            lines[i] = new_line
            changed += 1
        continue
    else:
        # Not a comment line, skip
        continue

    if in_comment:
        for cn, en in ALL:
            if cn in new_line:
                new_line = new_line.replace(cn, en)
        if new_line != line:
            lines[i] = new_line
            changed += 1

if not dry_run:
    with open(TARGET, 'w', encoding='utf-8') as f:
        f.writelines(lines)

action = '[DRY RUN]' if dry_run else '[WRITTEN]'
print(f"{action} pass2 on {TARGET}: {changed} changes")
if dry_run:
    print("Run with --write to apply.")
