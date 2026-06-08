"""Ultra-safe translator - only touches lines starting with '#', never data."""
import sys, os, re

ROOT = r'd:\SoftwareX_win_test\New folder2\SoftwareX'
CHINESE_RE = re.compile(r'[一-鿿　-〿＀-￯]')

# Same phrase list as _translate_v2.py — load from there
exec(open(os.path.join(ROOT, 'scripts/_translate_v2.py'), encoding='utf-8')
     .read().split("if __name__")[0])

FILES = ['1.0.0.py', 'respanno/dsp/features.py', 'tests/test_features_basic.py']

dry_run = '--write' not in sys.argv
total = 0
for rel in FILES:
    fpath = os.path.join(ROOT, rel)
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    changed = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        # ONLY translate lines that start with # (comment lines)
        if stripped.startswith('#') and CHINESE_RE.search(stripped):
            indent = line[:len(line) - len(line.lstrip())]
            comment = stripped[1:].lstrip()  # text after '# '
            translated = translate_chinese_in_text(comment)
            new_line = indent + '# ' + translated
            if new_line != line:
                lines[i] = new_line
                changed += 1
    if changed > 0 and not dry_run:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    action = '[DRY RUN]' if dry_run else '[WRITTEN]'
    print(f"  {action} {rel} ({changed} changes)")
    total += changed

print(f"\n=== {total} changes total ===")
if dry_run:
    print("Run with --write to apply.")
