"""Show remaining Chinese lines in 1.0.0.py, categorized."""
import os, re
C = re.compile(r'[一-鿿　-〿＀-￯]')
fpath = r'd:\SoftwareX_win_test\New folder2\SoftwareX\1.0.0.py'
with open(fpath, encoding='utf-8') as f:
    lines = f.readlines()

categories = {'comment': [], 'docstring': [], 'inline_comment': [], 'data': []}
for i, line in enumerate(lines, 1):
    if not C.search(line):
        continue
    s = line.strip()
    if s.startswith('"""') or s.startswith("'''"):
        categories['docstring'].append((i, s[:120]))
    elif s.startswith('#'):
        categories['comment'].append((i, s[:120]))
    elif '#' in s and C.search(s.split('#')[1] if '#' in s else ''):
        categories['inline_comment'].append((i, s[:120]))
    else:
        categories['data'].append((i, s[:120]))

for cat, items in categories.items():
    if items:
        print(f"\n=== {cat} ({len(items)} lines) ===")
        for lineno, text in items[:15]:
            print(f"  L{lineno}: {text}")
        if len(items) > 15:
            print(f"  ... and {len(items) - 15} more")
