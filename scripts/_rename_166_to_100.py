"""Replace all 1.0.0.py references with 1.0.0.py across the project."""
import os, re

ROOT = r'd:\SoftwareX_win_test\New folder2\SoftwareX'
EXCLUDE_DIRS = {'.git', '__pycache__'}

extensions = {'.py', '.md', '.toml', '.cff', '.yml', '.yaml'}

changed = 0
for root, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in extensions:
            continue
        fpath = os.path.join(root, f)
        with open(fpath, 'r', encoding='utf-8') as fh:
            content = fh.read()
        if '1.0.0.py' not in content:
            continue
        new_content = content.replace('1.0.0.py', '1.0.0.py')
        with open(fpath, 'w', encoding='utf-8') as fh:
            fh.write(new_content)
        c = content.count('1.0.0.py')
        changed += c
        rel = os.path.relpath(fpath, ROOT)
        print(f'  {rel}: {c} replacement(s)')

print(f'\nTotal: {changed} replacements')
