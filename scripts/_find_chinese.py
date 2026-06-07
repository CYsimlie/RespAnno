"""Find all Python files with Chinese characters in respanno/, tests/, scripts/."""
import os, re
CHINESE_RANGE = re.compile(r'[一-鿿　-〿＀-￯]')
PROJECT_DIR = r'd:\SoftwareX_win_test\New folder2\SoftwareX'

results = []
for root, dirs, files in os.walk(PROJECT_DIR):
    dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'legacy')]
    for f in files:
        if not f.endswith('.py'):
            continue
        fpath = os.path.join(root, f)
        try:
            with open(fpath, 'r', encoding='utf-8') as fh:
                for i, line in enumerate(fh, 1):
                    if CHINESE_RANGE.search(line):
                        results.append((os.path.relpath(fpath, PROJECT_DIR), i, line.rstrip()))
        except Exception:
            pass

print(f"Total lines with Chinese characters: {len(results)}")
for rpath, lineno, line in results:
    print(f"  {rpath}:{lineno}  {line.strip()[:120]}")
