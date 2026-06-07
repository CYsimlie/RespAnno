"""Count Chinese character lines per .py file."""
import os, re
C = re.compile(r'[一-鿿　-〿＀-￯]')
root = r'd:\SoftwareX_win_test\New folder2\SoftwareX'
counts = {}
for rp, ds, fs in os.walk(root):
    ds[:] = [d for d in ds if d not in ('__pycache__','.git','legacy')]
    for f in fs:
        if not f.endswith('.py'): continue
        fp = os.path.join(rp, f)
        try:
            with open(fp, encoding='utf-8') as fh:
                n = sum(1 for l in fh if C.search(l))
        except: continue
        if n > 0:
            counts[os.path.relpath(fp, root)] = n
for k in sorted(counts, key=lambda x: counts[x], reverse=True):
    print(f'{counts[k]:4d}  {k}')
