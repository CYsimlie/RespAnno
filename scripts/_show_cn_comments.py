import re
C = re.compile(r'[一-鿿]')
with open('1.6.6.py', encoding='utf-8') as f:
    lines = f.readlines()
for i, line in enumerate(lines, 1):
    s = line.strip()
    if not C.search(s):
        continue
    if s.startswith('#'):
        print(f'L{i}: {s[:150]}')
