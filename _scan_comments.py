"""Scan all Python files for comments that look like AI/dev process notes.
Outputs: file:line | type | content
"""
import os, re, glob

ROOT = r"d:\SoftwareX_win_test\New folder2\SoftwareX"
TARGETS = (
    glob.glob(os.path.join(ROOT, "1.0.0.py"))
    + glob.glob(os.path.join(ROOT, "respanno", "**", "*.py"), recursive=True)
    + glob.glob(os.path.join(ROOT, "tests", "**", "*.py"), recursive=True)
    + glob.glob(os.path.join(ROOT, "scripts", "**", "*.py"), recursive=True)
)
TARGETS = [t for t in TARGETS if "legacy" not in t and "__pycache__" not in t]

# Patterns to flag (comment text, type label)
PATTERNS = [
    (re.compile(r'[一-鿿]'), 'zh'),          # Chinese chars
    (re.compile(r'#\s*TODO'), 'TODO'),
    (re.compile(r'#\s*FIXME'), 'FIXME'),
    (re.compile(r'#\s*NOTE'), 'NOTE'),
    (re.compile(r'#\s*HACK'), 'HACK'),
    (re.compile(r'#\s*BUG'), 'BUG'),
    (re.compile(r'#\s*WORKAROUND'), 'WORKAROUND'),
    (re.compile(r'#\s*noqa'), 'noqa'),
    (re.compile(r'#\s*={3,}'), 'sep-eq'),
    (re.compile(r'#\s*\-{3,}'), 'sep-dash'),
    (re.compile(r'#\s*═{3,}'), 'sep-dbline'),   # ═══
    (re.compile(r'#\s*~\s'), 'sep-tilde'),
    (re.compile(r'#\s*\([A-Z]\s'), 'marker-enum'),    # # (A) or # (1)
]

# Also catch: verbose inline explanations (>15 words)
LONG_INLINE = re.compile(r'# .+')

results = []
for filepath in sorted(TARGETS):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_docstring = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip blank lines and shebangs
        if not stripped or stripped.startswith('#!'):
            continue

        # Skip if inside a triple-quoted docstring
        if stripped.count('"""') % 2 == 1 or stripped.count("'''") % 2 == 1:
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        # Also skip lines that are inside docstrings ("""...""")
        if stripped.startswith('"""') or stripped.startswith("'''"):
            continue

        # Only interested in comment lines
        if not stripped.startswith('#'):
            continue

        # Remove leading #
        comment_text = re.sub(r'^#\s*', '', stripped)

        matched_types = []
        for pat, tlabel in PATTERNS:
            if pat.search(comment_text):
                matched_types.append(tlabel)

        if not matched_types:
            # Check if it's a long verbose comment (>15 words = likely dev note)
            word_count = len(comment_text.split())
            if word_count >= 15 and not comment_text.startswith('"""'):
                matched_types.append('verbose')
            else:
                continue

        relpath = os.path.relpath(filepath, ROOT).replace('\\', '/')
        results.append((relpath, i, comment_text[:120], '|'.join(matched_types)))

# Print
for relpath, lineno, comment, tag in results:
    print(f"{relpath}:{lineno} | {tag:25s} | {comment}")

print(f"\n{'='*80}")
print(f"Total: {len(results)} flagged comments across {len(set(r[0] for r in results))} files")
