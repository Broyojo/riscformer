#!/bin/bash
# Rebuild the architecture figure: TikZ -> PDF -> SVG (explicit white ground
# so the figure stays readable on GitHub dark mode).
set -e
cd "$(dirname "$0")"
tectonic arch.tex
pdftocairo -svg arch.pdf architecture.svg
python3 - <<'PY'
s = open('architecture.svg').read()
mark = '<rect width="100%" height="100%" fill="#ffffff"/>'
if mark not in s:
    i = s.index('>', s.index('<svg')) + 1
    s = s[:i] + '\n' + mark + s[i:]
open('architecture.svg', 'w').write(s)
PY
echo "docs/architecture.svg rebuilt"
