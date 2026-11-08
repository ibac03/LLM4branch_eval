"""Write the evolved best program with its BO-tuned parameters baked into PARAMS.

openevolve saves best_program.py with the LLM's initial PARAMS; the tuned values live in
best_program_info.json -> metrics.params. The tester reads PARAMS from the file, so bake them in.

    python scripts/make_tuned_program.py <output_dir>/best  <out_program.py>
"""
import json
import re
import sys
from pathlib import Path

best = Path(sys.argv[1])
code = (best / "best_program.py").read_text()
info = json.loads((best / "best_program_info.json").read_text())
params = info["metrics"].get("params")
if params:
    code = re.sub(r"PARAMS\s*=\s*\[.*?\]", "PARAMS = " + repr([float(p) for p in params]), code, count=1, flags=re.S)
Path(sys.argv[2]).write_text(code)
print("tuned params:", params)
print("metrics:", {k: v for k, v in info["metrics"].items() if k != "params"})
