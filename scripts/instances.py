"""Make one Independent Set instance. Families: easy / medium / hard (Ecole BA graphs, official sizes),
rr (random regular graph) and ws (Watts-Strogatz graph), the last two from Hardik's generators."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "LLM4Branch", "examples", "brancher"))
SG = os.path.join(ROOT, "within-family-variation", "examples", "brancher", "experiments", "structural_generalization")


def make_instance(family, seed):
    if family in ("easy", "medium", "hard"):
        from utils.utils import create_instance
        return create_instance("indset", seed, family)
    if SG not in sys.path:
        sys.path.append(SG)
    if family == "rr":
        import category_rr_indset
        return category_rr_indset.generate_instance(seed)
    if family == "ws":
        import category_ws_indset
        return category_ws_indset.generate_instance(seed)
    raise ValueError(family)
