import pathlib
import sys

RESEARCH_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))
