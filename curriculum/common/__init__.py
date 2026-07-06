"""curriculum.common — the ONE sanctioned shared import for chapter artifacts.

Chapters run as standalone files from the repo root
(`python curriculum/phaseX/chY/artifact.py`), so they import these utilities
by first putting the repo root on sys.path:

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from curriculum.common.device import banner
    from curriculum.common.seeding import set_seed

Keep this package tiny and boring: seeding, device/tier banner, wall-clock
lookup, ONNX export, and torch-vs-ONNX parity. Nothing else belongs here.
"""
