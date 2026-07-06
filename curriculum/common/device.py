"""Device detection + the startup banner every chapter artifact prints.

Scale-lab spec: the banner prints tier + expected wall-clock at every script
start. Chapters call banner(...) as their first act, right after arg parsing.
Pass the device the run will ACTUALLY use so the banner is honest — a CPU run
must not quote the mps tier's wall-clock:

    from curriculum.common.device import banner, detect_device
    device = detect_device()          # or args.device, chosen by a --device flag
    banner("ch1.1-bc", device=device)

torch is an *optional* import here: detect_device() falls back to "cpu" when
torch is absent, so a torch-free phase-0 chapter can still print the banner.
"""

from curriculum.common import wallclock

# Human-readable tier name shown in the banner, per device.
_TIER_NAME = {
    "cuda": "t4-or-better (GPU)",
    "mps": "mps",
    "cpu": "cpu-laptop",
}

# Tier key used to look up wallclock.csv rows, per device. CUDA maps to the
# free-tier floor measurement (t4) — a faster GPU only finishes sooner.
_WALLCLOCK_TIER = {
    "cuda": "t4",
    "mps": "mps",
    "cpu": "cpu-laptop",
}


def detect_device() -> str:
    """Best available torch device: "cuda" > "mps" > "cpu".

    torch-optional: a phase-0 chapter with no torch dependency still imports
    this module for the banner, so a missing torch means "cpu", not a crash.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def banner(chapter_id: str, device: str | None = None) -> None:
    """Print the run's device, tier, and measured wall-clock line.

    Every chapter artifact calls this at start. Pass `device` (the value the
    run will actually use, e.g. args.device) so the banner reports THAT device
    and looks up ITS tier's wall-clock — never the best-available device's.
    device=None keeps the legacy behavior: report detect_device()'s best pick.
    The wall-clock line comes from wallclock.csv (measured or "not yet
    measured") — never a guess.
    """
    if device is None:
        device = detect_device()
    tier = _WALLCLOCK_TIER[device]
    print(f"[{chapter_id}] device: {device} | tier: {_TIER_NAME[device]}")
    print(f"[{chapter_id}] {wallclock.render_line(chapter_id, tier)}")
