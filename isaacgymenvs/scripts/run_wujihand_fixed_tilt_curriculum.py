#!/usr/bin/env python3
"""Run the two-worker 42-direction WujiHandFixedTilt curriculum.

Scheduling, checkpoint staging, timeout, resume, and nearest-successful-parent
selection intentionally reuse the already-tested Shadowhand18 curriculum core.
Only the task name, deterministic run prefix, and default manifest differ.
"""

import sys
from pathlib import Path

try:
    from . import run_shadowhand18_tilt_curriculum as _core
except ImportError:
    # Supports direct execution from isaacgymenvs/scripts.
    import run_shadowhand18_tilt_curriculum as _core


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "curricula"
    / "wujihand_fixed_tilt_42.yaml"
)
_CORE_BUILD_COMMAND = _core.build_command


def build_wuji_command(python_executable, target, staged_checkpoint,
                       run_name, training):
    command = _CORE_BUILD_COMMAND(
        python_executable, target, staged_checkpoint, run_name, training
    )
    return [
        "task=WujiHandFixedTilt" if token == "task=Shadowhand18Tilted" else token
        for token in command
    ]


def make_wuji_run_name(target, attempt):
    return "wujitilt_{:02d}_{}_a{:02d}".format(
        target.manifest_index + 1, target.target_id, attempt
    )


def main(argv=None):
    # The core resolves these globals when building each job, so this wrapper
    # retains identical scheduling behavior without changing ShadowHand files.
    _core.DEFAULT_MANIFEST = DEFAULT_MANIFEST
    _core.build_command = build_wuji_command
    _core.make_run_name = make_wuji_run_name
    _core.__doc__ = __doc__
    return _core.main(argv)


if __name__ == "__main__":
    sys.exit(main())
