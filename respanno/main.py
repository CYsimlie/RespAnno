"""Entry point for the Respiratory Sound Annotation Tool.

Currently wraps the legacy single-file implementation under `legacy/`.
As modules are extracted from the legacy file, this launcher will
import from the respanno subpackages instead.
"""

import sys
import os


def main() -> None:
    """Launch the annotation tool from the legacy implementation."""
    # Locate the legacy main program relative to this file.
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    _legacy = os.path.join(_root, "legacy", "1.0.0.py")

    if not os.path.exists(_legacy):
        sys.exit(
            f"ERROR: legacy entry point not found at {_legacy}\n"
            "Make sure the legacy/ directory contains 1.0.0.py."
        )

    # Inject the legacy file's directory so its internal imports work.
    sys.path.insert(0, os.path.dirname(_legacy))

    # exec the legacy file as __main__ so the if __name__ == '__main__'
    # block at the bottom fires.
    _globals = {
        "__name__": "__main__",
        "__file__": _legacy,
    }
    with open(_legacy, "rb") as fh:
        code = compile(fh.read(), _legacy, "exec")
    exec(code, _globals)


if __name__ == "__main__":
    main()
