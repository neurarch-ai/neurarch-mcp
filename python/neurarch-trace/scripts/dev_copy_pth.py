"""Put `neurarch_autopatch.pth` into site-packages for an editable install.

    pip install -e .
    python scripts/dev_copy_pth.py

A PEP 660 editable install builds no `build_lib` tree, so the hook in setup.py
never runs and the autopatch is simply absent: `NEURARCH_TRACE=1 python train.py`
does nothing at all, with no error to explain why. This script is the two lines
that fix that, and `--check` reports the state without changing anything.
"""
import argparse
import os
import shutil
import site
import sys

PTH_FILE = "neurarch_autopatch.pth"
SOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), PTH_FILE)


def site_dir() -> str:
    """The site-packages directory of the interpreter running this script."""
    candidates = []
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:  # a virtualenv old enough to lack it
        pass
    user = site.getusersitepackages() if site.ENABLE_USER_SITE else None
    if user:
        candidates.append(user)
    for path in candidates:
        if os.path.isdir(path) and os.access(path, os.W_OK):
            return path
    if candidates:
        return candidates[0]
    raise RuntimeError("no site-packages directory found for %s" % sys.executable)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", help="report whether it is installed, change nothing")
    p.add_argument("--remove", action="store_true", help="delete it from site-packages")
    args = p.parse_args()

    target = os.path.join(site_dir(), PTH_FILE)

    if args.check:
        print("%s: %s" % (target, "present" if os.path.exists(target) else "absent"))
        return 0 if os.path.exists(target) else 1
    if args.remove:
        if os.path.exists(target):
            os.remove(target)
            print("removed %s" % target)
        else:
            print("nothing to remove at %s" % target)
        return 0
    if not os.path.isfile(SOURCE):
        print("no %s next to setup.py; run this from a source checkout" % PTH_FILE, file=sys.stderr)
        return 1

    shutil.copyfile(SOURCE, target)
    print("copied %s -> %s" % (PTH_FILE, target))
    print("NEURARCH_TRACE=1 python your_script.py will now trace the first forward pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
