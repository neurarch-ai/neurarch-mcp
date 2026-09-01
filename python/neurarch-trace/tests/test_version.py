"""pyproject.toml and __init__.__version__ are two copies of one number. The
publish workflow refuses a tag that disagrees with pyproject; this refuses a
package that disagrees with itself, so the CLI never reports a stale version."""
import re
from pathlib import Path

import neurarch_trace


def test_version_matches_pyproject():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    declared = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)
    assert neurarch_trace.__version__ == declared
