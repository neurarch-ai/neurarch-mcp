"""Everything about this package is in pyproject.toml except one file.

`neurarch_autopatch.pth` has to land at the *root* of site-packages, because
that is the only place Python looks for `.pth` files, and that is what lets
`NEURARCH_TRACE=1 python train.py` work with no import and no edit to the
script. `data_files` cannot put it there: pip installs those relative to
`sys.prefix`, which in a virtualenv is a directory Python never scans.

So the file is copied into `build_lib` during `build_py`, from where the wheel
builder packs it as top-level content, and it is copied again after `install`
for the rare path that does not go through a wheel. Editable installs (PEP 660)
build no such tree; `scripts/dev_copy_pth.py` handles those.

If this file disappears, nothing fails loudly: `pip install neurarch-trace`
still works, `neurarch-trace` on the command line still works, and only the
autopatch silently stops existing. `tests/test_autopatch.py::test_pth_is_packaged`
is what notices.
"""
import os

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.install import install as _install

PTH_FILE = "neurarch_autopatch.pth"
HERE = os.path.dirname(os.path.abspath(__file__))


class build_py(_build_py):
    def run(self):
        super().run()
        source = os.path.join(HERE, PTH_FILE)
        if os.path.isfile(source):
            self.mkpath(self.build_lib)
            self.copy_file(source, os.path.join(self.build_lib, PTH_FILE))

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        return outputs + [os.path.join(self.build_lib, PTH_FILE)]


class install(_install):
    def run(self):
        super().run()
        source = os.path.join(HERE, PTH_FILE)
        target = os.path.join(self.install_lib, PTH_FILE)
        if os.path.isfile(source) and not os.path.exists(target):
            self.mkpath(self.install_lib)
            self.copy_file(source, target)


setup(cmdclass={"build_py": build_py, "install": install})
