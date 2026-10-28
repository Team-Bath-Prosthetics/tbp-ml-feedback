"""Import every module in the package, so broken imports fail CI even if no other test uses them."""

import importlib
import pkgutil
import pytest
import tactile

MODULES = [m.name for m in pkgutil.walk_packages(tactile.__path__, prefix="tactile.")]

@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)
