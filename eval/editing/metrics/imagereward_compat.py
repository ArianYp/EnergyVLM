"""Import ImageReward's scorer without executing its package __init__.

The shared env has NumPy 2.x while its pandas/matplotlib are NumPy-1 builds. ImageReward's
__init__ pulls in ReFL -> datasets -> pandas and dies on that mismatch, even though the SCORER
never touches pandas. Rather than upgrading packages in a shared environment, a synthetic
`ImageReward` package module with the right __path__ is registered and the scorer submodules are
imported directly, so __init__.py never runs.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path


def load_imagereward(name: str = "ImageReward-v1.0", device: str = "cuda", download_root=None):
    import torch  # noqa: F401  (import torch before touching sys.modules for _dynamo's find_spec)
    if "ImageReward.utils" not in sys.modules:
        pkg_dir = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}" \
                                     / "site-packages" / "ImageReward"
        if not pkg_dir.is_dir():
            raise ImportError(f"ImageReward package directory not found at {pkg_dir}")
        pkg = types.ModuleType("ImageReward")
        pkg.__path__ = [str(pkg_dir)]
        spec = importlib.machinery.ModuleSpec("ImageReward", None, is_package=True)
        spec.submodule_search_locations = [str(pkg_dir)]
        pkg.__spec__ = spec
        sys.modules["ImageReward"] = pkg
        importlib.import_module("ImageReward.models")
        importlib.import_module("ImageReward.ImageReward")
        importlib.import_module("ImageReward.utils")
    utils = sys.modules["ImageReward.utils"]
    kw = {"device": device}
    if download_root is not None:
        kw["download_root"] = str(download_root)
    return utils.load(name, **kw)
