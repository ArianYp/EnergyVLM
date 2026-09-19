"""Import-compatibility shim for the vendored t2v_metrics (third_party/t2v_metrics).

t2v_metrics/models/vqascore_models/__init__.py unconditionally imports the
API-based scorers (GPT4V, Gemini). The Gemini one does `from google import genai`
at module level, and the installed google-genai needs a newer typing_extensions
than the shared nichejepa env provides (PEP-728 `extra_items`), so merely

    import t2v_metrics

raises: TypeError: _TypedDictMeta.__new__() got an unexpected keyword argument
'extra_items'.

We never use the API scorers (scoring is local: clip-flant5-xxl + open_clip), so
we stub `google.genai` out BEFORE t2v_metrics is imported. Upgrading
typing_extensions in the shared env was rejected as too risky mid-campaign —
torch/transformers pin it, and training jobs share this env.

This shim lives in OUR tree (not inside t2v_metrics) so it survives a re-clone
of the vendored repo.

Usage — must precede the t2v_metrics import:
    import _t2v_compat  # noqa: F401
    import t2v_metrics
"""
from __future__ import annotations

import sys
import types

class _Unavailable:
    """Placeholder symbol; raises only if a stubbed backend is actually used."""

    _origin = "a stubbed-out backend"

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self._origin} is stubbed out by common/t2v_compat.py and is "
            "unavailable in this environment. Use a local image scorer "
            "(clip-flant5-xxl / open_clip)."
        )


def stub_module(path: str, attrs: tuple[str, ...] = ()) -> bool:
    """Install a dummy module at dotted `path` with placeholder `attrs`.

    Also links it as an attribute of its parent package so both
    `import a.b` and `from a import b` resolve. No-op if already imported.
    """
    if path in sys.modules:
        return False
    mod = types.ModuleType(path)
    for name in attrs:
        setattr(mod, name, _Unavailable)
    sys.modules[path] = mod
    parent, _, leaf = path.rpartition(".")
    if parent:
        try:
            pmod = sys.modules.get(parent) or __import__(parent, fromlist=[leaf])
            setattr(pmod, leaf, mod)
        except Exception:
            pass
    return True


# ── Backends t2v_metrics imports unconditionally but we never use ───────────
# google.genai  : Gemini API scorer. Needs typing_extensions >= PEP-728
#                 (`extra_items`); the shared env pins an older one.
# torchcodec    : video decoding for the PerceptionLM video scorer. Its
#                 compiled libtorchcodec is ABI-incompatible with this torch
#                 build (undefined symbol / libavutil.so.* missing).
# We score still images only, so both are safely stubbed.
stub_module("google.genai", ("Client",))
stub_module(
    "google.genai.types",
    ("GenerateContentConfig", "HarmBlockThreshold", "HarmCategory", "Part", "SafetySetting"),
)
stub_module("torchcodec")
stub_module("torchcodec.decoders", ("VideoDecoder", "AudioDecoder"))
stub_module("torchcodec.encoders")
stub_module("torchcodec.samplers")
stub_module("torchcodec.transforms")
