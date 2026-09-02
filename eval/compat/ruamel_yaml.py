"""Lets the unmodified T2I-CompBench evaluators run against a current ruamel.yaml.

The evaluators do `import ruamel_yaml as yaml` (falling back to `ruamel.yaml`) and then call the
module-level `yaml.load(..., Loader=yaml.Loader)` / `yaml.dump(...)` of ruamel.yaml < 0.17, which
newer releases removed. Putting this module on PYTHONPATH makes the first import succeed with a
YAML 1.2 parser, so the evaluator source stays byte-identical to upstream.
"""
import io

from ruamel.yaml import YAML
from ruamel.yaml import YAMLError  # noqa: F401


def _yaml():
    return YAML(typ="safe", pure=True)


Loader = object()
FullLoader = Loader
SafeLoader = Loader
Dumper = object()


def load(stream, Loader=None, **kwargs):
    return _yaml().load(stream)


def safe_load(stream, **kwargs):
    return _yaml().load(stream)


def dump(data, stream=None, Dumper=None, **kwargs):
    if stream is None:
        buf = io.StringIO()
        _yaml().dump(data, buf)
        return buf.getvalue()
    _yaml().dump(data, stream)


def safe_dump(data, stream=None, **kwargs):
    return dump(data, stream)
