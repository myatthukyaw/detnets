"""Lightweight smoke tests that run without the heavy model dependencies.

These guard the wiring in ``main.py`` and the config files - the parts most
prone to silent breakage (dispatch bugs, renamed config keys, bad YAML). They
deliberately avoid importing torch/ultralytics so they can run in plain CI.
"""
import ast
import glob
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

# main.py imports `tools.files`, which is only importable when the repo root is
# on sys.path. pytest does not guarantee that, so add it explicitly.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_main_module():
    """Load main.py as a module without executing model imports."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("detnets_main", REPO_ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_configs_are_valid_yaml():
    configs = glob.glob(str(CONFIG_DIR / "*.yml"))
    assert configs, "no config files found"
    for path in configs:
        with open(path) as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict), f"{path} did not parse to a mapping"


def test_model_choices_match_config_map():
    """Every --model choice must have a registered config that exists on disk."""
    main = _load_main_module()
    args_parser_src = (REPO_ROOT / "main.py").read_text()
    # Pull the choices list straight from the argparse call.
    tree = ast.parse(args_parser_src)
    choices = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "choices" and isinstance(kw.value, ast.List):
                    values = [e.value for e in kw.value.elts if isinstance(e, ast.Constant)]
                    if "yolov8" in values:
                        choices = values
    assert choices, "could not find --model choices in main.py"

    for model in choices:
        assert model in main.model_configs, f"{model} missing from model_configs"
        config_path = REPO_ROOT / main.model_configs[model]
        assert config_path.exists(), f"config for {model} not found: {config_path}"


def test_ultralytics_configs_expose_expected_sections():
    """yolov8 and rt-detr share the ultralytics wrappers, which read these keys."""
    for name in ("yolov8", "rt-detr"):
        with open(CONFIG_DIR / f"{name}.yml") as fh:
            cfg = yaml.safe_load(fh)
        for section in ("train_cfg", "test_cfg", "inference"):
            assert section in cfg, f"{name}.yml missing '{section}' section"


def test_python_sources_compile():
    sources = [REPO_ROOT / "main.py", REPO_ROOT / "tools" / "files.py"]
    sources += list((REPO_ROOT / "scripts").glob("*.py"))
    for src in sources:
        compile(src.read_text(), str(src), "exec")
