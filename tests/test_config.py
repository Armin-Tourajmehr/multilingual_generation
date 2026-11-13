from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ConfigError, load_config, validate_config


class ConfigurationTests(unittest.TestCase):
    def test_b200_preset_keeps_experiment_selection_and_uses_effective_batches(self):
        baseline = load_config(ROOT / "configs/config.yaml")
        cfg = load_config(ROOT / "configs/b200_config.yaml")
        self.assertEqual(cfg["languages"], baseline["languages"])
        self.assertEqual(cfg["data"], baseline["data"])
        self.assertEqual(cfg["generation"], baseline["generation"])
        self.assertTrue(cfg["lexical_analysis"]["deferred"])
        self.assertEqual(cfg["runtime"]["mixed_precision"], "bf16")
        for name, spec in cfg["models"].items():
            self.assertEqual(spec["enabled"], baseline["models"][name]["enabled"])
            self.assertEqual(spec["model_name"], baseline["models"][name]["model_name"])
            self.assertEqual(spec["torch_dtype"], "bfloat16")
            effective = {**cfg["runtime"], **spec.get("runtime_overrides", {})}
            for field in ("wikipedia_batch_size", "evaluation_batch_size", "generation_batch_size"):
                self.assertGreater(effective[field], 1)

    def test_invalid_runtime_tuning_values_fail_before_run(self):
        baseline = load_config(ROOT / "configs/config.yaml")
        for field, value in (("posterior_batch_size", 0), ("posterior_batch_size", True),
                             ("gc_collect_every_batches", -1), ("gc_collect_every_batches", 1.5),
                             ("log_performance", "yes")):
            with self.subTest(field=field, value=value):
                cfg = deepcopy(baseline)
                cfg["runtime"][field] = value
                with self.assertRaises(ConfigError):
                    validate_config(cfg)

    def test_invalid_model_overrides_and_attention_fail_before_loading(self):
        cfg = load_config(ROOT / "configs/config.yaml")
        cfg["models"]["qwen"]["runtime_overrides"]["generation_batch_size"] = 0
        with self.assertRaises(ConfigError):
            validate_config(cfg)
        cfg = load_config(ROOT / "configs/config.yaml")
        cfg["models"]["mgpt"]["attn_implementation"] = ""
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_optional_analysis_flags_must_be_boolean(self):
        baseline = load_config(ROOT / "configs/config.yaml")
        for section, field in (("lexical_analysis", "deferred"),
                               ("lexical_analysis", "save_pos_posterior_summary"),
                               ("outputs", "save_preceding_token_posteriors"),
                               ("representation", "equal_language_priors")):
            with self.subTest(field=field):
                cfg = deepcopy(baseline)
                cfg[section][field] = "true"
                with self.assertRaises(ConfigError):
                    validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
