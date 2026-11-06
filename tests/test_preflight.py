from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from scripts.run_experiment import cuda_preflight


class PreflightTests(unittest.TestCase):
    def setUp(self):
        patches = {
            "scripts.run_experiment._resolve_device": {"return_value": torch.device("cuda")},
            "torch.cuda.get_arch_list": {"return_value": ["sm_100"]},
            "torch.cuda.device_count": {"return_value": 1},
            "torch.cuda.get_device_properties": {"return_value": SimpleNamespace(name="test B200", total_memory=180 * 1024**3)},
            "torch.cuda.get_device_capability": {"return_value": (10, 0)},
            "torch.cuda.device": {"side_effect": lambda device: nullcontext()},
            "torch.cuda.synchronize": {},
        }
        for target, kwargs in patches.items():
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.cfg = {"runtime": {"mixed_precision": "bf16"}, "models": {}}

    def test_requested_bf16_is_checked_before_kernel_allocation(self):
        with patch("torch.cuda.is_bf16_supported", return_value=False), patch("torch.ones") as allocate:
            with self.assertRaisesRegex(RuntimeError, "BF16 was requested"):
                cuda_preflight(self.cfg)
            allocate.assert_not_called()

    def test_unsupported_kernel_error_has_actionable_context(self):
        with patch("torch.cuda.is_bf16_supported", return_value=True), \
             patch("torch.ones", side_effect=RuntimeError("no kernel image")):
            with self.assertRaisesRegex(RuntimeError, "Blackwell support") as raised:
                cuda_preflight(self.cfg)
            self.assertIn("no kernel image", str(raised.exception.__cause__))

    def test_small_kernel_is_executed_and_synchronized(self):
        allocate_cpu = torch.ones
        with patch("torch.cuda.is_bf16_supported", return_value=True), \
             patch("torch.ones", side_effect=lambda shape, **kwargs: allocate_cpu(shape, dtype=kwargs["dtype"])) as allocate, \
             patch("torch.cuda.synchronize") as synchronize:
            cuda_preflight(self.cfg)
            self.assertEqual(allocate.call_args.kwargs["dtype"], torch.bfloat16)
            synchronize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
