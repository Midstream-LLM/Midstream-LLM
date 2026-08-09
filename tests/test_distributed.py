from __future__ import annotations

import unittest
from unittest import mock

from jishui.distributed import (
    DistributedCapabilityError,
    LaunchEnvironment,
    build_runtime_plan,
    probe_accelerators,
)


class DistributedRuntimeTest(unittest.TestCase):
    def test_torchrun_environment_is_all_or_nothing(self):
        with self.assertRaisesRegex(
            DistributedCapabilityError, "incomplete torchrun environment"
        ):
            LaunchEnvironment.from_environ({"WORLD_SIZE": "8"})

        launch = LaunchEnvironment.from_environ(
            {"RANK": "5", "LOCAL_RANK": "1", "WORLD_SIZE": "8"}
        )
        self.assertEqual(launch, LaunchEnvironment(rank=5, local_rank=1, world_size=8))

    def test_cuda_plan_uses_local_rank_and_nccl(self):
        plan = build_runtime_plan(
            "cuda", LaunchEnvironment(rank=5, local_rank=1, world_size=8)
        )
        self.assertEqual(plan.device, "cuda:1")
        self.assertEqual(plan.process_group_backend, "nccl")
        self.assertTrue(plan.distributed)

    def test_ascend_plan_uses_hccl(self):
        plan = build_runtime_plan(
            "ascend-npu", LaunchEnvironment(rank=1, local_rank=1, world_size=2)
        )
        self.assertEqual(plan.device, "npu:1")
        self.assertEqual(plan.process_group_backend, "hccl")

    def test_mixed_cuda_and_npu_is_rejected(self):
        with self.assertRaisesRegex(
            DistributedCapabilityError, "cannot mix accelerator types"
        ):
            build_runtime_plan("cuda,ascend-npu")

    def test_ambiguous_npu_name_is_rejected(self):
        with self.assertRaisesRegex(DistributedCapabilityError, "ambiguous"):
            build_runtime_plan("npu")

    def test_apple_ane_is_not_exposed_as_ddp_device(self):
        with self.assertRaisesRegex(DistributedCapabilityError, "cannot be a PyTorch/DDP"):
            build_runtime_plan("apple-ane")

    def test_mps_cannot_form_a_multi_process_group(self):
        with self.assertRaisesRegex(DistributedCapabilityError, "single process"):
            build_runtime_plan(
                "mps", LaunchEnvironment(rank=0, local_rank=0, world_size=2)
            )

    def test_probe_survives_missing_torch(self):
        with mock.patch(
            "jishui.distributed._safe_import", return_value=(None, "not installed")
        ):
            probes = probe_accelerators(system_name="Darwin", machine="arm64")
        self.assertFalse(probes["cuda"].runtime_available)
        self.assertFalse(probes["ascend-npu"].runtime_available)
        self.assertTrue(probes["apple-ane"].hardware_available)
        self.assertFalse(probes["apple-ane"].ddp_supported)


if __name__ == "__main__":
    unittest.main()
