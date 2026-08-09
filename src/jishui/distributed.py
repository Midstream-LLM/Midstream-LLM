from __future__ import annotations

import importlib
import os
import platform
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


class DistributedCapabilityError(RuntimeError):
    """Raised when a requested accelerator topology cannot be supported."""


@dataclass(frozen=True)
class AcceleratorProbe:
    name: str
    framework: str
    runtime_available: bool
    hardware_available: bool
    device_count: int
    ddp_supported: bool
    process_group_backend: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LaunchEnvironment:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @classmethod
    def from_environ(
        cls, environ: Mapping[str, str] | None = None
    ) -> "LaunchEnvironment":
        values = os.environ if environ is None else environ
        keys = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        present = [key in values for key in keys]
        if any(present) and not all(present):
            missing = ", ".join(key for key, exists in zip(keys, present) if not exists)
            raise DistributedCapabilityError(
                "incomplete torchrun environment; missing " + missing
            )
        if not any(present):
            return cls()
        try:
            result = cls(
                rank=int(values["RANK"]),
                local_rank=int(values["LOCAL_RANK"]),
                world_size=int(values["WORLD_SIZE"]),
            )
        except ValueError as exc:
            raise DistributedCapabilityError(
                "RANK, LOCAL_RANK, and WORLD_SIZE must be integers"
            ) from exc
        if result.world_size < 1:
            raise DistributedCapabilityError("WORLD_SIZE must be positive")
        if not 0 <= result.rank < result.world_size:
            raise DistributedCapabilityError(
                f"RANK={result.rank} must be in [0, WORLD_SIZE={result.world_size})"
            )
        if result.local_rank < 0:
            raise DistributedCapabilityError("LOCAL_RANK must be non-negative")
        return result


@dataclass(frozen=True)
class RuntimePlan:
    accelerator: str
    device: str
    process_group_backend: str | None
    rank: int
    local_rank: int
    world_size: int

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["distributed"] = self.distributed
        return value


@dataclass
class RuntimeContext:
    """Initialized torch runtime. The torch module stays lazy until this point."""

    plan: RuntimePlan
    torch: Any
    torch_device: Any
    created_process_group: bool

    def close(self) -> None:
        distributed = getattr(self.torch, "distributed", None)
        if (
            self.created_process_group
            and distributed is not None
            and distributed.is_initialized()
        ):
            distributed.destroy_process_group()
        self.created_process_group = False

    def __enter__(self) -> "RuntimeContext":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


_ALIASES = {
    "ane": "apple-ane",
    "apple_npu": "apple-ane",
    "metal": "mps",
    "ascend": "ascend-npu",
    "ascend_npu": "ascend-npu",
    "torch-npu": "ascend-npu",
    "torch_npu": "ascend-npu",
}
_KNOWN_ACCELERATORS = {"auto", "cpu", "cuda", "mps", "apple-ane", "ascend-npu"}


def _safe_import(module_name: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # Binary extension import failures matter here too.
        return None, f"{type(exc).__name__}: {exc}"


def _call_bool(owner: Any, name: str, default: bool = False) -> bool:
    function = getattr(owner, name, None)
    if function is None:
        return default
    try:
        return bool(function())
    except Exception:
        return default


def _device_count(owner: Any) -> int:
    function = getattr(owner, "device_count", None)
    if function is None:
        return 0
    try:
        return max(0, int(function()))
    except Exception:
        return 0


def probe_accelerators(
    *,
    system_name: str | None = None,
    machine: str | None = None,
) -> dict[str, AcceleratorProbe]:
    """Inspect runtimes without importing torch when this module is imported."""

    system_name = platform.system() if system_name is None else system_name
    machine = platform.machine() if machine is None else machine
    apple_silicon = system_name == "Darwin" and machine in {"arm64", "aarch64"}

    torch, torch_error = _safe_import("torch")
    torch_version = getattr(torch, "__version__", "unknown") if torch else None
    distributed = getattr(torch, "distributed", None) if torch else None
    distributed_available = bool(
        distributed is not None and _call_bool(distributed, "is_available")
    )

    if torch is None:
        torch_detail = "PyTorch import failed: " + (torch_error or "unknown error")
        cuda_available = False
        cuda_count = 0
        nccl_available = False
        mps_available = False
    else:
        torch_detail = f"PyTorch {torch_version}"
        cuda = getattr(torch, "cuda", None)
        cuda_available = bool(cuda is not None and _call_bool(cuda, "is_available"))
        cuda_count = _device_count(cuda) if cuda_available else 0
        nccl_available = bool(
            distributed_available and _call_bool(distributed, "is_nccl_available")
        )
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None) if backends is not None else None
        mps_available = bool(mps is not None and _call_bool(mps, "is_available"))

    torch_npu, npu_error = _safe_import("torch_npu")
    npu_api = None
    if torch is not None:
        npu_api = getattr(torch, "npu", None)
    if npu_api is None and torch_npu is not None:
        npu_api = getattr(torch_npu, "npu", None)
    npu_available = bool(npu_api is not None and _call_bool(npu_api, "is_available"))
    npu_count = _device_count(npu_api) if npu_available else 0
    npu_version = getattr(torch_npu, "__version__", "unknown") if torch_npu else None

    if torch_npu is None:
        npu_detail = "torch_npu import failed: " + (npu_error or "unknown error")
    else:
        npu_detail = f"torch_npu {npu_version}; {torch_detail}"

    return {
        "cpu": AcceleratorProbe(
            name="cpu",
            framework="PyTorch",
            runtime_available=torch is not None,
            hardware_available=True,
            device_count=1,
            ddp_supported=distributed_available,
            process_group_backend="gloo",
            detail=torch_detail,
        ),
        "cuda": AcceleratorProbe(
            name="cuda",
            framework="PyTorch CUDA",
            runtime_available=cuda_available,
            hardware_available=cuda_available,
            device_count=cuda_count,
            ddp_supported=cuda_available and nccl_available,
            process_group_backend="nccl",
            detail=torch_detail,
        ),
        "mps": AcceleratorProbe(
            name="mps",
            framework="PyTorch MPS",
            runtime_available=mps_available,
            hardware_available=apple_silicon,
            device_count=1 if mps_available else 0,
            ddp_supported=False,
            process_group_backend=None,
            detail=(
                f"{torch_detail}; MPS is single-process here and has no ANE collectives"
            ),
        ),
        "ascend-npu": AcceleratorProbe(
            name="ascend-npu",
            framework="PyTorch torch_npu",
            runtime_available=npu_available,
            hardware_available=npu_available,
            device_count=npu_count,
            ddp_supported=npu_available and distributed_available,
            process_group_backend="hccl",
            detail=npu_detail,
        ),
        "apple-ane": AcceleratorProbe(
            name="apple-ane",
            framework="native ANE private API",
            runtime_available=apple_silicon,
            hardware_available=apple_silicon,
            device_count=1 if apple_silicon else 0,
            ddp_supported=False,
            process_group_backend=None,
            detail=(
                "Apple ANE is not a torch device. The native Jishui path can place "
                "individual operators on ANE and Metal, but it cannot join DDP."
            ),
        ),
    }


def _normalize_accelerators(value: str | Sequence[str]) -> tuple[str, ...]:
    raw_values = value.replace("+", ",").split(",") if isinstance(value, str) else value
    normalized: list[str] = []
    for raw in raw_values:
        name = str(raw).strip().lower()
        if not name:
            continue
        if name == "npu":
            raise DistributedCapabilityError(
                "'npu' is ambiguous: use 'apple-ane' for an Apple Neural Engine "
                "or 'ascend-npu' for torch_npu/HCCL"
            )
        if name == "gpu":
            raise DistributedCapabilityError(
                "'gpu' is ambiguous: use 'cuda' or 'mps'"
            )
        name = _ALIASES.get(name, name)
        if name not in _KNOWN_ACCELERATORS:
            raise DistributedCapabilityError(f"unknown accelerator: {raw!r}")
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise DistributedCapabilityError("at least one accelerator is required")
    if "auto" in normalized and len(normalized) > 1:
        raise DistributedCapabilityError("'auto' cannot be combined with another accelerator")
    return tuple(normalized)


def _select_auto_accelerator() -> str:
    probes = probe_accelerators()
    for name in ("cuda", "ascend-npu", "mps", "cpu"):
        if probes[name].runtime_available:
            return name
    raise DistributedCapabilityError(
        "no usable PyTorch runtime was found; run --probe for details"
    )


def build_runtime_plan(
    accelerators: str | Sequence[str],
    launch: LaunchEnvironment | None = None,
) -> RuntimePlan:
    launch = LaunchEnvironment() if launch is None else launch
    requested = _normalize_accelerators(accelerators)
    if requested == ("auto",):
        requested = (_select_auto_accelerator(),)
    if len(requested) != 1:
        joined = ", ".join(requested)
        raise DistributedCapabilityError(
            "one DDP process group cannot mix accelerator types "
            f"({joined}). CUDA uses NCCL, Ascend NPU uses HCCL, and Apple ANE "
            "has no PyTorch collective backend. Use a homogeneous job."
        )

    accelerator = requested[0]
    if accelerator == "apple-ane":
        raise DistributedCapabilityError(
            "Apple ANE cannot be a PyTorch/DDP replica. Use native/ane for "
            "operator-level ANE+Metal execution, or use homogeneous CUDA DDP."
        )
    if accelerator == "mps" and launch.world_size > 1:
        raise DistributedCapabilityError(
            "PyTorch MPS is supported only as a single process in this project; "
            "it cannot form a distributed group with Apple ANE"
        )

    backend = {
        "cpu": "gloo",
        "cuda": "nccl",
        "mps": None,
        "ascend-npu": "hccl",
    }[accelerator]
    device = {
        "cpu": "cpu",
        "cuda": f"cuda:{launch.local_rank}",
        "mps": "mps",
        "ascend-npu": f"npu:{launch.local_rank}",
    }[accelerator]
    return RuntimePlan(
        accelerator=accelerator,
        device=device,
        process_group_backend=backend,
        rank=launch.rank,
        local_rank=launch.local_rank,
        world_size=launch.world_size,
    )


def check_runtime(plan: RuntimePlan) -> AcceleratorProbe:
    probe = probe_accelerators()[plan.accelerator]
    if not probe.runtime_available:
        raise DistributedCapabilityError(
            f"{plan.accelerator} runtime is unavailable: {probe.detail}"
        )
    if plan.distributed and not probe.ddp_supported:
        raise DistributedCapabilityError(
            f"{plan.accelerator} has no usable distributed backend: {probe.detail}"
        )
    if plan.accelerator in {"cuda", "ascend-npu"}:
        if plan.local_rank >= probe.device_count:
            raise DistributedCapabilityError(
                f"LOCAL_RANK={plan.local_rank} exceeds the {probe.device_count} "
                f"visible {plan.accelerator} device(s)"
            )
    return probe


def initialize_runtime(plan: RuntimePlan) -> RuntimeContext:
    """Bind the local device and optionally initialize a homogeneous process group."""

    check_runtime(plan)
    torch, torch_error = _safe_import("torch")
    if torch is None:
        raise DistributedCapabilityError(
            "PyTorch import failed after probing: " + (torch_error or "unknown error")
        )

    if plan.accelerator == "cuda":
        torch.cuda.set_device(plan.local_rank)
    elif plan.accelerator == "ascend-npu":
        torch_npu, npu_error = _safe_import("torch_npu")
        if torch_npu is None:
            raise DistributedCapabilityError(
                "torch_npu import failed after probing: " + (npu_error or "unknown error")
            )
        npu_api = getattr(torch, "npu", None) or getattr(torch_npu, "npu", None)
        npu_api.set_device(plan.device)

    created_process_group = False
    if plan.distributed:
        if torch.distributed.is_initialized():
            raise DistributedCapabilityError("a torch process group is already initialized")
        torch.distributed.init_process_group(
            backend=plan.process_group_backend,
            init_method="env://",
            rank=plan.rank,
            world_size=plan.world_size,
        )
        created_process_group = True

    return RuntimeContext(
        plan=plan,
        torch=torch,
        torch_device=torch.device(plan.device),
        created_process_group=created_process_group,
    )
