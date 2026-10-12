# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AITER per-token FP8 linear support on gfx1100."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm._aiter_ops import (
    _aiter_w8a8_uses_triton,
    _rocm_aiter_w8a8_gemm_impl,
    _triton_gemm_config_exists,
    rocm_aiter_ops,
)
from vllm.model_executor.kernels.linear import FP8ScaledMMLinearLayerConfig
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    AiterPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)

pytestmark = pytest.mark.cpu_test

OPS = "vllm._aiter_ops"
AITER_KERNELS = "vllm.model_executor.kernels.linear.scaled_mm.aiter"
SPARSE_INDEXER = "rocm_aiter_sparse_attn_indexer"


@pytest.fixture(autouse=True)
def _clear_caches():
    for fn in (
        rocm_aiter_ops.is_per_token_w8a8_gemm_tuned,
        rocm_aiter_ops.has_per_token_w8a8_gemm_config,
    ):
        fn.cache_clear()
    yield
    for fn in (
        rocm_aiter_ops.is_per_token_w8a8_gemm_tuned,
        rocm_aiter_ops.has_per_token_w8a8_gemm_config,
    ):
        fn.cache_clear()


def _rocm_arch(**flags):
    """Mock ROCm architecture predicates without probing the GPU."""
    module = types.ModuleType("vllm.platforms.rocm")
    for name in ("on_gfx1100", "on_gfx11", "on_gfx950", "on_rdna", "on_rdna4"):
        setattr(module, name, MagicMock(return_value=flags.get(name, False)))
    return patch.dict(sys.modules, {"vllm.platforms.rocm": module})


def _fp8_config(act=kFp8DynamicTokenSym, weight=kFp8StaticChannelSym):
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=weight,
        activation_quant_key=act,
        weight_shape=(4096, 3840),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(
    "gfx1100, aiter_enabled, linear_enabled, expected",
    [
        (True, True, True, True),
        (True, False, True, False),
        (True, True, False, False),
        (False, True, True, False),
    ],
)
def test_gfx1100_per_token_linear_gate(
    gfx1100, aiter_enabled, linear_enabled, expected
):
    with (
        patch(f"{OPS}.current_platform", is_rocm=MagicMock(return_value=True)),
        patch(f"{OPS}.IS_AITER_FOUND", True),
        _rocm_arch(on_gfx1100=gfx1100),
        patch.object(rocm_aiter_ops, "_AITER_ENABLED", aiter_enabled),
        patch.object(rocm_aiter_ops, "_LINEAR_ENABLED", linear_enabled),
    ):
        assert rocm_aiter_ops.is_gfx1100_per_token_linear_enabled() is expected


def test_gfx1100_gate_requires_aiter():
    with (
        patch(f"{OPS}.current_platform", is_rocm=MagicMock(return_value=True)),
        patch(f"{OPS}.IS_AITER_FOUND", False),
        _rocm_arch(on_gfx1100=True),
        patch.object(rocm_aiter_ops, "_AITER_ENABLED", True),
        patch.object(rocm_aiter_ops, "_LINEAR_ENABLED", True),
    ):
        assert not rocm_aiter_ops.is_gfx1100_per_token_linear_enabled()


def _fake_aiter():
    ck = MagicMock(name="gemm_a8w8_CK")
    triton = MagicMock(name="gemm_a8w8")
    modules = {
        "aiter": types.SimpleNamespace(gemm_a8w8_CK=ck),
        "aiter.ops": types.ModuleType("aiter.ops"),
        "aiter.ops.triton": types.ModuleType("aiter.ops.triton"),
        "aiter.ops.triton.gemm_a8w8": types.SimpleNamespace(gemm_a8w8=triton),
    }
    return modules, ck, triton


@pytest.mark.parametrize(
    "rdna, scale_shape",
    [
        (True, ((4, 1), (1, 8))),  # gfx11/gfx12: per-token x per-channel
        (False, ((4, 1), (1, 8))),  # CDNA: per-token x per-channel
        (False, ((1,), (1,))),  # CDNA: per-tensor INT8 scales stay on CK
    ],
)
def test_w8a8_gemm_dispatch(rdna, scale_shape):
    modules, ck, triton = _fake_aiter()
    with (
        patch(f"{OPS}.current_platform", is_rocm=MagicMock(return_value=True)),
        patch(f"{OPS}.IS_AITER_FOUND", True),
        _rocm_arch(on_rdna=rdna),
        patch.dict(sys.modules, modules),
    ):
        assert _aiter_w8a8_uses_triton() is rdna
        a = torch.zeros(4, 16, dtype=torch.int8)
        b = torch.zeros(8, 16, dtype=torch.int8)
        _rocm_aiter_w8a8_gemm_impl(
            a,
            b,
            torch.ones(scale_shape[0]),
            torch.ones(scale_shape[1]),
            None,
            torch.bfloat16,
        )
    (triton if rdna else ck).assert_called_once()
    (ck if rdna else triton).assert_not_called()


@pytest.mark.parametrize("rdna", [True, False])
def test_per_token_tuning_check_follows_backend(rdna):
    with (
        patch(f"{OPS}.current_platform", is_rocm=MagicMock(return_value=True)),
        patch(f"{OPS}.IS_AITER_FOUND", True),
        _rocm_arch(on_rdna=rdna),
        patch(f"{OPS}._triton_gemm_config_is_tuned", return_value=False) as triton,
        patch(f"{OPS}._ck_gemm_shape_is_tuned", return_value=False) as ck,
    ):
        assert not rocm_aiter_ops.is_per_token_w8a8_gemm_tuned(
            4096, 3840, torch.float8_e4m3fn
        )
    if rdna:
        triton.assert_called_once_with("GEMM-A8W8", 4096, 3840)
        ck.assert_not_called()
    else:
        ck.assert_called_once()
        triton.assert_not_called()


@pytest.mark.parametrize(
    "side_effect, expected",
    [
        (None, True),
        (AssertionError("Required config file doesn't exist"), False),
        (KeyError("no M bucket"), False),
    ],
)
def test_triton_gemm_config_exists(side_effect, expected):
    get_gemm_config = MagicMock(return_value=({}, False), side_effect=side_effect)
    module = types.SimpleNamespace(get_gemm_config=get_gemm_config)
    with patch.dict(sys.modules, {"aiter.ops.triton.utils.gemm_config_utils": module}):
        assert _triton_gemm_config_exists("GEMM-A8W8", 4096, 3840) is expected


def _bucket_lookup(keys):
    """get_gemm_config over a single config file with the given M keys:
    M_LEQ_x ascending, then M_GEQ_x descending, then "any", else KeyError."""
    leq = [int(k[6:]) for k in keys if k.startswith("M_LEQ_")]
    geq = [int(k[6:]) for k in keys if k.startswith("M_GEQ_")]

    def get_gemm_config(config_name, M, N, K):
        if any(b >= M for b in leq) or any(b <= M for b in geq) or "any" in keys:
            return {}, False
        raise KeyError(M)

    return get_gemm_config


@pytest.mark.parametrize(
    "keys, expected",
    [
        ({"M_LEQ_32", "M_LEQ_256", "any"}, True),  # gfx1100 DEFAULT.json layout
        ({"any"}, True),
        ({"M_LEQ_16", "M_GEQ_4096"}, False),  # gap at 17 <= M < 4096
        ({"M_LEQ_8192"}, False),  # nothing above 8192
        ({"M_GEQ_4"}, False),  # nothing below 4
    ],
)
def test_triton_gemm_config_coverage(keys, expected):
    module = types.SimpleNamespace(get_gemm_config=_bucket_lookup(keys))
    with patch.dict(sys.modules, {"aiter.ops.triton.utils.gemm_config_utils": module}):
        assert _triton_gemm_config_exists("GEMM-A8W8", 4096, 3840) is expected


def test_incomplete_config_is_treated_as_untuned():
    with (
        patch(f"{OPS}._aiter_w8a8_uses_triton", return_value=True),
        patch(f"{OPS}._triton_gemm_config_is_tuned", side_effect=KeyError(128)),
    ):
        assert not rocm_aiter_ops.is_per_token_w8a8_gemm_tuned(
            4096, 3840, torch.float8_e4m3fn
        )


def test_has_config_is_false_on_ck_backend():
    with (
        patch(f"{OPS}._aiter_w8a8_uses_triton", return_value=False),
        patch(f"{OPS}._triton_gemm_config_exists") as exists,
    ):
        assert not rocm_aiter_ops.has_per_token_w8a8_gemm_config(4096, 3840)
    exists.assert_not_called()


def test_gfx1100_enables_only_unshuffled_per_token_kernel():
    with (
        patch.object(rocm_aiter_ops, "is_rdna_linear_enabled", return_value=False),
        patch.object(
            rocm_aiter_ops, "is_gfx1100_per_token_linear_enabled", return_value=True
        ),
        patch.object(rocm_aiter_ops, "is_linear_fp8_enabled", return_value=False),
        patch.object(rocm_aiter_ops, "is_linear_hipbmm_enabled", return_value=False),
        patch(f"{AITER_KERNELS}.current_platform") as platform,
    ):
        platform.is_rocm.return_value = True
        assert AiterPerTokenFp8ScaledMMLinearKernel.is_supported()[0]
        assert not AiterPreshuffledPerTokenFp8ScaledMMLinearKernel.is_supported()[0]
        assert not AiterHipbMMPerTokenFp8ScaledMMLinearKernel.is_supported()[0]


@pytest.mark.parametrize(
    "gfx1100, tuned, has_config, expected",
    [
        (True, True, True, True),  # tuned shape
        (True, False, True, True),  # gfx1100: untuned, AITER default config
        (True, False, False, False),  # gfx1100: no AITER config for the arch
        (False, False, True, False),  # RDNA4/CDNA: tuned-only rule unchanged
        (False, True, True, True),
    ],
)
def test_per_token_can_implement_tuning_policy(gfx1100, tuned, has_config, expected):
    with (
        patch(f"{AITER_KERNELS}.current_platform") as platform,
        patch.object(
            rocm_aiter_ops,
            "is_gfx1100_per_token_linear_enabled",
            return_value=gfx1100,
        ),
        patch.object(
            rocm_aiter_ops, "is_per_token_w8a8_gemm_tuned", return_value=tuned
        ),
        patch.object(
            rocm_aiter_ops,
            "has_per_token_w8a8_gemm_config",
            return_value=has_config,
        ),
    ):
        platform.fp8_dtype.return_value = torch.float8_e4m3fn
        ok, reason = AiterPerTokenFp8ScaledMMLinearKernel.can_implement(_fp8_config())
    assert ok is expected
    if not ok:
        assert reason is not None
        assert ("AITER GEMM-A8W8 config" in reason) is gfx1100


@pytest.mark.parametrize(
    "act, weight",
    [
        (kFp8StaticTensorSym, kFp8StaticTensorSym),
        (kFp8DynamicTensorSym, kFp8StaticTensorSym),
        (kFp8DynamicTokenSym, kFp8StaticTensorSym),
        (kFp8StaticTensorSym, kFp8StaticChannelSym),
    ],
)
def test_per_token_rejects_non_per_token_configs_on_gfx1100(act, weight):
    with (
        patch(f"{AITER_KERNELS}.current_platform") as platform,
        patch.object(
            rocm_aiter_ops, "is_gfx1100_per_token_linear_enabled", return_value=True
        ),
        patch.object(
            rocm_aiter_ops, "has_per_token_w8a8_gemm_config", return_value=True
        ),
    ):
        platform.fp8_dtype.return_value = torch.float8_e4m3fn
        ok, _ = AiterPerTokenFp8ScaledMMLinearKernel.can_implement(
            _fp8_config(act, weight)
        )
    assert not ok


@pytest.mark.parametrize(
    "gfx11, gfx1100, gfx950, expected",
    [
        (True, True, False, [SPARSE_INDEXER, "rocm_aiter_w8a8_gemm"]),
        (True, False, False, [SPARSE_INDEXER]),  # other gfx11, e.g. gfx1101
        (False, False, True, [SPARSE_INDEXER]),
        (False, False, False, []),  # e.g. gfx1201 without AITER
    ],
)
def test_ops_registered_without_full_aiter(gfx11, gfx1100, gfx950, expected):
    register = MagicMock()
    with (
        patch(f"{OPS}.current_platform", is_rocm=MagicMock(return_value=True)),
        patch(f"{OPS}.is_aiter_found_and_supported", return_value=False),
        patch(f"{OPS}.is_aiter_found_and_supported_on_rdna4", return_value=False),
        patch(f"{OPS}.is_aiter_found_and_supported_on_gfx1100", return_value=gfx1100),
        _rocm_arch(on_gfx11=gfx11, on_gfx950=gfx950),
        patch(f"{OPS}.direct_register_custom_op", register),
        patch(f"{OPS}._OPS_REGISTERED", False),
    ):
        rocm_aiter_ops.register_ops_once()
    assert [c.kwargs["op_name"] for c in register.call_args_list] == expected
