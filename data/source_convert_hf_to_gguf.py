#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import logging
import argparse
import contextlib
import json
import os
import re
import sys
from enum import IntEnum
from pathlib import Path
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Iterable, Iterator, Literal, Sequence, TypeVar, cast
from itertools import chain
from transformers import AutoConfig

import math
import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf
from gguf.vocab import MistralTokenizerType, MistralVocab

try:
    from mistral_common.tokens.tokenizers.base import TokenizerVersion # pyright: ignore[reportMissingImports]
    from mistral_common.tokens.tokenizers.multimodal import DATASET_MEAN as _MISTRAL_COMMON_DATASET_MEAN, DATASET_STD as _MISTRAL_COMMON_DATASET_STD # pyright: ignore[reportMissingImports]
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer # pyright: ignore[reportMissingImports]
    from mistral_common.tokens.tokenizers.sentencepiece import ( # pyright: ignore[reportMissingImports]
        SentencePieceTokenizer,
    )

    _mistral_common_installed = True
    _mistral_import_error_msg = ""
except ImportError:
    _MISTRAL_COMMON_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _MISTRAL_COMMON_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

    _mistral_common_installed = False
    TokenizerVersion = None
    Tekkenizer = None
    SentencePieceTokenizer = None
    _mistral_import_error_msg = (
        "Mistral format requires `mistral-common` to be installed. Please run "
        "`pip install mistral-common[image,audio]` to install it."
    )


logger = logging.getLogger("hf-to-gguf")


###### MODEL DEFINITIONS ######

class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


class ModelType(IntEnum):
    TEXT = 1
    MMPROJ = 2


AnyModel = TypeVar("AnyModel", bound="type[ModelBase]")


class ModelBase:
    _model_classes: dict[ModelType, dict[str, type[ModelBase]]] = {
        ModelType.TEXT: {},
        ModelType.MMPROJ: {},
    }

    dir_model: Path
    ftype: gguf.LlamaFileType
    fname_out: Path
    is_big_endian: bool
    endianess: gguf.GGUFEndian
    use_temp_file: bool
    lazy: bool
    dry_run: bool
    hparams: dict[str, Any]
    model_tensors: dict[str, Callable[[], Tensor]]
    gguf_writer: gguf.GGUFWriter
    model_name: str | None
    metadata_override: Path | None
    dir_model_card: Path
    remote_hf_model_id: str | None

    # subclasses should define this!
    model_arch: gguf.MODEL_ARCH

    # subclasses should initialize this!
    block_count: int
    tensor_map: gguf.TensorNameMap

    # Mistral format specifics
    is_mistral_format: bool = False
    disable_mistral_community_chat_template: bool = False
    sentence_transformers_dense_modules: bool = False

    def __init__(self, dir_model: Path, ftype: gguf.LlamaFileType, fname_out: Path, *, is_big_endian: bool = False,
                 use_temp_file: bool = False, eager: bool = False,
                 metadata_override: Path | None = None, model_name: str | None = None,
                 split_max_tensors: int = 0, split_max_size: int = 0, dry_run: bool = False,
                 small_first_shard: bool = False, hparams: dict[str, Any] | None = None, remote_hf_model_id: str | None = None,
                 disable_mistral_community_chat_template: bool = False,
                 sentence_transformers_dense_modules: bool = False):
        if type(self) is ModelBase or \
                type(self) is TextModel or \
                type(self) is MmprojModel:
            raise TypeError(f"{type(self).__name__!r} should not be directly instantiated")

        if self.is_mistral_format and not _mistral_common_installed:
            raise ImportError(_mistral_import_error_msg)

        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.use_temp_file = use_temp_file
        self.lazy = not eager or (remote_hf_model_id is not None)
        self.dry_run = dry_run
        self.remote_hf_model_id = remote_hf_model_id
        self.sentence_transformers_dense_modules = sentence_transformers_dense_modules
        self.hparams = ModelBase.load_hparams(self.dir_model, self.is_mistral_format) if hparams is None else hparams
        self.model_tensors = self.index_tensors(remote_hf_model_id=remote_hf_model_id)
        self.metadata_override = metadata_override
        self.model_name = model_name
        self.dir_model_card = dir_model  # overridden in convert_lora_to_gguf.py

        # Apply heuristics to figure out typical tensor encoding based on first tensor's dtype
        # NOTE: can't use field "torch_dtype" in config.json, because some finetunes lie.
        if self.ftype == gguf.LlamaFileType.GUESSED:
            for _, tensor in self.get_tensors():
                if tensor.dim() < 2:
                    continue

                if tensor.dtype == torch.bfloat16:
                    self.ftype = gguf.LlamaFileType.MOSTLY_BF16
                    logger.info("heuristics detected bfloat16 tensor dtype, setting --outtype bf16")
                    break
                elif tensor.dtype == torch.float16:
                    self.ftype = gguf.LlamaFileType.MOSTLY_F16
                    logger.info("heuristics detected float16 tensor dtype, setting --outtype f16")
                    break
            else:
                self.ftype = gguf.LlamaFileType.MOSTLY_F16
                logger.info("heuristics unable to detect tensor dtype, defaulting to --outtype f16")

        self.dequant_model()

        # Configure GGUF Writer
        self.gguf_writer = gguf.GGUFWriter(path=None, arch=gguf.MODEL_ARCH_NAMES[self.model_arch], endianess=self.endianess, use_temp_file=self.use_temp_file,
                                           split_max_tensors=split_max_tensors, split_max_size=split_max_size, dry_run=dry_run, small_first_shard=small_first_shard)

        # Mistral specific
        self.disable_mistral_community_chat_template = disable_mistral_community_chat_template

    @classmethod
    def add_prefix_to_filename(cls, path: Path, prefix: str) -> Path:
        stem, suffix = path.stem, path.suffix
        new_name = f"{prefix}{stem}{suffix}"
        return path.with_name(new_name)

    def find_hparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors: dict[str, Callable[[], Tensor]] = {}

        if remote_hf_model_id is not None:
            is_safetensors = True

            logger.info(f"Using remote model with HuggingFace id: {remote_hf_model_id}")
            remote_tensors = gguf.utility.SafetensorRemote.get_list_tensors_hf_model(remote_hf_model_id)
            for name, remote_tensor in remote_tensors.items():
                tensors[name] = lambda r=remote_tensor: LazyTorchTensor.from_remote_tensor(r)

            return tensors

        prefix = "model" if not self.is_mistral_format else "consolidated"
        part_names: list[str] = ModelBase.get_model_part_names(self.dir_model, prefix, ".safetensors")
        is_safetensors: bool = len(part_names) > 0
        if not is_safetensors:
            part_names = ModelBase.get_model_part_names(self.dir_model, "pytorch_model", ".bin")

        tensor_names_from_index: set[str] = set()

        if not self.is_mistral_format:
            index_name = "model.safetensors" if is_safetensors else "pytorch_model.bin"
            index_name += ".index.json"
            index_file = self.dir_model / index_name

            if index_file.is_file():
                logger.info(f"gguf: loading model weight map from '{index_name}'")
                with open(index_file, "r", encoding="utf-8") as f:
                    index: dict[str, Any] = json.load(f)
                    weight_map = index.get("weight_map")
                    if weight_map is None or not isinstance(weight_map, dict):
                        raise ValueError(f"Can't load 'weight_map' from {index_name!r}")
                    tensor_names_from_index.update(weight_map.keys())
                    part_dict: dict[str, None] = dict.fromkeys(weight_map.values(), None)
                    part_names = sorted(part_dict.keys())
            else:
                weight_map = {}
        else:
            weight_map = {}

        for part_name in part_names:
            logger.info(f"gguf: indexing model part '{part_name}'")
            ctx: ContextManager[Any]
            if is_safetensors:
                ctx = cast(ContextManager[Any], gguf.utility.SafetensorsLocal(self.dir_model / part_name))
            else:
                ctx = contextlib.nullcontext(torch.load(str(self.dir_model / part_name), map_location="cpu", mmap=True, weights_only=True))

            with ctx as model_part:
                assert model_part is not None

                for name in model_part.keys():
                    if is_safetensors:
                        data: gguf.utility.LocalTensor = model_part[name]
                        if self.lazy:
                            data_gen = lambda data=data: LazyTorchTensor.from_local_tensor(data)  # noqa: E731
                        else:
                            dtype = LazyTorchTensor._dtype_str_map[data.dtype]
                            data_gen = lambda data=data, dtype=dtype: torch.from_numpy(data.mmap_bytes()).view(dtype).reshape(data.shape)  # noqa: E731
                    else:
                        data_torch: Tensor = model_part[name]
                        if self.lazy:
                            data_gen = lambda data=data_torch: LazyTorchTensor.from_eager(data)  # noqa: E731
                        else:
                            data_gen = lambda data=data_torch: data  # noqa: E731
                    tensors[name] = data_gen

        # verify tensor name presence and identify potentially missing files
        if len(tensor_names_from_index) > 0:
            tensor_names_from_parts = set(tensors.keys())
            if len(tensor_names_from_parts.symmetric_difference(tensor_names_from_index)) > 0:
                missing = sorted(tensor_names_from_index.difference(tensor_names_from_parts))
                extra = sorted(tensor_names_from_parts.difference(tensor_names_from_index))
                missing_files = sorted(set(weight_map[n] for n in missing if n in weight_map))
                if len(extra) == 0 and len(missing_files) > 0:
                    raise ValueError(f"Missing or incomplete model files: {missing_files}\n"
                                     f"Missing tensors: {missing}")
                else:
                    raise ValueError("Mismatch between weight map and model parts for tensor names:\n"
                                     f"Missing tensors: {missing}\n"
                                     f"Extra tensors: {extra}")

        return tensors

    def dequant_model(self):
        tensors_to_remove: list[str] = []
        new_tensors: dict[str, Callable[[], Tensor]] = {}

        if (quant_config := self.hparams.get("quantization_config")) and isinstance(quant_config, dict):
            quant_method = quant_config.get("quant_method")

            def dequant_bitnet(weight: Tensor, scale: Tensor) -> Tensor:
                weight = weight.view(torch.uint8)
                orig_shape = weight.shape

                shift = torch.tensor([0, 2, 4, 6], dtype=torch.uint8).reshape((4, *(1 for _ in range(len(orig_shape)))))
                data = weight.unsqueeze(0).expand((4, *orig_shape)) >> shift
                data = data & 3
                data = (data.float() - 1).reshape((orig_shape[0] * 4, *orig_shape[1:]))

                # The scale is inverted
                return data / scale.float()

            def dequant_simple(weight: Tensor, scale: Tensor, block_size: Sequence[int] | None = None) -> Tensor:
                scale = scale.float()

                if block_size is not None:
                    for i, size in enumerate(block_size):
                        scale = scale.repeat_interleave(size, i)
                    # unpad the scale (e.g. when the tensor size isn't a multiple of the block size)
                    scale = scale[tuple(slice(0, size) for size in weight.shape)]

                return weight.float() * scale

            # ref: https://github.com/ModelCloud/GPTQModel/blob/037c5c0f6c9e33c500d975b038d02e7ca437546d/gptqmodel/nn_modules/qlinear/__init__.py#L437-L476
            def dequant_gptq(g_idx: Tensor, qweight: Tensor, qzeros: Tensor, scales: Tensor) -> Tensor:
                bits = quant_config["bits"]
                assert bits in (2, 3, 4, 8)
                assert qweight.dtype == qzeros.dtype
                maxq = (2 ** bits) - 1
                weight = None
                zeros = None
                pack_dtype_bits = qweight.dtype.itemsize * 8

                if bits in [2, 4, 8]:
                    pack_factor = pack_dtype_bits // bits
                    wf = torch.tensor(list(range(0, pack_dtype_bits, bits)), dtype=torch.int32).unsqueeze(0)
                    if self.lazy:
                        wf = LazyTorchTensor.from_eager(wf)

                    zeros = torch.bitwise_right_shift(
                        qzeros.unsqueeze(2).expand(-1, -1, pack_factor),
                        wf.unsqueeze(0)
                    ).to(torch.int16 if bits == 8 else torch.int8)
                    zeros = torch.bitwise_and(zeros, maxq).reshape(scales.shape)

                    weight = torch.bitwise_and(
                        torch.bitwise_right_shift(
                            qweight.unsqueeze(1).expand(-1, pack_factor, -1),
                            wf.unsqueeze(-1)
                        ).to(torch.int16 if bits == 8 else torch.int8),
                        maxq
                    )
                elif bits == 3:
                    raise NotImplementedError("3-bit gptq dequantization is not yet implemented")

                assert weight is not None
                assert zeros is not None

                weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

                # gptq_v2 doesn't need to offset zeros
                if quant_config.get("checkpoint_format", "gptq") == "gptq":
                    zeros += 1

                return (scales[g_idx].float() * (weight - zeros[g_idx]).float()).T

            def dequant_packed(w: Tensor, scale: Tensor, shape_tensor: Tensor, zero_point: Tensor | None, num_bits: int, group_size: int):
                assert w.dtype == torch.int32
                shape = tuple(shape_tensor.tolist())
                assert len(shape) == 2
                mask = (1 << num_bits) - 1

                shifts = torch.arange(0, 32 - (num_bits - 1), num_bits, dtype=torch.int32)
                if self.lazy:
                    shifts = LazyTorchTensor.from_eager(shifts)

                if zero_point is None:
                    offset = 1 << (num_bits - 1)
                else:
                    assert len(zero_point.shape) == 2
                    offset = (zero_point.unsqueeze(1) >> shifts.reshape(1, -1, 1)) & mask
                    offset = offset.reshape(-1, zero_point.shape[1])
                    # trim padding, and prepare for broadcast
                    # NOTE: the zero-point is packed along di