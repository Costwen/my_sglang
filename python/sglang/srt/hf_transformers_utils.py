"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Utilities for Huggingface Transformers."""

import functools
import json
import os
import warnings
from typing import AbstractSet, Collection, Dict, Literal, Optional, Type, Union

from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from vllm.transformers_utils.configs import ChatGLMConfig, DbrxConfig

from sglang.srt.utils import is_multimodal_model

_CONFIG_REGISTRY: Dict[str, Type[PretrainedConfig]] = {
    ChatGLMConfig.model_type: ChatGLMConfig,
    DbrxConfig.model_type: DbrxConfig,
}


def download_from_hf(model_path: str):
    if os.path.exists(model_path):
        return model_path

    return snapshot_download(model_path, allow_patterns=["*.json", "*.bin", "*.model"])


def get_config_json(model_path: str):
    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)
    return config


def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_overide_args: Optional[dict] = None,
):
    config = AutoConfig.from_pretrained(
        model, trust_remote_code=trust_remote_code, revision=revision
    )
    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        config = config_class.from_pretrained(model, revision=revision)
    if model_overide_args:
        config.update(model_overide_args)
    return config


# Models don't use the same configuration key for determining the maximum
# context length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important. Some models have two of these and we
# have a preference for which value gets used.
CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_position_embeddings",
    "max_seq_len",
    "model_max_length",
]


def get_context_length(config):
    """Get the context length of a model from a huggingface model config."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = config.rope_scaling["factor"]
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling_factor = 1
        if config.rope_scaling.get("rope_type", None) == "llama3":
            rope_scaling_factor = 1
    else:
        rope_scaling_factor = 1

    for key in CONTEXT_LENGTH_KEYS:
        val = getattr(config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"

def custom_encode(self, prompt, **kwargs):
    # 进行分割
    parts = prompt.split('<image>')
    encoded_parts = []

    for part in parts:
        # 对分割后的每个部分进行编码
        if part:
            encoded = self._base_encode(part, **kwargs)
            encoded_parts.extend(encoded)
        # 插入特殊标记
        encoded_parts.append(-200)

    # 去掉最后一个多余的特殊标记
    if encoded_parts[-1] == -200:
        encoded_parts.pop()

    return encoded_parts


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if tokenizer_name.endswith(".json"):
        return TiktokenTokenizer(tokenizer_name)

    if tokenizer_name.endswith(".model"):
        return SentencePieceTokenizer(tokenizer_name)

    """Gets a tokenizer for the given model name via Huggingface."""
    if is_multimodal_model(tokenizer_name):
        processor = get_processor(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            **kwargs,
        )
        tokenizer = processor.tokenizer
        return tokenizer

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if (
        "llama" in tokenizer_name.lower()
        and kwargs.get("use_fast", True)
        and tokenizer_name != _FAST_LLAMA_TOKENIZER
    ):
        pass
        # warnings.warn(
        #    "For some LLaMA V1 models, initializing the fast tokenizer may "
        #    "take a long time. To reduce the initialization time, consider "
        #    f"using '{_FAST_LLAMA_TOKENIZER}' instead of the original "
        #    "tokenizer."
        # )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            **kwargs,
        )
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA V1 model "
            f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
            "original tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not trust_remote_code and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )
    tokenizer._base_encode = tokenizer.encode
    tokenizer.encode = lambda prompt, **kwargs: custom_encode(tokenizer, prompt, **kwargs)
    return tokenizer


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
):
    processor = AutoProcessor.from_pretrained(
        tokenizer_name,
        *args,
        trust_remote_code=trust_remote_code,
        tokenizer_revision=tokenizer_revision,
        **kwargs,
    )
    return processor


class TiktokenTokenizer:
    def __init__(self, tokenizer_path):
        import tiktoken
        from jinja2 import Template

        PAT_STR_B = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

        # Read JSON
        name = "tmp-json"
        with open(tokenizer_path, "rb") as fin:
            tok_dict = json.load(fin)

        mergeable_ranks = {
            bytes(item["bytes"]): item["token"] for item in tok_dict["regular_tokens"]
        }
        special_tokens = {
            bytes(item["bytes"]).decode(): item["token"]
            for item in tok_dict["special_tokens"]
        }
        assert tok_dict["word_split"] == "V1"

        kwargs = {
            "name": name,
            "pat_str": tok_dict.get("pat_str", PAT_STR_B),
            "mergeable_ranks": mergeable_ranks,
            "special_tokens": special_tokens,
        }
        if "default_allowed_special" in tok_dict:
            default_allowed_special = set(
                [
                    bytes(bytes_list).decode()
                    for bytes_list in tok_dict["default_allowed_special"]
                ]
            )
        else:
            default_allowed_special = None
        if "vocab_size" in tok_dict:
            kwargs["explicit_n_vocab"] = tok_dict["vocab_size"]

        tokenizer = tiktoken.Encoding(**kwargs)
        tokenizer._default_allowed_special = default_allowed_special or set()
        tokenizer._default_allowed_special |= {"<|separator|>"}

        def encode_patched(
            self,
            text: str,
            *,
            allowed_special: Union[
                Literal["all"], AbstractSet[str]
            ] = set(),  # noqa: B006
            disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        ) -> list[int]:
            if isinstance(allowed_special, set):
                allowed_special |= self._default_allowed_special
            return tiktoken.Encoding.encode(
                self,
                text,
                allowed_special=allowed_special,
                disallowed_special=disallowed_special,
            )

        tokenizer.encode = functools.partial(encode_patched, tokenizer)

        # Convert to HF interface
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer._special_tokens["<|eos|>"]
        self.vocab_size = tokenizer.n_vocab
        self.chat_template = Template(
            "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'Human: ' + message['content'].strip() + '<|separator|>\n\n' }}{% elif message['role'] == 'system' %}{{ 'System: ' + message['content'].strip() + '<|separator|>\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: '  + message['content'] + '<|separator|>\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
        )

    def encode(self, x, add_special_tokens=False):
        return self.tokenizer.encode(x)

    def decode(self, x):
        return self.tokenizer.decode(x)

    def batch_decode(
        self, batch, skip_special_tokens=True, spaces_between_special_tokens=False
    ):
        if isinstance(batch[0], int):
            batch = [[x] for x in batch]
        return self.tokenizer.decode_batch(batch)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        ret = self.chat_template.render(
            messages=messages, add_generation_prompt=add_generation_prompt
        )
        return self.encode(ret) if tokenize else ret


class SentencePieceTokenizer:
    def __init__(self, tokenizer_path):
        import sentencepiece as spm
        from jinja2 import Template

        tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)

        # Convert to HF interface
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_id()
        self.vocab_size = tokenizer.vocab_size()
        self.chat_template = Template(
            "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'Human: ' + message['content'].strip() + '<|separator|>\n\n' }}{% elif message['role'] == 'system' %}{{ 'System: ' + message['content'].strip() + '<|separator|>\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: '  + message['content'] + '<|separator|>\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
        )

    def encode(self, x, add_special_tokens=False):
        return self.tokenizer.encode(x)

    def decode(self, x):
        return self.tokenizer.decode(x)

    def batch_decode(
        self, batch, skip_special_tokens=True, spaces_between_special_tokens=False
    ):
        if isinstance(batch[0], int):
            batch = [[x] for x in batch]
        return self.tokenizer.decode(batch)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        ret = self.chat_template.render(
            messages=messages, add_generation_prompt=add_generation_prompt
        )
        return self.encode(ret) if tokenize else ret
