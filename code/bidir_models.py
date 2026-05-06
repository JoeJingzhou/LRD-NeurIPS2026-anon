"""
Bidirectional LLM embedding models for geometric analysis.

Four models covering three different bidirectionalization methods:
  1. NV-Embed-v2      -- Mistral-7B with causal mask removed
  2. LLM2Vec-Mistral  -- Mistral-7B with LoRA adapter + MNTP bidirectional training
  3. Nemotron Embed 1B -- Llama-3.2-1B with native bidirectional attention flag
  4. ModernBERT-large  -- Native bidirectional encoder (not decoder-based)

Paired comparisons (bidirectional vs causal on same architecture):
  NV-Embed-v2      vs  Mistral-7B-v0.1    (already in LLM experiments)
  LLM2Vec-Mistral  vs  Mistral-7B-v0.1    (already in LLM experiments)
  Nemotron 1B      vs  Llama-3.2-1B       (added to LLM experiments)
  ModernBERT       vs  existing BERT/BGE   (already in embedding experiments)
"""

import pathlib
import torch

BIDIR_MODELS = [
    ("nvidia/NV-Embed-v2", "NV_Embed_v2"),
    ("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", "LLM2Vec_Mistral_7B"),
    ("nvidia/llama-nemotron-embed-1b-v2", "Nemotron_Embed_1B"),
    ("answerdotai/ModernBERT-large", "ModernBERT_large"),
]

# LLM2Vec adapter -> base model mapping
_LLM2VEC_BASES = {
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp": "mistralai/Mistral-7B-Instruct-v0.2",
}


def _patch_nemotron_py39():
    """Inject `from __future__ import annotations` into Nemotron's cached
    custom code so that `torch.Tensor | None` type hints work on Python 3.9."""
    import shutil
    cache_root = pathlib.Path.home() / ".cache/huggingface/modules/transformers_modules"
    for py_file in cache_root.rglob("llama_bidirectional_model.py"):
        if "nemotron" not in str(py_file).lower():
            continue
        text = py_file.read_text()
        if "from __future__ import annotations" not in text and "torch.Tensor | None" in text:
            py_file.write_text("from __future__ import annotations\n" + text)
            pycache = py_file.parent / "__pycache__"
            if pycache.exists():
                shutil.rmtree(pycache)


def _load_llm2vec_as_bidir(adapter_name: str, device: str):
    """Load an LLM2Vec LoRA adapter on top of its base model, then make
    attention bidirectional by disabling causal masks.

    Returns (model, tokenizer, info_dict) in the same format as _load_model.
    """
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    from peft import PeftModel

    base_name = _LLM2VEC_BASES[adapter_name]
    config = AutoConfig.from_pretrained(base_name, trust_remote_code=True)
    n_layers = config.num_hidden_layers
    hidden_size = config.hidden_size

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModel.from_pretrained(
        base_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)

    model = PeftModel.from_pretrained(base_model, adapter_name)
    model = model.merge_and_unload()

    # Make attention bidirectional: set is_causal=False on every self-attention layer
    for layer in model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.is_causal = False

    model.eval()
    info = dict(
        n_layers=n_layers,
        hidden_size=hidden_size,
        arch=config.model_type,
        is_decoder=False,
    )
    return model, tokenizer, info


def _patch_nvembed_forward():
    """Patch NV-Embed-v2's cached BidirectionalMistralModel.forward to
    compute and pass position_embeddings to each decoder layer, which is
    required by newer transformers (>=4.45) MistralDecoderLayer."""
    import shutil
    cache_root = pathlib.Path.home() / ".cache/huggingface/modules/transformers_modules"
    for py_file in cache_root.rglob("modeling_nvembed.py"):
        if "NV" not in str(py_file):
            continue
        text = py_file.read_text()
        if "position_embeddings" in text:
            continue  # already patched

        # Add position_embeddings computation before the decoder loop
        old = """\
        hidden_states = inputs_embeds

        # decoder layers"""
        new = """\
        hidden_states = inputs_embeds

        # Compute rotary position embeddings once for all layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers"""
        if old not in text:
            continue

        text = text.replace(old, new)

        # Pass position_embeddings to each decoder_layer call (non-checkpointing path)
        old_call = """\
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )"""
        new_call = """\
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )"""
        text = text.replace(old_call, new_call)

        # Handle new transformers where decoder_layer returns a tensor, not tuple
        old_unpack = """\
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)"""
        new_unpack = """\
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

            if use_cache and isinstance(layer_outputs, tuple) and len(layer_outputs) > 1:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions and isinstance(layer_outputs, tuple) and len(layer_outputs) > 1:
                all_self_attns += (layer_outputs[1],)"""
        text = text.replace(old_unpack, new_unpack)

        # Guard next_decoder_cache against None
        text = text.replace(
            "next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache",
            "next_cache = (next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache) if next_decoder_cache is not None else None",
        )

        py_file.write_text(text)
        pycache = py_file.parent / "__pycache__"
        if pycache.exists():
            shutil.rmtree(pycache)
        print(f"  [NV-Embed patch] Patched {py_file}")


def _load_nvembed(model_name: str, device: str):
    """Load NV-Embed-v2 and return its inner BidirectionalMistralModel.

    The wrapper NVEmbedModel doesn't support output_hidden_states and its
    custom BidirectionalMistralModel.forward() is incompatible with newer
    transformers (missing position_embeddings kwarg).  We patch the cached
    source, then return the inner model so callers can treat it like a
    standard MistralModel.
    """
    from transformers import AutoModel, AutoTokenizer, AutoConfig

    _patch_nvembed_forward()

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tc = getattr(config, "text_config", None)
    if tc is not None:
        n_layers = tc.get("num_hidden_layers") if isinstance(tc, dict) else getattr(tc, "num_hidden_layers", None)
        hidden_size = tc.get("hidden_size") if isinstance(tc, dict) else getattr(tc, "hidden_size", None)
    else:
        n_layers = config.num_hidden_layers
        hidden_size = config.hidden_size

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wrapper = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)

    inner = wrapper.embedding_model
    inner.eval()

    # Attach the wrapper so it doesn't get GC'd
    inner._nvembed_wrapper = wrapper

    info = dict(n_layers=n_layers, hidden_size=hidden_size,
                arch="bidir_mistral", is_decoder=False)
    return inner, tokenizer, info


def is_nvembed_model(model_name: str) -> bool:
    return "nv-embed" in model_name.lower() or "nv_embed" in model_name.lower()


def is_llm2vec_model(model_name: str) -> bool:
    return model_name in _LLM2VEC_BASES


def _get_nested_config_value(config, attr: str):
    """Get a config attribute, falling back to text_config for wrapper models
    like NV-Embed-v2 where top-level num_hidden_layers is None."""
    val = getattr(config, attr, None)
    if val is not None:
        return val
    tc = getattr(config, "text_config", None)
    if tc is not None:
        if isinstance(tc, dict):
            return tc.get(attr)
        return getattr(tc, attr, None)
    return None
