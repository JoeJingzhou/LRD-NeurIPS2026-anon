"""
Pure LLM model list and benchmark scores for geometric analysis.

These are base (non-embedding-tuned) language models.
Downstream quality metric: MMLU (5-shot) from public leaderboards / model cards.
All scores are the same across datasets (model-level quality, not task-specific).
"""

LLM_MODELS = [
    ("openai-community/gpt2", "gpt2"),
    ("openai-community/gpt2-xl", "gpt2_xl"),
    ("microsoft/phi-2", "phi_2"),
    ("Qwen/Qwen2-1.5B", "Qwen2_1_5B"),
    ("Qwen/Qwen2-7B", "Qwen2_7B"),
    ("mistralai/Mistral-7B-v0.1", "Mistral_7B_v0_1"),
    ("google/gemma-2-2b", "gemma_2_2b"),
    ("tiiuae/falcon-7b", "falcon_7b"),
    ("unsloth/Llama-3.2-1B", "Llama_3_2_1B"),
]

# MMLU (5-shot) - model-level quality score, same value for every dataset.
_MMLU = {
    "gpt2": 26.0,
    "gpt2_xl": 26.0,
    "phi_2": 57.9,
    "Qwen2_1_5B": 56.5,
    "Qwen2_7B": 70.5,
    "Mistral_7B_v0_1": 64.1,
    "gemma_2_2b": 51.3,
    "falcon_7b": 27.8,
    "Llama_3_2_1B": 32.2,
}


def llm_benchmark_scores_for(dataset_keys):
    """Build a MTEB_SCORES-shaped dict: {ds_key: {short_name: score}}."""
    return {ds: dict(_MMLU) for ds in dataset_keys}
