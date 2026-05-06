#!/usr/bin/env python3
"""
Representational Frenet Curvature on Grassmann Manifold.

For each (dataset, model), extract embeddings from ALL hidden layers,
project each layer's embedding onto Grassmann manifold via SVD (top-r
right singular vectors), then compute two layer-wise primitives:
  - Speed:     Grassmann geodesic distance between adjacent subspaces
  - Curvature: Menger curvature from 3 consecutive subspaces

Outputs JSON + correlation analysis with MTEB, including partial
correlation controlling for log(dim).
"""

import argparse
import gc
import gzip
import json
import math
import os
import sys
import time
from typing import Dict, List, Set

os.environ["TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

MAX_N = None  # None = use all available texts per task
LAYERS_PER_PASS = 8
SUBSPACE_RANK = None        # None = adaptive via variance threshold
VARIANCE_THRESHOLD = 0.95   # retain 95% explained variance

MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "all_MiniLM_L6_v2"),
    ("intfloat/e5-large-v2", "e5_large_v2"),
    ("NovaSearch/stella_en_1.5B_v5", "stella_en_1.5B_v5"),
    ("BAAI/bge-multilingual-gemma2", "bge_multilingual_gemma2"),
    ("Alibaba-NLP/gte-Qwen2-7B-instruct", "gte_Qwen2_7B_instruct"),
    ("Linq-AI-Research/Linq-Embed-Mistral", "Linq_Embed_Mistral"),
    ("Salesforce/SFR-Embedding-Mistral", "SFR_Embedding_Mistral"),
    ("zeta-alpha-ai/Zeta-Alpha-E5-Mistral", "Zeta_Alpha_E5_Mistral"),
    ("intfloat/e5-mistral-7b-instruct", "e5_mistral_7b_instruct"),
    ("Salesforce/SFR-Embedding-2_R", "SFR_Embedding_2_R"),
    ("GritLM/GritLM-7B", "GritLM_7B"),
    ("BAAI/bge-en-icl", "bge_en_icl"),
    ("Qwen/Qwen3-Embedding-8B", "Qwen3_Embedding_8B"),
    ("BAAI/bge-large-en-v1.5", "bge_large_en_v1_5"),
    ("mixedbread-ai/mxbai-embed-large-v1", "mxbai_embed_large_v1"),
    ("Snowflake/snowflake-arctic-embed-l", "snowflake_arctic_embed_l"),
    ("Alibaba-NLP/gte-large-en-v1.5", "gte_large_en_v1_5"),
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3_Embedding_0_6B"),
    # ---------------------------------------------------------------------- new models (wave 2) ----------------------------------------------------------------------
    ("nomic-ai/nomic-embed-text-v1.5", "nomic_embed_text_v1_5"),
    ("BAAI/bge-base-en-v1.5", "bge_base_en_v1_5"),
    ("intfloat/e5-base-v2", "e5_base_v2"),
    ("sentence-transformers/all-mpnet-base-v2", "all_mpnet_base_v2"),
    ("jinaai/jina-embeddings-v3", "jina_embeddings_v3"),
    ("Alibaba-NLP/gte-Qwen2-1.5B-instruct", "gte_Qwen2_1_5B_instruct"),
    ("Snowflake/snowflake-arctic-embed-m-v2.0", "snowflake_arctic_embed_m_v2"),
    ("thenlper/gte-large", "gte_large"),
    ("thenlper/gte-base", "gte_base"),
]

DATASET_CONFIGS = {
    "mteb_banking77": dict(
        hf="mteb/banking77", config=None, split="train",
        text_col="text", task="CLS", load_mode="standard",
    ),
    "mteb_ImdbClassification": dict(
        hf="mteb/imdb", config=None, split="test",
        text_col="text", task="CLS", load_mode="standard",
    ),
    "mteb_arguana": dict(
        hf="mteb/arguana", config="corpus", split="corpus",
        text_col="text", task="RET", load_mode="standard",
    ),
    "mteb_scidocs": dict(
        hf="mteb/scidocs", config="corpus", split="corpus",
        text_col="text", task="RET", load_mode="standard",
    ),
    "mteb_ToxicConversationsClassification": dict(
        hf="mteb/toxic_conversations_50k", config=None, split="test",
        text_col="text", task="CLS", load_mode="standard",
    ),
    "mteb_EmotionClassification": dict(
        hf="mteb/emotion", config=None, split="test",
        text_col="text", task="CLS", load_mode="standard",
    ),
    "mteb_FiQA2018": dict(
        hf="mteb/fiqa", config="corpus", split="corpus",
        text_col="text", task="RET", load_mode="standard",
    ),
    "mteb_SciFact": dict(
        hf="mteb/scifact", config="corpus", split="corpus",
        text_col="text", task="RET", load_mode="standard",
    ),
    # ---------------------------------------------------------------------- new datasets (wave 2) ----------------------------------------------------------------------
    "mteb_AmazonCounterfactualClassification": dict(
        hf="mteb/amazon_counterfactual", config="en", split="test",
        text_col="text", task="CLS", load_mode="standard",
    ),
    "mteb_AmazonPolarityClassification": dict(
        hf="mteb/amazon_polarity", config=None, split="test",
        text_col="text", task="CLS", load_mode="standard",
    ),
    "mteb_TweetSentimentExtractionClassification": dict(
        hf="mteb/tweet_sentiment_extraction", config=None, split="test",
        text_col="text", task="CLS", load_mode="standard",
    ),
    "mteb_NFCorpus": dict(
        hf="mteb/nfcorpus", config="corpus", split="corpus",
        text_col="text", task="RET", load_mode="standard",
    ),
    "mteb_QuoraRetrieval": dict(
        hf="mteb/quora", config="corpus", split="corpus",
        text_col="text", task="RET", load_mode="standard",
    ),
    "mteb_TRECCOVID": dict(
        hf="mteb/trec-covid", config="corpus", split="corpus",
        text_col="text", task="RET", load_mode="standard",
    ),
    # ---------------------------------------------------------------------- new datasets (wave 3) ----------------------------------------------------------------------
    "mteb_MTOPIntentClassification": dict(
        hf="mteb/MTOPIntentClassification", config="en", split="test",
        text_col="text", task="CLS", load_mode="standard",
    ),
    "mteb_DBpediaClassification": dict(
        hf="fancyzhx/dbpedia_14", config=None, split="test",
        text_col="content", task="CLS", load_mode="standard",
    ),
    "mteb_TweetTopicSingleClassification": dict(
        hf="mteb/TweetTopicSingleClassification", config=None, split="test_2021",
        text_col="text", task="CLS", load_mode="standard",
    ),
    "mteb_HotpotQA": dict(
        hf="mteb/hotpotqa", config="corpus", split="corpus",
        text_col="text", task="RET", load_mode="standard",
    ),
    "mteb_NQ": dict(
        hf="mteb/nq", config="corpus", split="corpus",
        text_col="text", task="RET", load_mode="standard",
    ),
    "mteb_CQADupstackRetrieval": dict(
        hf="mteb/cqadupstack-android", config="corpus", split="corpus",
        text_col="text", task="RET", load_mode="standard",
    ),
    "mteb_MassiveIntentClassification_en": dict(
        hf="mteb/MassiveIntentClassification", config="en", split="train",
        text_col="text", label_col="label", task="CLS", load_mode="standard",
    ),
    "mteb_Touche2020": dict(
        hf="mteb/touche2020", config="corpus", split="corpus",
        text_col="text", task="RET", load_mode="standard",
    ),
    # ---------------------------------------------------------------------- new datasets (formal30 extension) ----------------------------------------------------------------------
    "mteb_MTOPDomainClassification_en": dict(
        hf="mteb/MTOPDomainClassification", config="en", split="test",
        text_col="text", task="CLS", load_mode="standard",
    ),
    "mteb_MassiveScenarioClassification_en": dict(
        hf="mteb/amazon_massive_scenario", config="en", split="test",
        text_col="text", task="CLS", load_mode="standard",
    ),
    "mteb_AskUbuntuDupQuestions": dict(
        hf="mteb/AskUbuntuDupQuestions", config="corpus", split="test",
        text_col="text", task="RET", load_mode="standard",
    ),
    "mteb_STSBenchmark": dict(
        hf="mteb/stsbenchmark-sts", config=None, split="test",
        task="STS", load_mode="sts_pairs", col_a="sentence1", col_b="sentence2",
    ),
    "mteb_SICK_R": dict(
        hf="mteb/sickr-sts", config=None, split="test",
        task="STS", load_mode="sts_pairs", col_a="sentence1", col_b="sentence2",
    ),
    "mteb_MindSmallReranking": dict(
        hf="mteb/MindSmallReranking", config="corpus", split="test",
        text_col="text", task="RET", load_mode="standard",
    ),
    "mteb_SprintDuplicateQuestions": dict(
        hf="mteb/sprintduplicatequestions-pairclassification", config=None, split="test",
        task="PAIR", load_mode="pair_cls", col_a="sent1", col_b="sent2",
    ),
    "mteb_TwitterURLCorpus": dict(
        hf="mteb/twitterurlcorpus-pairclassification", config=None, split="test",
        task="PAIR", load_mode="pair_cls", col_a="sent1", col_b="sent2",
    ),
}

MTEB_SCORES = {
    "mteb_banking77": {
        "all_MiniLM_L6_v2": 73.79, "e5_large_v2": 86.99,
        "SFR_Embedding_Mistral": 90.45, "Linq_Embed_Mistral": 90.70,
        "Zeta_Alpha_E5_Mistral": 89.85, "bge_multilingual_gemma2": 88.60,
        "gte_Qwen2_7B_instruct": 87.18, "stella_en_1.5B_v5": 89.63,
        "e5_mistral_7b_instruct": 87.08, "SFR_Embedding_2_R": 90.33,
        "GritLM_7B": 86.97, "bge_en_icl": 90.11,
        "Qwen3_Embedding_8B": 92.21,
        "bge_large_en_v1_5": 86.44, "mxbai_embed_large_v1": 87.37,
        "snowflake_arctic_embed_l": 86.77, "gte_large_en_v1_5": 86.35,
        "Qwen3_Embedding_0_6B": 88.76,
        "nomic_embed_text_v1_5": 84.25, "bge_base_en_v1_5": 86.95,
        "e5_base_v2": 83.53, "all_mpnet_base_v2": 81.70,
        "jina_embeddings_v3": 84.08, "gte_Qwen2_1_5B_instruct": 87.31,
        "snowflake_arctic_embed_m_v2": 80.17,
        "gte_large": 86.06, "gte_base": 85.07,
    },
    "mteb_ImdbClassification": {
        "all_MiniLM_L6_v2": 62.37, "e5_large_v2": 95.10,
        "SFR_Embedding_Mistral": 96.16, "Linq_Embed_Mistral": 95.92,
        "Zeta_Alpha_E5_Mistral": 95.62, "bge_multilingual_gemma2": 95.97,
        "gte_Qwen2_7B_instruct": 88.69, "stella_en_1.5B_v5": 95.94,
        "e5_mistral_7b_instruct": 95.19, "SFR_Embedding_2_R": 96.11,
        "GritLM_7B": 93.53, "bge_en_icl": 96.05,
        "Qwen3_Embedding_8B": 96.46,
        "bge_large_en_v1_5": 91.66, "mxbai_embed_large_v1": 94.18,
        "snowflake_arctic_embed_l": 90.53, "gte_large_en_v1_5": 94.00,
        "Qwen3_Embedding_0_6B": 95.00,
        "nomic_embed_text_v1_5": 85.31, "bge_base_en_v1_5": 90.81,
        "e5_base_v2": 86.15, "all_mpnet_base_v2": 71.17,
        "jina_embeddings_v3": 91.90, "gte_Qwen2_1_5B_instruct": 95.83,
        "snowflake_arctic_embed_m_v2": 69.28,
        "gte_large": 88.46, "gte_base": 85.95,
    },
    "mteb_arguana": {
        "all_MiniLM_L6_v2": 36.35, "e5_large_v2": 49.53,
        "SFR_Embedding_Mistral": 64.46, "Linq_Embed_Mistral": 63.44,
        "Zeta_Alpha_E5_Mistral": 65.35, "bge_multilingual_gemma2": 67.26,
        "gte_Qwen2_7B_instruct": 68.74, "stella_en_1.5B_v5": 64.07,
        "e5_mistral_7b_instruct": 64.79, "SFR_Embedding_2_R": 63.22,
        "GritLM_7B": 60.39, "bge_en_icl": 65.36,
        "Qwen3_Embedding_8B": 69.78,
        "bge_large_en_v1_5": 63.55, "mxbai_embed_large_v1": 54.61,
        "snowflake_arctic_embed_l": 55.24, "gte_large_en_v1_5": 57.33,
        "Qwen3_Embedding_0_6B": 63.86,
        "nomic_embed_text_v1_5": 52.02, "bge_base_en_v1_5": 63.75,
        "e5_base_v2": 44.57, "all_mpnet_base_v2": 46.52,
        "jina_embeddings_v3": 43.29, "gte_Qwen2_1_5B_instruct": 69.72,
        "snowflake_arctic_embed_m_v2": 57.88,
        "gte_large": 57.16, "gte_base": 57.12,
    },
    "mteb_scidocs": {
        "all_MiniLM_L6_v2": 15.02, "e5_large_v2": 17.39,
        "SFR_Embedding_Mistral": 20.17, "Linq_Embed_Mistral": 22.44,
        "Zeta_Alpha_E5_Mistral": 21.58, "bge_multilingual_gemma2": 21.09,
        "gte_Qwen2_7B_instruct": 22.13, "stella_en_1.5B_v5": 22.47,
        "e5_mistral_7b_instruct": 19.97, "SFR_Embedding_2_R": 20.17,
        "GritLM_7B": 18.79, "bge_en_icl": 22.22,
        "Qwen3_Embedding_8B": 23.09,
        "bge_large_en_v1_5": 16.68, "mxbai_embed_large_v1": 18.17,
        "snowflake_arctic_embed_l": 18.30, "gte_large_en_v1_5": 17.92,
        "Qwen3_Embedding_0_6B": 19.78,
        "nomic_embed_text_v1_5": 17.63, "bge_base_en_v1_5": 21.73,
        "e5_base_v2": 18.68, "all_mpnet_base_v2": 23.77,
        "jina_embeddings_v3": 19.92, "gte_Qwen2_1_5B_instruct": 24.98,
        "snowflake_arctic_embed_m_v2": 20.32,
        "gte_large": 23.44, "gte_base": 23.13,
    },
    "mteb_ToxicConversationsClassification": {
        "gte_Qwen2_7B_instruct": 85.74, "bge_multilingual_gemma2": 87.34,
        "Linq_Embed_Mistral": 71.29, "SFR_Embedding_Mistral": 69.33,
        "e5_large_v2": 63.29, "all_MiniLM_L6_v2": 62.09,
        "Zeta_Alpha_E5_Mistral": 75.88, "stella_en_1.5B_v5": 88.76,
        "e5_mistral_7b_instruct": 69.59, "SFR_Embedding_2_R": 91.14,
        "GritLM_7B": 70.80, "bge_en_icl": 93.17, "Qwen3_Embedding_8B": 91.65,
        "bge_large_en_v1_5": 70.91, "mxbai_embed_large_v1": 71.48,
        "snowflake_arctic_embed_l": 64.71, "gte_large_en_v1_5": 82.61,
        "Qwen3_Embedding_0_6B": 82.13,
        "nomic_embed_text_v1_5": 67.18, "bge_base_en_v1_5": 67.04,
        "e5_base_v2": 65.87, "all_mpnet_base_v2": 61.05,
        "jina_embeddings_v3": 91.27, "gte_Qwen2_1_5B_instruct": 82.66,
        "snowflake_arctic_embed_m_v2": 64.84,
        "gte_large": 70.56, "gte_base": 71.61,
    },
    "mteb_EmotionClassification": {
        "gte_Qwen2_7B_instruct": 79.46, "bge_multilingual_gemma2": 92.98,
        "Linq_Embed_Mistral": 51.82, "SFR_Embedding_Mistral": 50.24,
        "e5_large_v2": 49.45, "all_MiniLM_L6_v2": 40.83,
        "Zeta_Alpha_E5_Mistral": 57.72, "stella_en_1.5B_v5": 84.30,
        "e5_mistral_7b_instruct": 49.77, "SFR_Embedding_2_R": 93.37,
        "GritLM_7B": 52.81, "bge_en_icl": 93.36, "Qwen3_Embedding_8B": 66.29,
        "bge_large_en_v1_5": 51.52, "mxbai_embed_large_v1": 50.88,
        "snowflake_arctic_embed_l": 46.46, "gte_large_en_v1_5": 46.77,
        "Qwen3_Embedding_0_6B": 63.25,
        "nomic_embed_text_v1_5": 47.99, "bge_base_en_v1_5": 51.90,
        "e5_base_v2": 46.95, "all_mpnet_base_v2": 42.22,
        "jina_embeddings_v3": 73.30, "gte_Qwen2_1_5B_instruct": 61.37,
        "snowflake_arctic_embed_m_v2": 44.25,
        "gte_large": 47.88, "gte_base": 48.65,
    },
    "mteb_FiQA2018": {
        "gte_Qwen2_7B_instruct": 62.03, "bge_multilingual_gemma2": 60.04,
        "Linq_Embed_Mistral": 61.21, "SFR_Embedding_Mistral": 60.40,
        "e5_large_v2": 41.14, "all_MiniLM_L6_v2": 36.87,
        "Zeta_Alpha_E5_Mistral": 58.78, "stella_en_1.5B_v5": 60.48,
        "e5_mistral_7b_instruct": 56.59, "SFR_Embedding_2_R": 61.77,
        "GritLM_7B": 59.95, "bge_en_icl": 59.67, "Qwen3_Embedding_8B": 65.00,
        "bge_large_en_v1_5": 45.02, "mxbai_embed_large_v1": 45.27,
        "snowflake_arctic_embed_l": 44.71, "gte_large_en_v1_5": 63.23,
        "Qwen3_Embedding_0_6B": 47.00,
        "nomic_embed_text_v1_5": 37.46, "bge_base_en_v1_5": 40.65,
        "e5_base_v2": 39.88, "all_mpnet_base_v2": 49.96,
        "jina_embeddings_v3": 47.35, "gte_Qwen2_1_5B_instruct": 54.70,
        "snowflake_arctic_embed_m_v2": 44.17,
        "gte_large": 44.50, "gte_base": 40.76,
    },
    "mteb_SciFact": {
        "gte_Qwen2_7B_instruct": 79.06, "bge_multilingual_gemma2": 72.05,
        "Linq_Embed_Mistral": 78.32, "SFR_Embedding_Mistral": 77.66,
        "e5_large_v2": 72.24, "all_MiniLM_L6_v2": 64.51,
        "Zeta_Alpha_E5_Mistral": 77.38, "stella_en_1.5B_v5": 80.09,
        "e5_mistral_7b_instruct": 76.41, "SFR_Embedding_2_R": 85.91,
        "GritLM_7B": 79.17, "bge_en_icl": 79.09, "Qwen3_Embedding_8B": 78.00,
        "bge_large_en_v1_5": 74.61, "mxbai_embed_large_v1": 74.73,
        "snowflake_arctic_embed_l": 73.82, "gte_large_en_v1_5": 82.43,
        "Qwen3_Embedding_0_6B": 70.00,
        "nomic_embed_text_v1_5": 70.28, "bge_base_en_v1_5": 74.34,
        "e5_base_v2": 71.94, "all_mpnet_base_v2": 65.57,
        "jina_embeddings_v3": 72.53, "gte_Qwen2_1_5B_instruct": 78.44,
        "snowflake_arctic_embed_m_v2": 72.30,
        "gte_large": 74.27, "gte_base": 76.18,
    },
    # ---------------------------------------------------------------------- new datasets (wave 2) ----------------------------------------------------------------------
    "mteb_AmazonCounterfactualClassification": {
        "all_MiniLM_L6_v2": 65.59, "e5_large_v2": 77.78,
        "SFR_Embedding_Mistral": 74.03, "Linq_Embed_Mistral": 87.00,
        "Zeta_Alpha_E5_Mistral": 77.76, "bge_multilingual_gemma2": 89.48,
        "gte_Qwen2_7B_instruct": 93.04, "stella_en_1.5B_v5": None,
        "e5_mistral_7b_instruct": 74.60, "SFR_Embedding_2_R": 98.63,
        "GritLM_7B": 79.61, "bge_en_icl": 93.15,
        "Qwen3_Embedding_8B": 93.94,
        "bge_large_en_v1_5": 75.04, "mxbai_embed_large_v1": 76.48,
        "snowflake_arctic_embed_l": 63.16, "gte_large_en_v1_5": None,
        "Qwen3_Embedding_0_6B": 94.00,
        "nomic_embed_text_v1_5": 74.49, "bge_base_en_v1_5": 74.66,
        "e5_base_v2": 76.15, "all_mpnet_base_v2": 55.66,
        "jina_embeddings_v3": 95.54, "gte_Qwen2_1_5B_instruct": 85.22,
        "snowflake_arctic_embed_m_v2": 66.69,
        "gte_large": 72.35, "gte_base": 73.40,
    },
    "mteb_AmazonPolarityClassification": {
        "all_MiniLM_L6_v2": 64.26, "e5_large_v2": 93.75,
        "SFR_Embedding_Mistral": 95.97, "Linq_Embed_Mistral": 95.70,
        "Zeta_Alpha_E5_Mistral": 96.62, "bge_multilingual_gemma2": 96.90,
        "gte_Qwen2_7B_instruct": 97.50, "stella_en_1.5B_v5": None,
        "e5_mistral_7b_instruct": 96.26, "SFR_Embedding_2_R": 97.31,
        "GritLM_7B": 96.64, "bge_en_icl": 96.98,
        "Qwen3_Embedding_8B": 97.54,
        "bge_large_en_v1_5": 92.42, "mxbai_embed_large_v1": 93.84,
        "snowflake_arctic_embed_l": 78.40, "gte_large_en_v1_5": None,
        "Qwen3_Embedding_0_6B": 96.19,
        "nomic_embed_text_v1_5": 91.81, "bge_base_en_v1_5": 93.39,
        "e5_base_v2": 92.81, "all_mpnet_base_v2": 67.14,
        "jina_embeddings_v3": 95.38, "gte_Qwen2_1_5B_instruct": 96.61,
        "snowflake_arctic_embed_m_v2": 70.38,
        "gte_large": 92.52, "gte_base": 91.77,
    },
    "mteb_TweetSentimentExtractionClassification": {
        "all_MiniLM_L6_v2": 54.04, "e5_large_v2": 60.94,
        "SFR_Embedding_Mistral": 63.64, "Linq_Embed_Mistral": 64.76,
        "Zeta_Alpha_E5_Mistral": 65.22, "bge_multilingual_gemma2": 78.86,
        "gte_Qwen2_7B_instruct": 72.58, "stella_en_1.5B_v5": None,
        "e5_mistral_7b_instruct": 64.89, "SFR_Embedding_2_R": 79.70,
        "GritLM_7B": 66.26, "bge_en_icl": 79.93,
        "Qwen3_Embedding_8B": 78.98,
        "bge_large_en_v1_5": 59.94, "mxbai_embed_large_v1": 59.70,
        "snowflake_arctic_embed_l": 56.74, "gte_large_en_v1_5": None,
        "Qwen3_Embedding_0_6B": 76.05,
        "nomic_embed_text_v1_5": 60.92, "bge_base_en_v1_5": 59.38,
        "e5_base_v2": 60.39, "all_mpnet_base_v2": 55.05,
        "jina_embeddings_v3": 71.40, "gte_Qwen2_1_5B_instruct": 72.95,
        "snowflake_arctic_embed_m_v2": 58.24,
        "gte_large": 56.58, "gte_base": 57.01,
    },
    "mteb_NFCorpus": {
        "all_MiniLM_L6_v2": 31.59, "e5_large_v2": 37.13,
        "SFR_Embedding_Mistral": 41.88, "Linq_Embed_Mistral": 42.02,
        "Zeta_Alpha_E5_Mistral": 40.46, "bge_multilingual_gemma2": 38.11,
        "gte_Qwen2_7B_instruct": 40.60, "stella_en_1.5B_v5": None,
        "e5_mistral_7b_instruct": 38.58, "SFR_Embedding_2_R": 41.34,
        "GritLM_7B": 40.86, "bge_en_icl": 41.85,
        "Qwen3_Embedding_8B": 41.45,
        "bge_large_en_v1_5": 38.06, "mxbai_embed_large_v1": 38.67,
        "snowflake_arctic_embed_l": 37.65, "gte_large_en_v1_5": None,
        "Qwen3_Embedding_0_6B": 36.71,
        "nomic_embed_text_v1_5": 34.67, "bge_base_en_v1_5": 37.37,
        "e5_base_v2": 35.39, "all_mpnet_base_v2": 33.29,
        "jina_embeddings_v3": 36.63, "gte_Qwen2_1_5B_instruct": 39.34,
        "snowflake_arctic_embed_m_v2": 35.87,
        "gte_large": 38.17, "gte_base": 37.90,
    },
    "mteb_QuoraRetrieval": {
        "all_MiniLM_L6_v2": 87.55, "e5_large_v2": 86.84,
        "SFR_Embedding_Mistral": 89.78, "Linq_Embed_Mistral": 90.27,
        "Zeta_Alpha_E5_Mistral": 89.89, "bge_multilingual_gemma2": 90.04,
        "gte_Qwen2_7B_instruct": 90.09, "stella_en_1.5B_v5": None,
        "e5_mistral_7b_instruct": 89.61, "SFR_Embedding_2_R": 89.58,
        "GritLM_7B": None, "bge_en_icl": 90.95,
        "Qwen3_Embedding_8B": 88.90,
        "bge_large_en_v1_5": 89.07, "mxbai_embed_large_v1": 88.85,
        "snowflake_arctic_embed_l": 87.41, "gte_large_en_v1_5": None,
        "Qwen3_Embedding_0_6B": 87.78,
        "nomic_embed_text_v1_5": 88.00, "bge_base_en_v1_5": 88.90,
        "e5_base_v2": 86.56, "all_mpnet_base_v2": 87.45,
        "jina_embeddings_v3": 89.09, "gte_Qwen2_1_5B_instruct": 89.64,
        "snowflake_arctic_embed_m_v2": 89.06,
        "gte_large": 88.32, "gte_base": 88.15,
    },
    "mteb_TRECCOVID": {
        "all_MiniLM_L6_v2": 47.23, "e5_large_v2": 66.64,
        "SFR_Embedding_Mistral": 87.60, "Linq_Embed_Mistral": 87.10,
        "Zeta_Alpha_E5_Mistral": 83.68, "bge_multilingual_gemma2": 64.27,
        "gte_Qwen2_7B_instruct": 80.37, "stella_en_1.5B_v5": None,
        "e5_mistral_7b_instruct": 87.03, "SFR_Embedding_2_R": 88.44,
        "GritLM_7B": 74.31, "bge_en_icl": 79.08,
        "Qwen3_Embedding_8B": 94.99,
        "bge_large_en_v1_5": 74.70, "mxbai_embed_large_v1": 75.53,
        "snowflake_arctic_embed_l": 80.72, "gte_large_en_v1_5": None,
        "Qwen3_Embedding_0_6B": 90.52,
        "nomic_embed_text_v1_5": 63.44, "bge_base_en_v1_5": 78.03,
        "e5_base_v2": 69.63, "all_mpnet_base_v2": 51.33,
        "jina_embeddings_v3": 77.74, "gte_Qwen2_1_5B_instruct": 85.38,
        "snowflake_arctic_embed_m_v2": 80.34,
        "gte_large": 70.22, "gte_base": 68.78,
    },
    # ---------------------------------------------------------------------- new datasets (wave 3) ----------------------------------------------------------------------
    "mteb_MTOPIntentClassification": {
        "all_MiniLM_L6_v2": 61.55, "e5_large_v2": 65.71,
        "stella_en_1.5B_v5": 90.13, "bge_multilingual_gemma2": 95.51,
        "gte_Qwen2_7B_instruct": 91.25, "Linq_Embed_Mistral": 86.70,
        "SFR_Embedding_Mistral": 78.49, "Zeta_Alpha_E5_Mistral": 80.52,
        "e5_mistral_7b_instruct": 78.99, "SFR_Embedding_2_R": 90.62,
        "GritLM_7B": 81.23, "bge_en_icl": 94.00,
        "Qwen3_Embedding_8B": None,
        "bge_large_en_v1_5": 69.92, "mxbai_embed_large_v1": 77.00,
        "snowflake_arctic_embed_l": 56.85, "gte_large_en_v1_5": None,
        "Qwen3_Embedding_0_6B": 85.56,
        "nomic_embed_text_v1_5": 64.92, "bge_base_en_v1_5": 67.62,
        "e5_base_v2": 63.63, "all_mpnet_base_v2": 44.28,
        "jina_embeddings_v3": 78.69, "gte_Qwen2_1_5B_instruct": 87.08,
        "snowflake_arctic_embed_m_v2": 65.96,
        "gte_large": 57.70, "gte_base": 57.88,
    },
    "mteb_DBpediaClassification": {
        "all_MiniLM_L6_v2": 85.44, "e5_large_v2": 89.45,
        "stella_en_1.5B_v5": 91.07, "bge_multilingual_gemma2": None,
        "gte_Qwen2_7B_instruct": 95.46, "Linq_Embed_Mistral": 95.84,
        "SFR_Embedding_Mistral": 91.64, "Zeta_Alpha_E5_Mistral": None,
        "e5_mistral_7b_instruct": 90.86, "SFR_Embedding_2_R": 90.88,
        "GritLM_7B": 92.93, "bge_en_icl": None,
        "Qwen3_Embedding_8B": 99.26,
        "bge_large_en_v1_5": 81.66, "mxbai_embed_large_v1": 85.73,
        "snowflake_arctic_embed_l": 86.61, "gte_large_en_v1_5": None,
        "Qwen3_Embedding_0_6B": 98.84,
        "nomic_embed_text_v1_5": 90.96, "bge_base_en_v1_5": 82.71,
        "e5_base_v2": 89.31, "all_mpnet_base_v2": 89.76,
        "jina_embeddings_v3": 76.05, "gte_Qwen2_1_5B_instruct": 94.62,
        "snowflake_arctic_embed_m_v2": 89.29,
        "gte_large": 82.14, "gte_base": 82.92,
    },
    "mteb_TweetTopicSingleClassification": {
        "all_MiniLM_L6_v2": 67.76, "e5_large_v2": 67.98,
        "stella_en_1.5B_v5": 68.06, "bge_multilingual_gemma2": None,
        "gte_Qwen2_7B_instruct": 73.05, "Linq_Embed_Mistral": 72.24,
        "SFR_Embedding_Mistral": 77.26, "Zeta_Alpha_E5_Mistral": None,
        "e5_mistral_7b_instruct": 76.60, "SFR_Embedding_2_R": 67.51,
        "GritLM_7B": 75.03, "bge_en_icl": None,
        "Qwen3_Embedding_8B": 81.61,
        "bge_large_en_v1_5": 71.90, "mxbai_embed_large_v1": 72.22,
        "snowflake_arctic_embed_l": 68.96, "gte_large_en_v1_5": None,
        "Qwen3_Embedding_0_6B": 80.75,
        "nomic_embed_text_v1_5": 68.35, "bge_base_en_v1_5": 71.75,
        "e5_base_v2": 69.96, "all_mpnet_base_v2": 68.96,
        "jina_embeddings_v3": 57.06, "gte_Qwen2_1_5B_instruct": 76.63,
        "snowflake_arctic_embed_m_v2": 63.85,
        "gte_large": 72.98, "gte_base": 71.97,
    },
    "mteb_HotpotQA": {
        "all_MiniLM_L6_v2": 46.51, "e5_large_v2": 73.13,
        "stella_en_1.5B_v5": 76.67, "bge_multilingual_gemma2": 83.26,
        "gte_Qwen2_7B_instruct": 73.08, "Linq_Embed_Mistral": 76.24,
        "SFR_Embedding_Mistral": 77.02, "Zeta_Alpha_E5_Mistral": 79.45,
        "e5_mistral_7b_instruct": 75.72, "SFR_Embedding_2_R": 81.36,
        "GritLM_7B": None, "bge_en_icl": 85.14,
        "Qwen3_Embedding_8B": 76.78,
        "bge_large_en_v1_5": 74.10, "mxbai_embed_large_v1": 72.04,
        "snowflake_arctic_embed_l": 75.18, "gte_large_en_v1_5": None,
        "Qwen3_Embedding_0_6B": 65.74,
        "nomic_embed_text_v1_5": 72.62, "bge_base_en_v1_5": 72.60,
        "e5_base_v2": 69.15, "all_mpnet_base_v2": 39.29,
        "jina_embeddings_v3": 64.67, "gte_Qwen2_1_5B_instruct": 68.95,
        "snowflake_arctic_embed_m_v2": 72.42,
        "gte_large": 67.16, "gte_base": 65.75,
    },
    "mteb_NQ": {
        "all_MiniLM_L6_v2": 43.87, "e5_large_v2": 63.44,
        "stella_en_1.5B_v5": 71.80, "bge_multilingual_gemma2": 71.45,
        "gte_Qwen2_7B_instruct": 67.00, "Linq_Embed_Mistral": 70.63,
        "SFR_Embedding_Mistral": 69.92, "Zeta_Alpha_E5_Mistral": 70.52,
        "e5_mistral_7b_instruct": 63.53, "SFR_Embedding_2_R": 73.96,
        "GritLM_7B": None, "bge_en_icl": 73.88,
        "Qwen3_Embedding_8B": 65.25,
        "bge_large_en_v1_5": 55.03, "mxbai_embed_large_v1": 55.80,
        "snowflake_arctic_embed_l": 63.11, "gte_large_en_v1_5": None,
        "Qwen3_Embedding_0_6B": 53.46,
        "nomic_embed_text_v1_5": 59.72, "bge_base_en_v1_5": 54.15,
        "e5_base_v2": 58.22, "all_mpnet_base_v2": 50.45,
        "jina_embeddings_v3": 64.23, "gte_Qwen2_1_5B_instruct": 64.00,
        "snowflake_arctic_embed_m_v2": 64.65,
        "gte_large": 54.78, "gte_base": 52.84,
    },
    "mteb_CQADupstackRetrieval": {
        "all_MiniLM_L6_v2": 41.32, "e5_large_v2": 37.66,
        "stella_en_1.5B_v5": 32.54, "bge_multilingual_gemma2": 47.94,
        "gte_Qwen2_7B_instruct": 42.66, "Linq_Embed_Mistral": 47.51,
        "SFR_Embedding_Mistral": 48.03, "Zeta_Alpha_E5_Mistral": 48.85,
        "e5_mistral_7b_instruct": 46.00, "SFR_Embedding_2_R": 44.70,
        "GritLM_7B": 46.39, "bge_en_icl": 47.31,
        "Qwen3_Embedding_8B": 52.94,
        "bge_large_en_v1_5": 42.23, "mxbai_embed_large_v1": 41.60,
        "snowflake_arctic_embed_l": 46.97, "gte_large_en_v1_5": None,
        "Qwen3_Embedding_0_6B": 46.03,
        "nomic_embed_text_v1_5": 37.76, "bge_base_en_v1_5": 41.59,
        "e5_base_v2": 38.54, "all_mpnet_base_v2": 44.96,
        "jina_embeddings_v3": 42.59, "gte_Qwen2_1_5B_instruct": 44.76,
        "snowflake_arctic_embed_m_v2": 47.20,
        "gte_large": 43.18, "gte_base": 42.91,
    },
    "mteb_MassiveIntentClassification_en": {
        "bge_multilingual_gemma2": 82.05, "gte_Qwen2_7B_instruct": 85.43,
        "Linq_Embed_Mistral": 76.42, "SFR_Embedding_Mistral": 75.88,
        "Zeta_Alpha_E5_Mistral": 77.31, "all_MiniLM_L6_v2": 66.94,
        "stella_en_1.5B_v5": 84.51, "e5_large_v2": 68.14,
        "e5_mistral_7b_instruct": 71.14, "SFR_Embedding_2_R": 85.97,
        "GritLM_7B": 80.78, "bge_en_icl": 82.93, "Qwen3_Embedding_8B": 75.57,
        "bge_large_en_v1_5": 77.56, "mxbai_embed_large_v1": 76.24,
        "snowflake_arctic_embed_l": 65.79, "gte_large_en_v1_5": 78.94,
        "Qwen3_Embedding_0_6B": 73.76,
    },
}


# ----------------------------------------------------------------------
# Model loading  (same as gfmi_all_layers.py)
# ----------------------------------------------------------------------

from pooling_utils import (
    cls_pool, mean_pooling, last_token_pool, get_pooling_strategy,
)

def _load_model(model_name, device, force_bidir=False):
    from bidir_models import is_llm2vec_model, _load_llm2vec_as_bidir, \
        _patch_nemotron_py39, _get_nested_config_value, \
        is_nvembed_model, _load_nvembed
    if is_llm2vec_model(model_name):
        return _load_llm2vec_as_bidir(model_name, device)
    if is_nvembed_model(model_name):
        return _load_nvembed(model_name, device)
    if "nemotron" in model_name.lower():
        _patch_nemotron_py39()

    from transformers import AutoModel, AutoTokenizer, AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    n_layers = _get_nested_config_value(config, "num_hidden_layers")
    hidden_size = _get_nested_config_value(config, "hidden_size")
    if n_layers is None:
        raise ValueError(f"Cannot determine num_hidden_layers for {model_name}")
    arch = getattr(config, "model_type", "unknown")

    _DECODER_TYPES = (
        "qwen2", "qwen3", "mistral", "llama", "gemma", "gemma2",
        "gpt2", "falcon", "phi", "phi3", "phimoe", "starcoder2",
    )
    is_decoder = arch in _DECODER_TYPES and not force_bidir

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {"trust_remote_code": True,
                   "torch_dtype": torch.float16 if (arch in _DECODER_TYPES) else torch.float32}
    if arch in _DECODER_TYPES:
        load_kwargs["attn_implementation"] = "eager"
    if getattr(config, "use_memory_efficient_attention", False) or getattr(config, "unpad_inputs", False):
        config.use_memory_efficient_attention = False
        config.unpad_inputs = False
        load_kwargs["config"] = config
        load_kwargs["attn_implementation"] = "eager"
    model = AutoModel.from_pretrained(model_name, **load_kwargs).to(device)
    try:
        model.resize_token_embeddings(len(tokenizer))
    except NotImplementedError:
        pass
    model.eval()
    info = dict(n_layers=n_layers, hidden_size=hidden_size,
                arch=arch, is_decoder=is_decoder)
    return model, tokenizer, info


def _run_one_pass(model, tokenizer, texts, target_layers: Set[int],
                  batch_size, max_length, device, pooling, normalize, is_decoder):
    layer_chunks: Dict[int, List[np.ndarray]] = {i: [] for i in target_layers}

    # Hook-based fallback for models that don't support output_hidden_states
    use_hooks = False
    hooked_states: Dict[int, torch.Tensor] = {}
    hooks = []

    # Test if model supports output_hidden_states
    try:
        test_enc = tokenizer(["test"], max_length=32, padding=True,
                             truncation=True, return_tensors="pt")
        test_enc = {k: v.to(device) for k, v in test_enc.items()}
        test_fwd = dict(output_hidden_states=True)
        if is_decoder:
            test_fwd["use_cache"] = False
        with torch.inference_mode():
            test_out = model(**test_enc, **test_fwd)
        if not hasattr(test_out, "hidden_states") or test_out.hidden_states is None:
            use_hooks = True
        del test_out, test_enc
    except TypeError:
        use_hooks = True

    if use_hooks:
        print("  [INFO] Using hook-based hidden state extraction")
        embeddings_module = None
        for name, mod in model.named_modules():
            cname = mod.__class__.__name__.lower()
            if ("embed" in name and not any(x in name for x in ["word", "position", "token_type", "LayerNorm", "dropout", "self_attn", "mlp"])) \
               or (name == "embed_tokens"):
                embeddings_module = mod
                break
        encoder_layers = []
        import re
        _layer_re = re.compile(r'^(?:.*\.)?(layers?|h|block)\.\d+$')
        for name, mod in model.named_modules():
            if _layer_re.match(name) and hasattr(mod, "forward"):
                encoder_layers.append((name, mod))

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hooked_states[layer_idx] = output[0].detach()
                else:
                    hooked_states[layer_idx] = output.detach()
            return hook_fn

        if embeddings_module:
            hooks.append(embeddings_module.register_forward_hook(make_hook(0)))
        for idx, (name, mod) in enumerate(encoder_layers):
            hooks.append(mod.register_forward_hook(make_hook(idx + 1)))

    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size),
                          desc=f"  fwd layers {min(target_layers)}-{max(target_layers)}",
                          leave=False):
            batch = [t if t.strip() else " " for t in texts[start:start + batch_size]]
            if is_decoder:
                tokenizer.padding_side = "left"
            enc = tokenizer(batch, max_length=max_length, padding=True,
                            truncation=True, return_tensors="pt")
            if enc["input_ids"].numel() == 0:
                continue
            enc = {k: v.to(device) for k, v in enc.items()}
            fwd_kwargs = {}
            if not use_hooks:
                fwd_kwargs["output_hidden_states"] = True
            if is_decoder:
                fwd_kwargs["use_cache"] = False
            hooked_states.clear()
            outputs = model(**enc, **fwd_kwargs)
            mask = enc["attention_mask"]
            for i in target_layers:
                if use_hooks:
                    if i not in hooked_states:
                        continue
                    hs = hooked_states[i].float()
                else:
                    hs = outputs.hidden_states[i].float()
                # Handle dimension mismatch between hs and mask
                if hs.shape[1] != mask.shape[1]:
                    pooled = hs.mean(dim=1)
                elif pooling == "last_token":
                    pooled = last_token_pool(hs, mask)
                elif pooling == "cls":
                    pooled = cls_pool(hs, mask)
                else:
                    pooled = mean_pooling(hs, mask)
                if normalize:
                    pooled = F.normalize(pooled, p=2, dim=1)
                layer_chunks[i].append(pooled.cpu().numpy())
            del outputs, enc
            hooked_states.clear()
            if start % (batch_size * 20) == 0:
                gc.collect(); torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    result = {}
    for i in target_layers:
        if layer_chunks[i]:
            result[i] = np.vstack(layer_chunks[i]).astype(np.float32)
    return result


def load_texts(ds_key: str, max_n: int) -> List[str]:
    from datasets import load_dataset
    cfg = DATASET_CONFIGS[ds_key]
    mode = cfg.get("load_mode", "standard")
    print(f"Loading dataset: {cfg['hf']}  config={cfg.get('config')}  "
          f"split={cfg['split']}  mode={mode}")
    try:
        if cfg.get("config"):
            ds = load_dataset(cfg["hf"], cfg["config"], split=cfg["split"])
        else:
            ds = load_dataset(cfg["hf"], split=cfg["split"])
    except ValueError as e:
        # Some MMTEB pair-classification datasets store a single gzipped JSON
        # file with list-valued columns that older `datasets` versions can't parse.
        if mode != "pair_cls" or "Feature type 'List' not found" not in str(e):
            raise
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=cfg["hf"],
            repo_type="dataset",
            filename=f"{cfg['split']}.json.gz",
        )
        with gzip.open(path, "rt", encoding="utf-8") as f:
            ds = json.load(f)

    if mode == "standard":
        texts = [str(x) for x in ds[cfg["text_col"]]]
    elif mode == "sts_pairs":
        col_a, col_b = cfg["col_a"], cfg["col_b"]
        seen = set()
        texts = []
        for row in ds:
            for s in (str(row[col_a]), str(row[col_b])):
                if s not in seen:
                    seen.add(s)
                    texts.append(s)
    elif mode == "pair_cls":
        col_a, col_b = cfg["col_a"], cfg["col_b"]
        seen = set()
        texts = []
        if isinstance(ds, dict):
            a_seq = ds[col_a]
            b_seq = ds[col_b]
            for s in list(a_seq) + list(b_seq):
                s = str(s)
                if s not in seen:
                    seen.add(s)
                    texts.append(s)
        else:
            for row in ds:
                a_list = row[col_a] if isinstance(row[col_a], list) else [row[col_a]]
                b_list = row[col_b] if isinstance(row[col_b], list) else [row[col_b]]
                for s in a_list + b_list:
                    s = str(s)
                    if s not in seen:
                        seen.add(s)
                        texts.append(s)
    else:
        raise ValueError(f"Unknown load_mode: {mode}")

    if max_n is not None and len(texts) > max_n:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(texts), max_n, replace=False)
        idx.sort()
        texts = [texts[i] for i in idx]
    print(f"  {len(texts)} unique texts loaded (max_n={max_n})")
    return texts


# ----------------------------------------------------------------------
# Grassmann manifold geometry
# ----------------------------------------------------------------------

def _center_and_svd(Z: np.ndarray):
    """Center Z and return full SVD components."""
    Z_c = Z - Z.mean(axis=0, keepdims=True)
    bad = ~np.isfinite(Z_c).all(axis=1)
    if bad.any():
        Z_c[bad] = np.random.RandomState(0).randn(int(bad.sum()), Z_c.shape[1]).astype(Z_c.dtype) * 1e-6
    _, S, Vt = np.linalg.svd(Z_c, full_matrices=False)
    return S, Vt


def determine_rank(Z: np.ndarray, variance_threshold: float = 0.9) -> int:
    """
    Determine the rank r such that the top-r singular values
    explain at least `variance_threshold` fraction of total variance.
    """
    S, _ = _center_and_svd(Z)
    var = S ** 2
    cumvar = np.cumsum(var) / var.sum()
    r = int(np.searchsorted(cumvar, variance_threshold) + 1)
    r = max(r, 2)  # need at least 2 for meaningful Grassmann geometry
    return r


def embedding_to_subspace(Z: np.ndarray, r: int) -> np.ndarray:
    """
    Z: (N, d) embedding matrix
    Returns: (d, r) orthonormal basis for principal r-dimensional subspace
    """
    S, Vt = _center_and_svd(Z)
    r = min(r, len(S), Vt.shape[1])
    return Vt[:r].T  # (d, r)


def grassmann_distance(Q1: np.ndarray, Q2: np.ndarray) -> float:
    """
    Geodesic distance on Grassmann manifold.
    Q1, Q2: (d, r) orthonormal bases.
    dist = sqrt(sum(theta_i^2)) where theta_i are principal angles.
    """
    M = Q1.T @ Q2  # (r, r)
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    return float(np.sqrt((angles ** 2).sum()))


def grassmann_distance_matrix(subspaces: Dict[int, np.ndarray]) -> Dict:
    """Compute pairwise geodesic distances for adjacent layers."""
    layers = sorted(subspaces.keys())
    dists = {}
    for i in range(len(layers) - 1):
        l1, l2 = layers[i], layers[i + 1]
        dists[(l1, l2)] = grassmann_distance(subspaces[l1], subspaces[l2])
    return dists


# ----------------------------------------------------------------------
# Discrete Frenet quantities
# ----------------------------------------------------------------------

def menger_curvature(d01: float, d12: float, d02: float) -> float:
    """
    Menger curvature for three points on a metric space.
    kappa = 4 * Area / (d01 * d12 * d02)
    Area via Heron's formula.
    """
    s = (d01 + d12 + d02) / 2.0
    sq = s * (s - d01) * (s - d12) * (s - d02)
    if sq <= 0 or d01 * d12 * d02 < 1e-15:
        return 0.0
    area = math.sqrt(sq)
    return 4.0 * area / (d01 * d12 * d02)


def compute_frenet_profile(subspaces: Dict[int, np.ndarray]):
    """
    Given subspaces indexed by layer, compute:
      - speed[l]:     dist(l, l+1)
      - curvature[l]: Menger curvature at l (needs l-1, l, l+1)
    """
    layers = sorted(subspaces.keys())
    n = len(layers)

    # Pairwise adjacent distances
    adj_dist = {}
    for i in range(n - 1):
        adj_dist[i] = grassmann_distance(subspaces[layers[i]], subspaces[layers[i + 1]])

    # Skip distances (for curvature: need dist(l-1, l+1))
    skip_dist = {}
    for i in range(n - 2):
        skip_dist[i] = grassmann_distance(subspaces[layers[i]], subspaces[layers[i + 2]])

    # Speed: s[i] = dist(layer_i, layer_{i+1})
    speed = [adj_dist[i] for i in range(n - 1)]

    # Curvature: Menger curvature at interior points
    curvature = []
    for i in range(n - 2):
        d01 = adj_dist[i]
        d12 = adj_dist[i + 1]
        d02 = skip_dist[i]
        curvature.append(menger_curvature(d01, d12, d02))

    # Summary metrics
    speed_arr = np.array(speed)
    curv_arr = np.array(curvature)

    total_arc = float(speed_arr.sum())
    chord = grassmann_distance(subspaces[layers[0]], subspaces[layers[-1]])
    straightness = chord / total_arc if total_arc > 1e-15 else 1.0

    # Normalized by n_layers for cross-model comparability
    n_layers = len(layers) - 1

    result = dict(
        layers=[int(l) for l in layers],
        n_layers=n_layers,
        speed=speed_arr.tolist(),
        curvature=curv_arr.tolist(),
        # Summary metrics
        total_arc_length=total_arc,
        chord_distance=chord,
        straightness=straightness,
        mean_speed=float(speed_arr.mean()),
        std_speed=float(speed_arr.std()),
        max_speed=float(speed_arr.max()),
        speed_cv=float(speed_arr.std() / speed_arr.mean()) if speed_arr.mean() > 0 else 0,
        mean_curvature=float(curv_arr.mean()),
        std_curvature=float(curv_arr.std()),
        max_curvature=float(curv_arr.max()),
        # Normalized (per-layer)
        norm_arc_length=total_arc / n_layers,
        norm_mean_curvature=float(curv_arr.mean()) * n_layers,
        # Profile shape: where is the max curvature?
        peak_curvature_depth=float(np.argmax(curv_arr) / len(curv_arr)) if len(curv_arr) > 0 else 0.5,
        # Speed variation: early vs late
        early_speed=float(speed_arr[:n_layers // 3].mean()) if n_layers >= 3 else float(speed_arr.mean()),
        late_speed=float(speed_arr[-n_layers // 3:].mean()) if n_layers >= 3 else float(speed_arr.mean()),
    )
    if result['late_speed'] > 0:
        result['speed_early_late_ratio'] = result['early_speed'] / result['late_speed']
    else:
        result['speed_early_late_ratio'] = 1.0

    return result


# ----------------------------------------------------------------------
# Main pipeline for one (dataset, model)
# ----------------------------------------------------------------------

def compute_frenet_for_model(
    texts: List[str],
    hf_model: str,
    device: str = "cuda",
    batch_size_small: int = 64,
    batch_size_large: int = 4,
    max_length: int = 512,
    subspace_rank: int = SUBSPACE_RANK,
    variance_threshold: float = VARIANCE_THRESHOLD,
    layers_per_pass: int = LAYERS_PER_PASS,
) -> dict:
    force_bidir = globals().get("_FORCE_BIDIR", False)
    model, tokenizer, info = _load_model(hf_model, device, force_bidir=force_bidir)
    n_layers = info["n_layers"]
    is_decoder = info["is_decoder"]
    pooling = get_pooling_strategy(hf_model, is_decoder)
    batch_size = batch_size_large if is_decoder else batch_size_small
    if info["hidden_size"] >= 4096 and not is_decoder:
        batch_size = min(batch_size, batch_size_large)

    print(f"  arch={info['arch']}  layers={n_layers}  "
          f"hidden={info['hidden_size']}  pooling={pooling}  bs={batch_size}")

    # All layers: 0 (embedding) through n_layers (last hidden)
    all_layer_ids = list(range(n_layers + 1))
    groups = [all_layer_ids[i:i + layers_per_pass]
              for i in range(0, len(all_layer_ids), layers_per_pass)]

    # ---------------------------------------------------------------------- Determine subspace rank ----------------------------------------------------------------------
    if subspace_rank is not None:
        r = min(subspace_rank, info["hidden_size"])
        print(f"  Using fixed subspace rank r={r}")
    else:
        # Adaptive: extract last layer first, determine r from 90% variance
        print(f"  Determining adaptive r from last layer (variance_threshold={variance_threshold}) ...")
        probe_data = _run_one_pass(
            model, tokenizer, texts, {n_layers},
            batch_size, max_length, device, pooling, True, is_decoder,
        )
        Z_probe = probe_data[n_layers]
        r = determine_rank(Z_probe, variance_threshold)
        del probe_data, Z_probe
        gc.collect(); torch.cuda.empty_cache()
        print(f"  Adaptive subspace rank: r={r} (retains {variance_threshold*100:.0f}% variance)")

    subspaces = {}

    print(f"  Extracting {len(all_layer_ids)} layers in {len(groups)} pass(es), r={r}")
    for gi, group in enumerate(groups):
        t0 = time.time()
        layer_data = _run_one_pass(
            model, tokenizer, texts, set(group),
            batch_size, max_length, device, pooling, True, is_decoder,
        )
        for li in group:
            Z = layer_data[li]
            subspaces[li] = embedding_to_subspace(Z, r)
            del Z
        del layer_data
        gc.collect(); torch.cuda.empty_cache()
        elapsed = time.time() - t0
        print(f"    Pass {gi + 1}/{len(groups)} done in {elapsed:.0f}s", flush=True)

    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

    print(f"  Computing Frenet profile on Grassmann manifold ...")
    frenet = compute_frenet_profile(subspaces)
    frenet["hf_model"] = hf_model
    frenet["hidden_size"] = info["hidden_size"]
    frenet["arch"] = info["arch"]
    frenet["subspace_rank"] = r
    frenet["variance_threshold"] = variance_threshold
    frenet["n_samples"] = len(texts)

    return frenet


# ----------------------------------------------------------------------
# Correlation helpers
# ----------------------------------------------------------------------

def residualize(x, z):
    z = np.asarray(z, dtype=float)
    x = np.asarray(x, dtype=float)
    zc = z - z.mean()
    d = (zc ** 2).sum()
    if d < 1e-12:
        return x - x.mean()
    b = (x * zc).sum() / d
    return x - b * z - (x.mean() - b * z.mean())

def sig(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""


def analyse_correlations(all_results: dict, ds_key: str):
    mteb = MTEB_SCORES.get(ds_key, {})
    if not mteb:
        return

    models_ordered, mteb_vals, dims = [], [], []
    metrics_dict = {}
    metric_names = [
        "total_arc_length", "chord_distance", "straightness",
        "mean_speed", "std_speed", "max_speed", "speed_cv",
        "mean_curvature", "std_curvature", "max_curvature",
        "norm_arc_length", "norm_mean_curvature",
        "peak_curvature_depth", "speed_early_late_ratio",
    ]
    for mn in metric_names:
        metrics_dict[mn] = []

    for short_name, data in all_results.items():
        score = mteb.get(short_name)
        if score is None:
            continue
        models_ordered.append(short_name)
        mteb_vals.append(score)
        dims.append(data["hidden_size"])
        for mn in metric_names:
            metrics_dict[mn].append(data.get(mn, 0.0))

    if len(models_ordered) < 4:
        print(f"  Too few models ({len(models_ordered)})")
        return

    mteb_arr = np.array(mteb_vals)
    log_dims = np.log(np.array(dims, dtype=float))

    print(f"\n  Correlation with MTEB ({ds_key}, n={len(models_ordered)}):")
    print(f"  {'Metric':>28s}  {'Spearman':>10s} {'p':>7s}   "
          f"{'Pearson':>10s} {'p':>7s}   {'Partial_r':>10s} {'p':>7s}")

    for mn in metric_names:
        vals = np.array(metrics_dict[mn])
        if np.std(vals) < 1e-12:
            continue
        rs, ps = spearmanr(vals, mteb_arr)
        rp, pp = pearsonr(vals, mteb_arr)
        vr = residualize(vals, log_dims)
        mr = residualize(mteb_arr, log_dims)
        pr, prp = pearsonr(vr, mr)
        print(f"  {mn:>28s}  {rs:+.4f}  {ps:.4f}{sig(ps):3s}  "
              f"{rp:+.4f}  {pp:.4f}{sig(pp):3s}  "
              f"{pr:+.4f}  {prp:.4f}{sig(prp):3s}")

    rs, ps = spearmanr(log_dims, mteb_arr)
    rp, pp = pearsonr(log_dims, mteb_arr)
    print(f"  {'log(dim)':>28s}  {rs:+.4f}  {ps:.4f}{sig(ps):3s}  "
          f"{rp:+.4f}  {pp:.4f}{sig(pp):3s}  {'---':>10s}")


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_profiles(all_results: dict, ds_key: str, save_dir: str):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_results)))

    for idx, (name, data) in enumerate(all_results.items()):
        n = data["n_layers"]
        rel_speed = np.linspace(0, 1, len(data["speed"]))
        rel_curv = np.linspace(0, 1, len(data["curvature"]))
        c = colors[idx]
        label = f"{name} ({data.get('hidden_size','?')}d)"

        axes[0].plot(rel_speed, data["speed"], color=c, marker=".", ms=2, lw=1.2, label=label)
        axes[1].plot(rel_curv, data["curvature"], color=c, marker=".", ms=2, lw=1.2, label=label)

    for ax, title, ylabel in zip(axes,
            ["Speed (Grassmann dist)", "Menger Curvature"],
            ["geodesic dist", "kappa"]):
        ax.set_xlabel("Relative layer depth")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - {ds_key}")
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=6, loc="best")
    plt.tight_layout()
    path = os.path.join(save_dir, f"frenet_profile_{ds_key}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="banking77,arguana",
                        help="Comma-separated dataset key fragments")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=None,
                        help="Subspace rank r. 0 or omitted = adaptive via variance threshold")
    parser.add_argument("--variance", type=float, default=VARIANCE_THRESHOLD,
                        help="Variance threshold for adaptive rank (default: %(default)s)")
    parser.add_argument("--max_n", type=int, default=MAX_N,
                        help="Optional cap on per-task texts (default: use all).")
    parser.add_argument("--layers_per_pass", type=int, default=LAYERS_PER_PASS)
    parser.add_argument("--output_dir", default="./outputs/frenet_profile_v2")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model short-name fragments")
    parser.add_argument(
        "--model_source", type=str, default="embed",
        choices=["embed", "llm", "bidir"],
        help="embed = embedding models; llm = pure LLM; bidir = bidirectional LLM embeddings",
    )
    args = parser.parse_args()

    # rank=0 or None means adaptive
    subspace_rank = args.rank if args.rank and args.rank > 0 else None
    variance_threshold = args.variance

    os.makedirs(args.output_dir, exist_ok=True)

    global _FORCE_BIDIR
    _FORCE_BIDIR = (args.model_source == "bidir")

    if args.model_source == "llm":
        from llm_models import LLM_MODELS, llm_benchmark_scores_for
        _model_pool = LLM_MODELS
        global MTEB_SCORES
        MTEB_SCORES = llm_benchmark_scores_for(DATASET_CONFIGS.keys())
    elif args.model_source == "bidir":
        from bidir_models import BIDIR_MODELS
        _model_pool = BIDIR_MODELS
        MTEB_SCORES = {}
    else:
        _model_pool = MODELS

    tags = [t.strip() for t in args.ds.split(",") if t.strip()]
    ds_list = [d for d in sorted(DATASET_CONFIGS.keys())
               if any(tag in d for tag in tags)]

    model_list = _model_pool
    if args.models:
        mtags = [t.strip() for t in args.models.split(",") if t.strip()]
        model_list = [(hf, sn) for hf, sn in _model_pool
                      if any(mt in sn for mt in mtags)]

    print(f"Frenet primitives on Grassmann manifold (adaptive r, speed + curvature)")
    print(f"  Datasets: {ds_list}")
    print(f"  Models: {[sn for _, sn in model_list]}")
    print(f"  Subspace rank: {'adaptive (' + str(variance_threshold*100) + '% variance)' if subspace_rank is None else subspace_rank}")

    for ds_key in ds_list:
        print(f"\n{'='*70}")
        print(f"# Dataset: {ds_key}  [{DATASET_CONFIGS[ds_key]['task']}]")
        print(f"{'='*70}")

        ds_dir = os.path.join(args.output_dir, ds_key)
        os.makedirs(ds_dir, exist_ok=True)
        texts = load_texts(ds_key, args.max_n)

        for hf_model, short_name in model_list:
            per_model_path = os.path.join(ds_dir, f"{short_name}.json")
            fail_marker = per_model_path + ".failed"
            if os.path.exists(per_model_path):
                print(f"\n  [SKIP] {short_name} (already computed)")
                continue
            if os.path.exists(fail_marker):
                print(f"\n  [SKIP] {short_name} (previously failed)")
                continue

            print(f"\n  Model: {short_name}  ({hf_model})")
            try:
                result = compute_frenet_for_model(
                    texts, hf_model, args.device,
                    subspace_rank=subspace_rank,
                    variance_threshold=variance_threshold,
                    layers_per_pass=args.layers_per_pass,
                )
                with open(per_model_path, "w") as wf:
                    json.dump(result, wf, indent=2)
                print(f"  Saved: {per_model_path}")

                print(f"    arc_len={result['total_arc_length']:.4f}  "
                      f"chord={result['chord_distance']:.4f}  "
                      f"straight={result['straightness']:.4f}  "
                      f"mean_kappa={result['mean_curvature']:.4f}  "
                      f"mean_speed={result['mean_speed']:.4f}")

            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                import traceback; traceback.print_exc()
                fail_marker = per_model_path + ".failed"
                with open(fail_marker, "w") as fm:
                    fm.write(str(e))
                if "device-side assert" in str(e) or "CUDA" in type(e).__name__:
                    print("  CUDA context corrupted, exiting for restart...", flush=True)
                    sys.exit(42)
                try:
                    gc.collect(); torch.cuda.empty_cache()
                except Exception:
                    print("  CUDA cleanup failed, exiting for restart...", flush=True)
                    sys.exit(42)
                continue

        # Merge per-model results (scan directory for all .json files)
        all_results = {}
        for fname in sorted(os.listdir(ds_dir)):
            if fname.endswith(".json") and fname != "all_results.json":
                sn = fname[:-5]
                with open(os.path.join(ds_dir, fname)) as rf:
                    all_results[sn] = json.load(rf)
        if all_results:
            with open(os.path.join(ds_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f, indent=2)
            plot_profiles(all_results, ds_key, ds_dir)
            analyse_correlations(all_results, ds_key)

    print("\nDone.")


if __name__ == "__main__":
    main()
