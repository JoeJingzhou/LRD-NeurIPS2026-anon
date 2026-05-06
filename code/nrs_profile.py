#!/usr/bin/env python3
"""
Neighborhood Rank Stability (NRS) Profile.

For each (dataset, model), extract embeddings from ALL hidden layers,
then measure how local neighborhoods (k-NN) reorganize between
consecutive layers. This captures *local* geometric stability,
complementary to Frenet's *global* subspace geometry.

Metrics per layer transition (l -> l+1):
  - Jaccard overlap of k-NN sets
  - Spearman rank correlation of pairwise distances within neighborhood

Summary metrics across profile:
  - mean/std/min Jaccard, early vs late Jaccard
  - convergence rate, AUC of stability profile
"""

import argparse
import gc
import gzip
import json
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
MAX_N = None  # None = use all available texts per task
LAYERS_PER_PASS = 8
K_NEIGHBORS = 20
N_ANCHOR = 500          # subsample anchors for speed

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
    # ---------------------------------------------------------------------- formal30 extension (sync with frenet_curvature) ----------------------------------------------------------------------
    "mteb_MTOPDomainClassification_en": dict(
        hf="mteb/MTOPDomainClassification", config="en", split="test",
        text_col="text", label_col="label", task="CLS", load_mode="standard",
    ),
    "mteb_MassiveScenarioClassification_en": dict(
        hf="mteb/amazon_massive_scenario", config="en", split="test",
        text_col="text", label_col="label", task="CLS", load_mode="standard",
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
# Model loading (reused from frenet_curvature.py)
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

def _run_one_pass(model, tokenizer, texts, target_layers,
                  batch_size, max_length, device, pooling, normalize, is_decoder):
    layer_chunks = {i: [] for i in target_layers}

    use_hooks = False
    hooked_states = {}
    hooks = []

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

def load_texts(ds_key, max_n):
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
# NRS computation
# ----------------------------------------------------------------------

def compute_knn(Z, k, anchor_idx=None):
    """
    Compute k-NN for anchor points.
    Z: (N, d), anchor_idx: indices of anchor points (subset).
    Returns: (n_anchors, k) indices of nearest neighbors.
    """
    from sklearn.neighbors import NearestNeighbors
    bad = ~np.isfinite(Z).all(axis=1)
    if bad.any():
        Z = Z.copy()
        Z[bad] = 0.0
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        Z = Z / norms
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
    nn.fit(Z)
    if anchor_idx is not None:
        dists, indices = nn.kneighbors(Z[anchor_idx])
    else:
        dists, indices = nn.kneighbors(Z)
    return indices[:, 1:], dists[:, 1:]   # exclude self


def jaccard_overlap(knn_a, knn_b):
    """
    knn_a, knn_b: (n_anchors, k) - kNN index arrays.
    Returns: (n_anchors,) Jaccard overlap per anchor.
    """
    n = knn_a.shape[0]
    jaccards = np.zeros(n)
    for i in range(n):
        sa = set(knn_a[i])
        sb = set(knn_b[i])
        inter = len(sa & sb)
        union = len(sa | sb)
        jaccards[i] = inter / union if union > 0 else 0.0
    return jaccards


def rank_correlation_per_anchor(dists_a, dists_b):
    """
    Spearman rank correlation of distance orderings.
    dists_a, dists_b: (n_anchors, k).
    Returns: (n_anchors,) correlation per anchor.
    """
    n = dists_a.shape[0]
    corrs = np.zeros(n)
    for i in range(n):
        if np.std(dists_a[i]) < 1e-12 or np.std(dists_b[i]) < 1e-12:
            corrs[i] = 0.0
        else:
            corrs[i], _ = spearmanr(dists_a[i], dists_b[i])
    return corrs


def compute_nrs_profile(embeddings: Dict[int, np.ndarray], k: int, n_anchor: int):
    """
    embeddings: {layer_idx: (N, d) array}
    Returns dict with profile and summary metrics.
    """
    layers = sorted(embeddings.keys())
    N = embeddings[layers[0]].shape[0]

    rng = np.random.RandomState(42)
    anchor_idx = rng.choice(N, min(n_anchor, N), replace=False)
    anchor_idx.sort()

    # Precompute kNN for all layers
    print(f"    Computing k-NN (k={k}) for {len(layers)} layers, {len(anchor_idx)} anchors...")
    layer_knn = {}
    layer_dists = {}
    for li in layers:
        knn_idx, knn_dists = compute_knn(embeddings[li], k, anchor_idx)
        layer_knn[li] = knn_idx
        layer_dists[li] = knn_dists

    # Compute pairwise NRS between consecutive layers
    jaccard_profile = []
    rank_corr_profile = []

    for i in range(len(layers) - 1):
        l1, l2 = layers[i], layers[i + 1]
        jacc = jaccard_overlap(layer_knn[l1], layer_knn[l2])
        jaccard_profile.append(float(jacc.mean()))

        # For rank correlation, use union of neighbors from both layers
        # and compare distances in both spaces
        rc = rank_correlation_per_anchor(layer_dists[l1], layer_dists[l2])
        rank_corr_profile.append(float(np.nanmean(rc)))

    # Also compute first-to-last layer NRS
    jacc_first_last = float(jaccard_overlap(layer_knn[layers[0]], layer_knn[layers[-1]]).mean())
    rc_first_last = float(np.nanmean(rank_correlation_per_anchor(
        layer_dists[layers[0]], layer_dists[layers[-1]])))

    jacc_arr = np.array(jaccard_profile)
    rc_arr = np.array(rank_corr_profile)
    n_transitions = len(jacc_arr)
    third = max(1, n_transitions // 3)

    result = dict(
        layers=[int(l) for l in layers],
        n_layers=len(layers) - 1,
        k=k,
        n_anchors=len(anchor_idx),
        # Jaccard profile
        jaccard_profile=jaccard_profile,
        mean_jaccard=float(jacc_arr.mean()),
        std_jaccard=float(jacc_arr.std()),
        min_jaccard=float(jacc_arr.min()),
        max_jaccard=float(jacc_arr.max()),
        early_jaccard=float(jacc_arr[:third].mean()),
        mid_jaccard=float(jacc_arr[third:2*third].mean()),
        late_jaccard=float(jacc_arr[-third:].mean()),
        jaccard_auc=float(np.trapz(jacc_arr) / n_transitions) if n_transitions > 1 else 0,
        jaccard_trend=float(jacc_arr[-third:].mean() - jacc_arr[:third].mean()),
        jaccard_first_last=jacc_first_last,
        # Rank correlation profile
        rank_corr_profile=rank_corr_profile,
        mean_rank_corr=float(rc_arr.mean()),
        std_rank_corr=float(rc_arr.std()),
        min_rank_corr=float(rc_arr.min()),
        early_rank_corr=float(rc_arr[:third].mean()),
        late_rank_corr=float(rc_arr[-third:].mean()),
        rank_corr_auc=float(np.trapz(rc_arr) / n_transitions) if n_transitions > 1 else 0,
        rank_corr_trend=float(rc_arr[-third:].mean() - rc_arr[:third].mean()),
        rank_corr_first_last=rc_first_last,
        # Instability = 1 - jaccard (how much neighborhoods change)
        mean_instability=float(1 - jacc_arr.mean()),
        max_instability=float(1 - jacc_arr.min()),
        instability_cv=float((1 - jacc_arr).std() / (1 - jacc_arr).mean()) if (1 - jacc_arr).mean() > 0 else 0,
    )
    return result


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def compute_nrs_for_model(texts, hf_model, device, k=K_NEIGHBORS,
                          n_anchor=N_ANCHOR, layers_per_pass=LAYERS_PER_PASS):
    force_bidir = globals().get("_FORCE_BIDIR", False)
    model, tokenizer, info = _load_model(hf_model, device, force_bidir=force_bidir)
    n_layers = info["n_layers"]
    is_decoder = info["is_decoder"]
    pooling = get_pooling_strategy(hf_model, is_decoder)
    batch_size = 4 if is_decoder else 64
    if info["hidden_size"] >= 4096 and not is_decoder:
        batch_size = min(batch_size, 4)

    print(f"  arch={info['arch']}  layers={n_layers}  "
          f"hidden={info['hidden_size']}  pooling={pooling}  bs={batch_size}")

    all_layer_ids = list(range(n_layers + 1))
    groups = [all_layer_ids[i:i + layers_per_pass]
              for i in range(0, len(all_layer_ids), layers_per_pass)]

    embeddings = {}
    print(f"  Extracting {len(all_layer_ids)} layers in {len(groups)} pass(es)")
    for gi, group in enumerate(groups):
        t0 = time.time()
        layer_data = _run_one_pass(
            model, tokenizer, texts, set(group),
            batch_size, 512, device, pooling, True, is_decoder,
        )
        for li in group:
            embeddings[li] = layer_data[li]
        del layer_data
        gc.collect(); torch.cuda.empty_cache()
        print(f"    Pass {gi + 1}/{len(groups)} done in {time.time() - t0:.0f}s", flush=True)

    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

    print(f"  Computing NRS profile...")
    result = compute_nrs_profile(embeddings, k, n_anchor)
    result["hf_model"] = hf_model
    result["hidden_size"] = info["hidden_size"]
    result["arch"] = info["arch"]
    result["n_samples"] = len(texts)
    return result


# ----------------------------------------------------------------------
# Correlation
# ----------------------------------------------------------------------

def residualize(x, z):
    z = np.asarray(z, float); x = np.asarray(x, float)
    zc = z - z.mean(); d = (zc**2).sum()
    if d < 1e-12: return x - x.mean()
    b = (x * zc).sum() / d
    return x - b * z - (x.mean() - b * z.mean())

def sig(p):
    if p < 0.01: return "***"
    if p < 0.05: return "** "
    if p < 0.1: return "*  "
    return "   "

def analyse_correlations(all_results, ds_key):
    mteb = MTEB_SCORES.get(ds_key, {})
    if not mteb:
        return
    metric_names = [
        "mean_jaccard", "std_jaccard", "min_jaccard", "max_jaccard",
        "early_jaccard", "mid_jaccard", "late_jaccard",
        "jaccard_auc", "jaccard_trend", "jaccard_first_last",
        "mean_rank_corr", "std_rank_corr", "min_rank_corr",
        "early_rank_corr", "late_rank_corr",
        "rank_corr_auc", "rank_corr_trend", "rank_corr_first_last",
        "mean_instability", "max_instability", "instability_cv",
    ]
    names, mteb_vals, dims = [], [], []
    metrics = {m: [] for m in metric_names}
    for sn, data in all_results.items():
        score = mteb.get(sn)
        if score is None: continue
        names.append(sn); mteb_vals.append(score); dims.append(data["hidden_size"])
        for m in metric_names:
            metrics[m].append(data.get(m, 0.0))
    if len(names) < 4:
        return

    mteb_arr = np.array(mteb_vals)
    log_dims = np.log(np.array(dims, float))

    print(f"\n  Correlation with MTEB ({ds_key}, n={len(names)}):")
    print("  %28s  %8s %7s   %8s %7s   %9s %7s" % (
        "Metric","Spearman","p","Pearson","p","Partial_r","p"))
    for m in metric_names:
        v = np.array(metrics[m])
        if np.std(v) < 1e-12: continue
        rs, ps = spearmanr(v, mteb_arr)
        rp, pp = pearsonr(v, mteb_arr)
        vr = residualize(v, log_dims); mr = residualize(mteb_arr, log_dims)
        pr, prp = pearsonr(vr, mr)
        print("  %28s  %+.4f %.4f%s  %+.4f %.4f%s  %+.4f %.4f%s" % (
            m,rs,ps,sig(ps),rp,pp,sig(pp),pr,prp,sig(prp)))
    rs, ps = spearmanr(log_dims, mteb_arr); rp, pp = pearsonr(log_dims, mteb_arr)
    print("  %28s  %+.4f %.4f%s  %+.4f %.4f%s  %9s" % (
        "log(dim)",rs,ps,sig(ps),rp,pp,sig(pp),"---"))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="banking77,arguana")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_n", type=int, default=MAX_N,
                        help="Optional cap on per-task texts (default: use all).")
    parser.add_argument("--k", type=int, default=K_NEIGHBORS)
    parser.add_argument("--n_anchor", type=int, default=N_ANCHOR)
    parser.add_argument("--layers_per_pass", type=int, default=LAYERS_PER_PASS)
    parser.add_argument("--output_dir", default="./outputs/nrs_profile")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument(
        "--model_source", type=str, default="embed",
        choices=["embed", "llm", "bidir"],
        help="embed = embedding models; llm = pure LLM; bidir = bidirectional LLM embeddings",
    )
    args = parser.parse_args()

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

    print(f"Neighborhood Rank Stability Profile")
    print(f"  Datasets: {ds_list}")
    print(f"  Models: {[sn for _, sn in model_list]}")
    print(f"  k={args.k}, n_anchor={args.n_anchor}")

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
                print(f"\n  [SKIP] {short_name} (previously failed, marker exists)")
                continue

            print(f"\n  Model: {short_name}  ({hf_model})")
            try:
                result = compute_nrs_for_model(
                    texts, hf_model, args.device,
                    k=args.k, n_anchor=args.n_anchor,
                    layers_per_pass=args.layers_per_pass,
                )
                with open(per_model_path, "w") as wf:
                    json.dump(result, wf, indent=2)
                print(f"  Saved: {per_model_path}")
                print(f"    mean_jaccard={result['mean_jaccard']:.4f}  "
                      f"late_jaccard={result['late_jaccard']:.4f}  "
                      f"jaccard_trend={result['jaccard_trend']:.4f}  "
                      f"mean_rank_corr={result['mean_rank_corr']:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                import traceback; traceback.print_exc()
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

        # Merge and analyse (scan directory for all .json files)
        all_results = {}
        for fname in sorted(os.listdir(ds_dir)):
            if fname.endswith(".json") and fname != "all_results.json":
                sn = fname[:-5]
                with open(os.path.join(ds_dir, fname)) as rf:
                    all_results[sn] = json.load(rf)
        if all_results:
            with open(os.path.join(ds_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f, indent=2)
            analyse_correlations(all_results, ds_key)

    print("\nDone.")


if __name__ == "__main__":
    main()
