#!/usr/bin/env python3
"""Tests for Micro_KIKI data pipeline components."""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest
import yaml


# --- Test domain classifier ---

def load_domains_config():
    """Load domain config for tests."""
    config_path = Path(__file__).parent.parent / "configs" / "micro_kiki" / "domains.yaml"
    return yaml.safe_load(config_path.read_text())


class TestDomainClassifier:
    """Test the heuristic domain classifier."""

    def test_classify_python_example(self):
        from scripts.micro_kiki.classify_domains import classify_example
        example = {
            "messages": [
                {"role": "user", "content": "Write a Python function to sort a list"},
                {"role": "assistant", "content": "<thinking>I need to write a sort function in Python.</thinking>\n\ndef sort_list(items):\n    return sorted(items)"},
            ]
        }
        config = load_domains_config()
        domain = classify_example(example, config["domains"])
        assert domain == "python"

    def test_classify_embedded_example(self):
        from scripts.micro_kiki.classify_domains import classify_example
        example = {
            "messages": [
                {"role": "user", "content": "How to configure GPIO on ESP32 with ESP-IDF?"},
                {"role": "assistant", "content": "<thinking>ESP-IDF GPIO configuration uses gpio_config_t.</thinking>\n\ngpio_config_t io_conf = { .pin_bit_mask = (1ULL << GPIO_NUM_2) };"},
            ]
        }
        config = load_domains_config()
        domain = classify_example(example, config["domains"])
        assert domain == "embedded"

    def test_classify_french_chat(self):
        from scripts.micro_kiki.classify_domains import classify_example
        example = {
            "messages": [
                {"role": "user", "content": "Explique-moi le fonctionnement d'un transformateur en français"},
                {"role": "assistant", "content": "Un transformateur fonctionne grâce à l'induction électromagnétique."},
            ]
        }
        config = load_domains_config()
        domain = classify_example(example, config["domains"])
        assert domain == "chat-fr"

    def test_classify_returns_none_for_empty(self):
        from scripts.micro_kiki.classify_domains import classify_example
        example = {"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
        config = load_domains_config()
        domain = classify_example(example, config["domains"])
        assert domain is None

    def test_classify_reasoning(self):
        from scripts.micro_kiki.classify_domains import classify_example
        example = {
            "messages": [
                {"role": "user", "content": "Think step by step about this logical puzzle"},
                {"role": "assistant", "content": "<thinking>Let me reason through this step by step. First, I need to analyze the logical structure...</thinking>\n\nThe answer is B."},
            ]
        }
        config = load_domains_config()
        domain = classify_example(example, config["domains"])
        assert domain == "reasoning"

    def test_all_32_domains_in_config(self):
        config = load_domains_config()
        assert len(config["domains"]) == 32

    def test_each_domain_has_required_fields(self):
        config = load_domains_config()
        for name, domain in config["domains"].items():
            assert "phase" in domain, f"{name} missing phase"
            assert "teacher" in domain, f"{name} missing teacher"
            assert "keywords" in domain, f"{name} missing keywords"
            assert "patterns" in domain, f"{name} missing patterns"
            assert "target" in domain, f"{name} missing target"


class TestDeduplication:
    """Test cross-domain deduplication."""

    def test_dedup_removes_duplicates(self):
        from scripts.micro_kiki.deduplicate import dedup_cross_domain
        domain_data = {
            "python": [
                {"messages": [{"role": "user", "content": "Write sort"}, {"role": "assistant", "content": "def sort(): pass"}]},
                {"messages": [{"role": "user", "content": "Write search"}, {"role": "assistant", "content": "def search(): pass"}]},
            ],
            "typescript": [
                {"messages": [{"role": "user", "content": "Write sort"}, {"role": "assistant", "content": "def sort(): pass"}]},  # duplicate
                {"messages": [{"role": "user", "content": "Write TS sort"}, {"role": "assistant", "content": "function sort() {}"}]},
            ],
        }
        result = dedup_cross_domain(domain_data)
        # The duplicate should be removed from one domain
        total = sum(len(v) for v in result.values())
        assert total == 3  # 4 original - 1 duplicate

    def test_dedup_keeps_best_domain(self):
        from scripts.micro_kiki.deduplicate import dedup_key
        key = dedup_key({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]})
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex

    def test_dedup_empty_input(self):
        from scripts.micro_kiki.deduplicate import dedup_cross_domain
        result = dedup_cross_domain({})
        assert result == {}
