"""
SPHERE Dataset Preparation Script
==================================
Splits each SPHERE domain into train/dev/test (80/10/10) and optionally
converts to SciERC-compatible format for use with Z-NERD and HA-GNN.

Usage:
    python prepare_sphere.py                           # Split all domains
    python prepare_sphere.py --domain "computer science"  # Split one domain
    python prepare_sphere.py --convert_to_scierc       # Also convert to SciERC format
"""

import os
import json
import random
import argparse
from collections import defaultdict

SEED = 42
random.seed(SEED)

DOMAIN_FILES = {
    "biology": "annotated_biology_sentences.jsonl",
    "computer science": "computer_science.jsonl",
    "material science": "annotated_materials_sentences.jsonl",
    "physics": "annotated_physics_sentences.jsonl",
}

SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}


def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def save_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def split_domain(domain_dir, filename):
    """Split a single domain's JSONL file into train/dev/test."""
    filepath = os.path.join(domain_dir, filename)
    if not os.path.exists(filepath):
        print(f"  Skipping: {filepath} not found")
        return

    data = load_jsonl(filepath)
    random.shuffle(data)

    n = len(data)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_dev = int(n * SPLIT_RATIOS["dev"])

    splits = {
        "train": data[:n_train],
        "dev": data[n_train : n_train + n_dev],
        "test": data[n_train + n_dev :],
    }

    for split_name, split_data in splits.items():
        out_path = os.path.join(domain_dir, f"{split_name}.jsonl")
        save_jsonl(split_data, out_path)
        print(f"  {split_name}: {len(split_data)} sentences -> {out_path}")


def sphere_to_scierc_format(domain_dir):
    """
    Convert SPHERE split files (train/dev/test.jsonl) to SciERC JSON format.
    SciERC format groups sentences into documents. Since SPHERE sentences are
    independent, each sentence becomes its own single-sentence 'document'.
    """
    for split in ["train", "dev", "test"]:
        jsonl_path = os.path.join(domain_dir, f"{split}.jsonl")
        if not os.path.exists(jsonl_path):
            continue

        data = load_jsonl(jsonl_path)
        scierc_docs = []

        for item in data:
            text = item.get("text", "")
            tokens = text.split()  # Whitespace tokenization
            entities = item.get("entities", [])
            relations = item.get("relations", [])

            # Convert character spans to token spans
            char_to_token = {}
            char_pos = 0
            for tok_idx, tok in enumerate(tokens):
                for c in range(char_pos, char_pos + len(tok)):
                    char_to_token[c] = tok_idx
                char_pos += len(tok) + 1  # +1 for space

            ner_annotations = []
            entity_id_to_span = {}
            for ent in entities:
                char_start, char_end = ent["span"]
                tok_start = char_to_token.get(char_start)
                tok_end = char_to_token.get(max(char_start, char_end - 1))
                if tok_start is not None and tok_end is not None:
                    ner_annotations.append([tok_start, tok_end, ent["type"]])
                    entity_id_to_span[ent["id"]] = (tok_start, tok_end)

            rel_annotations = []
            for rel in relations:
                src_span = entity_id_to_span.get(rel["source"])
                tgt_span = entity_id_to_span.get(rel["target"])
                if src_span and tgt_span:
                    rel_annotations.append(
                        [src_span[0], src_span[1], tgt_span[0], tgt_span[1], rel["type"]]
                    )

            scierc_docs.append(
                {
                    "sentences": [tokens],
                    "ner": [ner_annotations],
                    "relations": [rel_annotations],
                }
            )

        out_path = os.path.join(domain_dir, f"{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scierc_docs, f)
        print(f"  SciERC format: {split}.json ({len(scierc_docs)} docs)")


def main():
    parser = argparse.ArgumentParser(description="Prepare SPHERE dataset splits")
    parser.add_argument(
        "--domain", type=str, default=None, help="Single domain to process (default: all)"
    )
    parser.add_argument(
        "--convert_to_scierc",
        action="store_true",
        help="Also convert to SciERC JSON format",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    domains = {args.domain: DOMAIN_FILES[args.domain]} if args.domain else DOMAIN_FILES

    for domain, filename in domains.items():
        domain_dir = os.path.join(base_dir, domain)
        print(f"\n--- Processing: {domain} ---")
        split_domain(domain_dir, filename)

        if args.convert_to_scierc:
            print(f"  Converting to SciERC format...")
            sphere_to_scierc_format(domain_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
