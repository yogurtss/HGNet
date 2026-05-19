"""
SPHERE Dataset Preparation Script
==================================
Splits SPHERE domains into train/dev/test and can merge all domains into one
cleaned SciERC-compatible dataset for Z-NERD and HA-GNN.

Usage:
    python prepare_sphere.py
    python prepare_sphere.py --domain "computer science" --convert_to_scierc
    python prepare_sphere.py --merge_all --convert_to_scierc
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

SEED = 42
SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}

RAW_DOMAIN_FILES = {
    "biology": "annotated_biology_sentences.jsonl",
    "computer science": "computer_science.jsonl",
    "material science": "annotated_materials_sentences.jsonl",
    "physics": "annotated_physics_sentences.jsonl",
}

CLEANED_DOMAIN_FILES = {
    "biology": "annotated_biology_sentences_cleaned.jsonl",
    "computer science": "computer_science_cleaned.jsonl",
    "material science": "annotated_materials_sentences_cleaned.jsonl",
    "physics": "annotated_physics_sentences_cleaned.jsonl",
}

# Kept for backward-compatible single-domain behavior.
DOMAIN_FILES = RAW_DOMAIN_FILES

RELATION_LABEL_ALIASES = {
    "related-to": "Related-To",
    "dependent-on": "Dependent-On",
    "is-a-subconcept-of": "Part-Of",
}
CANONICAL_RELATION_TYPES = set(RELATION_LABEL_ALIASES.values())
RAW_RELATION_TYPES = set(RELATION_LABEL_ALIASES.keys())

ENTITY_LABEL_ALIASES = {
    "BiologicalProcess": "Biological Process",
    "TaxonomicRank": "Taxonomic Rank",
    "MetabolicPathway": "Pathway",
    "Processing Technique": "ProcessingTechnique",
    " ProcessingTechnique": "ProcessingTechnique",
    "Phenomenom": "Phenomenon",
    "Phenomenusm": "Phenomenon",
    "Phenomenation": "Phenomenon",
    "Theor y": "Theory",
    "Theorem": "Theory",
    "Theoretical": "Theory",
    "Technic": "Technique",
    "field": "Field",
}


def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  Skipping invalid JSON at {filepath}:{line_no}")
    return data


def save_jsonl(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_entity_type(label):
    if label is None:
        return None
    normalized = str(label).strip()
    if not normalized:
        return None
    normalized = ENTITY_LABEL_ALIASES.get(normalized, normalized)
    if normalized in RAW_RELATION_TYPES or normalized in CANONICAL_RELATION_TYPES:
        return None
    return normalized


def normalize_relation_type(label):
    if label is None:
        return None
    normalized = str(label).strip()
    if not normalized:
        return None
    return RELATION_LABEL_ALIASES.get(normalized, normalized)


def normalize_space(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def char_span_to_token_span(text, span):
    """Convert a character span [start, end) to an inclusive token span."""
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append((match.group(0), match.start(), match.end()))

    start, end = span
    token_indices = [
        idx
        for idx, (_, tok_start, tok_end) in enumerate(tokens)
        if max(tok_start, start) < min(tok_end, end)
    ]
    if not token_indices:
        return None, [tok for tok, _, _ in tokens]
    return (token_indices[0], token_indices[-1]), [tok for tok, _, _ in tokens]


def remove_overlapping_entities(entities, report):
    indexed = list(enumerate(entities))
    indexed.sort(
        key=lambda pair: (
            -(pair[1]["span"][1] - pair[1]["span"][0]),
            pair[1]["span"][0],
            pair[0],
        )
    )
    kept = []
    removed_ids = set()
    for _, entity in indexed:
        start, end = entity["span"]
        overlaps = any(max(start, kept_ent["span"][0]) < min(end, kept_ent["span"][1]) for kept_ent in kept)
        if overlaps:
            removed_ids.add(entity["id"])
            report["dropped_entities"]["overlap_span"] += 1
            continue
        kept.append(entity)

    kept.sort(key=lambda entity: (entity["span"][0], entity["span"][1], str(entity["id"])))
    return kept, removed_ids


def clean_sphere_item(item, domain, source_index, report):
    text = item.get("text", "")
    if not isinstance(text, str) or not text.strip():
        report["dropped_sentences"]["empty_text"] += 1
        return None

    raw_entities = item.get("entities") or []
    raw_relations = item.get("relations") or []
    report["input"]["sentences"] += 1
    report["input"]["entities"] += len(raw_entities)
    report["input"]["relations"] += len(raw_relations)

    entities = []
    seen_entity_ids = {}
    ambiguous_ids = set()
    removed_entity_ids = set()

    for entity in raw_entities:
        entity_type = normalize_entity_type(entity.get("type"))
        entity_id = entity.get("id")
        span = entity.get("span")

        if entity_type is None:
            report["dropped_entities"]["empty_or_relation_type_label"] += 1
            removed_entity_ids.add(entity_id)
            continue
        if not isinstance(span, list) or len(span) != 2 or not all(isinstance(value, int) for value in span):
            report["dropped_entities"]["bad_span_shape"] += 1
            removed_entity_ids.add(entity_id)
            continue

        start, end = span
        if start < 0 or end <= start or end > len(text):
            report["dropped_entities"]["span_out_of_bounds"] += 1
            removed_entity_ids.add(entity_id)
            continue

        surface = text[start:end]
        name = entity.get("name") or surface
        if entity.get("name") and normalize_space(surface) != normalize_space(str(entity.get("name"))):
            report["warnings"]["span_name_mismatch"] += 1

        signature = (entity_id, name, start, end, entity_type)
        previous_signature = seen_entity_ids.get(entity_id)
        if previous_signature == signature:
            report["dropped_entities"]["duplicate_entity_exact"] += 1
            continue
        if previous_signature is not None and previous_signature != signature:
            ambiguous_ids.add(entity_id)
            report["warnings"]["ambiguous_entity_id"] += 1

        seen_entity_ids[entity_id] = signature
        entities.append(
            {
                "id": entity_id,
                "name": name,
                "span": [start, end],
                "type": entity_type,
            }
        )

    entities, overlap_removed_ids = remove_overlapping_entities(entities, report)
    removed_entity_ids.update(overlap_removed_ids)
    valid_entity_ids = {entity["id"] for entity in entities}
    entity_id_to_span = {entity["id"]: tuple(entity["span"]) for entity in entities}

    relations = []
    seen_relations = set()
    for relation in raw_relations:
        source = relation.get("source")
        target = relation.get("target")
        relation_type = normalize_relation_type(relation.get("type"))

        if relation_type is None:
            report["dropped_relations"]["empty_label"] += 1
            continue
        if source == target:
            report["dropped_relations"]["self_relation"] += 1
            continue
        if source in ambiguous_ids or target in ambiguous_ids:
            report["dropped_relations"]["ambiguous_entity_id"] += 1
            continue
        if source not in valid_entity_ids or target not in valid_entity_ids:
            reason = "references_removed_entity" if source in removed_entity_ids or target in removed_entity_ids else "dangling_relation"
            report["dropped_relations"][reason] += 1
            continue

        key = (source, target, relation_type)
        if key in seen_relations:
            report["dropped_relations"]["duplicate_relation"] += 1
            continue

        source_span = entity_id_to_span[source]
        target_span = entity_id_to_span[target]
        if source_span == target_span:
            report["dropped_relations"]["self_span_relation"] += 1
            continue

        seen_relations.add(key)
        relations.append({"source": source, "target": target, "type": relation_type})

    if not entities:
        report["dropped_sentences"]["no_valid_entities"] += 1
        return None

    report["output"]["sentences"] += 1
    report["output"]["entities"] += len(entities)
    report["output"]["relations"] += len(relations)

    source_sentence_id = item.get("sentence_id", source_index + 1)
    doc_id = f"sphere-{domain.replace(' ', '_')}-{source_sentence_id}"
    return {
        "doc_id": doc_id,
        "text": text,
        "entities": entities,
        "relations": relations,
        "sentence_id": source_sentence_id,
        "source_domain": domain,
    }


def split_items(items):
    shuffled = list(items)
    random.Random(SEED).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_dev = int(n * SPLIT_RATIOS["dev"])
    return {
        "train": shuffled[:n_train],
        "dev": shuffled[n_train : n_train + n_dev],
        "test": shuffled[n_train + n_dev :],
    }


def split_domain(domain_dir, filename):
    """Split a single domain's JSONL file into train/dev/test."""
    filepath = os.path.join(domain_dir, filename)
    if not os.path.exists(filepath):
        print(f"  Skipping: {filepath} not found")
        return

    data = load_jsonl(filepath)
    splits = split_items(data)

    for split_name, split_data in splits.items():
        out_path = os.path.join(domain_dir, f"{split_name}.jsonl")
        save_jsonl(split_data, out_path)
        print(f"  {split_name}: {len(split_data)} sentences -> {out_path}")


def sphere_items_to_scierc_docs(data):
    scierc_docs = []
    skipped = Counter()

    for item in data:
        text = item.get("text", "")
        tokens = []
        entity_candidates = []

        for entity in item.get("entities", []):
            token_span, tokens = char_span_to_token_span(text, entity["span"])
            if token_span is None:
                skipped["entities"] += 1
                continue
            tok_start, tok_end = token_span
            entity_candidates.append(
                {
                    "id": entity["id"],
                    "span": (tok_start, tok_end),
                    "type": entity["type"],
                }
            )

        if not tokens:
            tokens = text.split()

        entity_candidates.sort(
            key=lambda entity: (
                -(entity["span"][1] - entity["span"][0]),
                entity["span"][0],
                str(entity["id"]),
            )
        )
        kept_entities = []
        removed_token_entity_ids = set()
        for entity in entity_candidates:
            start, end = entity["span"]
            overlaps = any(max(start, kept["span"][0]) <= min(end, kept["span"][1]) for kept in kept_entities)
            if overlaps:
                removed_token_entity_ids.add(entity["id"])
                skipped["token_overlap_entities"] += 1
                continue
            kept_entities.append(entity)

        kept_entities.sort(key=lambda entity: (entity["span"][0], entity["span"][1], str(entity["id"])))
        ner_annotations = [[entity["span"][0], entity["span"][1], entity["type"]] for entity in kept_entities]
        entity_id_to_span = {entity["id"]: entity["span"] for entity in kept_entities}

        rel_annotations = []
        for relation in item.get("relations", []):
            src_span = entity_id_to_span.get(relation["source"])
            tgt_span = entity_id_to_span.get(relation["target"])
            if src_span is None or tgt_span is None:
                if relation["source"] in removed_token_entity_ids or relation["target"] in removed_token_entity_ids:
                    skipped["token_overlap_relations"] += 1
                else:
                    skipped["relations"] += 1
                continue
            if src_span == tgt_span:
                skipped["self_span_relations"] += 1
                continue
            rel_annotations.append(
                [src_span[0], src_span[1], tgt_span[0], tgt_span[1], relation["type"]]
            )

        scierc_docs.append(
            {
                "doc_id": item.get("doc_id"),
                "source_domain": item.get("source_domain"),
                "sentences": [tokens],
                "ner": [ner_annotations],
                "relations": [rel_annotations],
            }
        )

    return scierc_docs, skipped


def sphere_to_scierc_format(domain_dir):
    """
    Convert SPHERE split files (train/dev/test.jsonl) to SciERC JSON format.
    SciERC format groups sentences into documents. Since SPHERE sentences are
    independent, each sentence becomes its own single-sentence document.
    """
    for split in ["train", "dev", "test"]:
        jsonl_path = os.path.join(domain_dir, f"{split}.jsonl")
        if not os.path.exists(jsonl_path):
            continue

        data = load_jsonl(jsonl_path)
        scierc_docs, skipped = sphere_items_to_scierc_docs(data)
        out_path = os.path.join(domain_dir, f"{split}.json")
        save_json(scierc_docs, out_path)
        print(
            f"  SciERC format: {split}.json ({len(scierc_docs)} docs, "
            f"skipped {skipped['entities']} entities/{skipped['relations']} relations)"
        )


def build_label_maps(splits):
    entity_types = set()
    relation_types = {"No-Relation"}
    for split_data in splits.values():
        for item in split_data:
            for entity in item.get("entities", []):
                entity_types.add(entity["type"])
            for relation in item.get("relations", []):
                relation_types.add(relation["type"])
    return {
        "entity_types": sorted(entity_types),
        "relation_types": sorted(relation_types),
        "entity_map": {label: idx for idx, label in enumerate(sorted(entity_types))},
        "relation_map": {label: idx for idx, label in enumerate(sorted(relation_types))},
        "entity_label_aliases": ENTITY_LABEL_ALIASES,
        "relation_label_aliases": RELATION_LABEL_ALIASES,
    }


def add_split_report(report, splits):
    report["splits"] = {}
    for split_name, split_data in splits.items():
        entity_counts = Counter()
        relation_counts = Counter()
        domain_counts = Counter()
        for item in split_data:
            domain_counts[item.get("source_domain")] += 1
            for entity in item.get("entities", []):
                entity_counts[entity["type"]] += 1
            for relation in item.get("relations", []):
                relation_counts[relation["type"]] += 1

        report["splits"][split_name] = {
            "sentences": len(split_data),
            "domains": dict(sorted(domain_counts.items())),
            "entity_labels": dict(sorted(entity_counts.items())),
            "relation_labels": dict(sorted(relation_counts.items())),
        }


def default_report():
    return {
        "input": Counter(),
        "output": Counter(),
        "dropped_sentences": Counter(),
        "dropped_entities": Counter(),
        "dropped_relations": Counter(),
        "warnings": Counter(),
        "scierc_conversion_skipped": Counter(),
    }


def counter_to_dict(value):
    if isinstance(value, Counter):
        return dict(sorted(value.items()))
    if isinstance(value, dict):
        return {key: counter_to_dict(val) for key, val in value.items()}
    return value


def merge_all_domains(base_dir, output_dir, source, convert_to_scierc):
    source_files = CLEANED_DOMAIN_FILES if source == "cleaned" else RAW_DOMAIN_FILES
    per_domain_splits = {}
    report = default_report()
    report["source"] = source
    report["domains"] = {}

    for domain, filename in source_files.items():
        domain_report = default_report()
        domain_dir = os.path.join(base_dir, domain)
        filepath = os.path.join(domain_dir, filename)
        if not os.path.exists(filepath):
            print(f"  Skipping: {filepath} not found")
            continue

        print(f"\n--- Cleaning: {domain} ({filename}) ---")
        raw_items = load_jsonl(filepath)
        cleaned_items = []
        for idx, item in enumerate(raw_items):
            cleaned = clean_sphere_item(item, domain, idx, domain_report)
            if cleaned is not None:
                cleaned_items.append(cleaned)

        domain_splits = split_items(cleaned_items)
        per_domain_splits[domain] = domain_splits
        report["domains"][domain] = counter_to_dict(domain_report)

        for section in ["input", "output", "dropped_sentences", "dropped_entities", "dropped_relations", "warnings"]:
            report[section].update(domain_report[section])

        print(
            f"  kept {len(cleaned_items)}/{len(raw_items)} sentences, "
            f"{domain_report['output']['entities']} entities, "
            f"{domain_report['output']['relations']} relations"
        )

    merged_splits = {"train": [], "dev": [], "test": []}
    for domain_splits in per_domain_splits.values():
        for split_name, split_data in domain_splits.items():
            merged_splits[split_name].extend(split_data)

    for split_name in merged_splits:
        random.Random(SEED).shuffle(merged_splits[split_name])

    os.makedirs(output_dir, exist_ok=True)
    for split_name, split_data in merged_splits.items():
        save_jsonl(split_data, os.path.join(output_dir, f"{split_name}.jsonl"))
        print(f"  merged {split_name}: {len(split_data)} sentences")

    if convert_to_scierc:
        for split_name, split_data in merged_splits.items():
            scierc_docs, skipped = sphere_items_to_scierc_docs(split_data)
            report["scierc_conversion_skipped"].update(skipped)
            save_json(scierc_docs, os.path.join(output_dir, f"{split_name}.json"))
            print(f"  SciERC {split_name}: {len(scierc_docs)} docs")

    label_maps = build_label_maps(merged_splits)
    save_json(label_maps, os.path.join(output_dir, "label_maps.json"))
    add_split_report(report, merged_splits)
    save_json(counter_to_dict(report), os.path.join(output_dir, "prepare_report.json"))
    print(f"\nMerged SPHERE data written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare SPHERE dataset splits")
    parser.add_argument(
        "--domain", type=str, default=None, help="Single domain to process (default: all)"
    )
    parser.add_argument(
        "--convert_to_scierc",
        action="store_true",
        help="Also convert JSONL splits to SciERC JSON format",
    )
    parser.add_argument(
        "--merge_all",
        action="store_true",
        help="Clean, split, and merge all SPHERE domains into one output directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Merged output directory (default: datasets/SPHERE/merged)",
    )
    parser.add_argument(
        "--source",
        choices=["cleaned", "raw"],
        default="cleaned",
        help="Input files for merged mode (default: cleaned)",
    )
    args = parser.parse_args()

    random.seed(SEED)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.merge_all:
        output_dir = args.output_dir or os.path.join(base_dir, "merged")
        merge_all_domains(base_dir, output_dir, args.source, args.convert_to_scierc)
        print("\nDone!")
        return

    domains = {args.domain: DOMAIN_FILES[args.domain]} if args.domain else DOMAIN_FILES

    for domain, filename in domains.items():
        domain_dir = os.path.join(base_dir, domain)
        print(f"\n--- Processing: {domain} ---")
        split_domain(domain_dir, filename)

        if args.convert_to_scierc:
            print("  Converting to SciERC format...")
            sphere_to_scierc_format(domain_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
