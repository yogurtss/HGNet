"""
SPHERE Dataset Post-Processing: Span Correction & Relation Normalization
=========================================================================
Fixes three issues in the SPHERE dataset:
1. Broken span offsets — recomputes [start, end] via string matching
2. Relation type explosion — maps all types to the 3 canonical KG types
3. Removes unfixable entities (name not findable in text at all)

Usage:
    python fix_sphere_spans.py                  # Fix all domains
    python fix_sphere_spans.py --domain cs      # Fix one domain
    python fix_sphere_spans.py --dry-run        # Stats only, no output files
"""
import json
import re
import os
import argparse
from collections import Counter

# ============================================================================
# Relation type mapping → 3 canonical KG types
# ============================================================================
RELATION_MAP = {
    # === related-to ===
    "related_to": "related-to",
    "related-to": "related-to",
    "enhances": "related-to",
    "improves": "related-to",
    "leverages": "related-to",
    "optimizes": "related-to",
    "benefits_from": "related-to",
    "applied_to": "related-to",
    "applies_to": "related-to",
    "applies-to": "related-to",
    "influences": "related-to",
    "affects": "related-to",
    "impacts": "related-to",
    "interacts-with": "related-to",
    "interacts_with": "related-to",
    "exhibits-property": "related-to",
    "exhibits_property": "related-to",
    "measures-property": "related-to",
    "measures_property": "related-to",
    "modifies-property": "related-to",
    "modifies_property": "related-to",
    "regulates": "related-to",
    "inhibits": "related-to",
    "produces": "related-to",
    "causes": "related-to",
    "enables": "related-to",
    "supports": "related-to",
    "contributes_to": "related-to",
    "contributes-to": "related-to",
    "complements": "related-to",
    "uses": "related-to",
    "employs": "related-to",
    "utilizes": "related-to",
    "implements": "related-to",
    "extends": "related-to",
    "modifies": "related-to",
    "transforms": "related-to",
    "generates": "related-to",
    "evaluates": "related-to",
    "measures": "related-to",
    "analyzes": "related-to",
    "processes": "related-to",
    "studies": "related-to",
    "addresses": "related-to",
    "handles": "related-to",
    "manages": "related-to",
    "integrates": "related-to",
    "combines": "related-to",
    "connects": "related-to",
    "links": "related-to",
    "maps_to": "related-to",
    "maps-to": "related-to",
    "associated_with": "related-to",
    "associated-with": "related-to",
    "correlates_with": "related-to",
    "correlates-with": "related-to",
    "compared_to": "related-to",
    "compared-to": "related-to",
    "contrasts_with": "related-to",
    "contrasts-with": "related-to",
    "activates": "related-to",
    "stimulates": "related-to",
    "triggers": "related-to",
    "induces": "related-to",
    "promotes": "related-to",
    "facilitates": "related-to",
    "reduces": "related-to",
    "suppresses": "related-to",
    "blocks": "related-to",
    "prevents": "related-to",
    "limits": "related-to",
    "constrains": "related-to",
    "encodes": "related-to",
    "expresses": "related-to",
    "binds_to": "related-to",
    "binds-to": "related-to",
    "catalyzes": "related-to",
    "transports": "related-to",
    "signals": "related-to",
    "mediates": "related-to",
    "None": "related-to",

    # === dependent-on ===
    "dependent-on": "dependent-on",
    "dependent_on": "dependent-on",
    "depends_on": "dependent-on",
    "depends-on": "dependent-on",
    "requires": "dependent-on",
    "relies_on": "dependent-on",
    "relies-on": "dependent-on",
    "based_on": "dependent-on",
    "based-on": "dependent-on",
    "derived_from": "dependent-on",
    "derived-from": "dependent-on",
    "is-caused-by": "dependent-on",
    "is_caused_by": "dependent-on",
    "caused_by": "dependent-on",
    "caused-by": "dependent-on",
    "determined_by": "dependent-on",
    "determined-by": "dependent-on",
    "constrained_by": "dependent-on",
    "constrained-by": "dependent-on",

    # === is-a-subconcept-of ===
    "is-a-subconcept-of": "is-a-subconcept-of",
    "is_a_subconcept_of": "is-a-subconcept-of",
    "is-a-type-of": "is-a-subconcept-of",
    "is_a_type_of": "is-a-subconcept-of",
    "belongs-to": "is-a-subconcept-of",
    "belongs_to": "is-a-subconcept-of",
    "is-a-part-of": "is-a-subconcept-of",
    "is_a_part_of": "is-a-subconcept-of",
    "is-part-of": "is-a-subconcept-of",
    "is_part_of": "is-a-subconcept-of",
    "located-in": "is-a-subconcept-of",
    "located_in": "is-a-subconcept-of",
    "subclass-of": "is-a-subconcept-of",
    "subclass_of": "is-a-subconcept-of",
    "instance-of": "is-a-subconcept-of",
    "instance_of": "is-a-subconcept-of",
    "specializes": "is-a-subconcept-of",
    "categorized_as": "is-a-subconcept-of",
    "categorized-as": "is-a-subconcept-of",
    "classified_as": "is-a-subconcept-of",
    "classified-as": "is-a-subconcept-of",
}


def strip_parenthetical(name):
    """Strip trailing parenthetical from entity name for surface form matching.
    'Intergalactic Medium (IGM)' → 'Intergalactic Medium'
    'Density Functional Theory (DFT) Calculations' → 'Density Functional Theory Calculations'
    """
    # Remove parenthetical groups like (IGM), (DFT), (Astronomy), etc.
    stripped = re.sub(r'\s*\([^)]*\)\s*', ' ', name).strip()
    # Collapse multiple spaces
    stripped = re.sub(r'\s+', ' ', stripped)
    return stripped


def find_entity_in_text(text, name, original_span):
    """Find the best matching span for an entity name in the text.
    Returns (start, end) or None if not found.
    Uses 0-based indexing with exclusive end: text[start:end] == mention.
    """
    text_lower = text.lower()

    # Try 1: exact case-insensitive match of full name
    name_lower = name.lower()
    matches = []
    start = 0
    while True:
        idx = text_lower.find(name_lower, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(name)))
        start = idx + 1

    if matches:
        return _pick_closest(matches, original_span)

    # Try 2: strip parentheticals and search
    stripped = strip_parenthetical(name)
    if stripped.lower() != name_lower:
        stripped_lower = stripped.lower()
        matches = []
        start = 0
        while True:
            idx = text_lower.find(stripped_lower, start)
            if idx == -1:
                break
            matches.append((idx, idx + len(stripped)))
            start = idx + 1

        if matches:
            return _pick_closest(matches, original_span)

    # Try 3: singular/plural variants
    for variant in _inflection_variants(stripped if stripped != name else name):
        variant_lower = variant.lower()
        matches = []
        start = 0
        while True:
            idx = text_lower.find(variant_lower, start)
            if idx == -1:
                break
            matches.append((idx, idx + len(variant)))
            start = idx + 1
        if matches:
            return _pick_closest(matches, original_span)

    return None


def _pick_closest(matches, original_span):
    """Pick the match whose start is closest to the original (broken) span start."""
    if len(matches) == 1:
        return matches[0]
    orig_start = original_span[0]
    return min(matches, key=lambda m: abs(m[0] - orig_start))


def _inflection_variants(name):
    """Generate simple singular/plural variants."""
    variants = []
    if name.endswith('s'):
        variants.append(name[:-1])       # Transients → Transient
        if name.endswith('ies'):
            variants.append(name[:-3] + 'y')  # Properties → Property
        if name.endswith('es'):
            variants.append(name[:-2])    # Processes → Process
    else:
        variants.append(name + 's')       # Element → Elements
        variants.append(name + 'es')      # Process → Processes
    return variants


def normalize_relation_type(rel_type):
    """Map a relation type to one of the 3 canonical types."""
    if not rel_type:
        return "related-to"

    # Direct lookup
    canonical = RELATION_MAP.get(rel_type)
    if canonical:
        return canonical

    # Try lowercase
    canonical = RELATION_MAP.get(rel_type.lower())
    if canonical:
        return canonical

    # Try replacing hyphens/underscores
    normalized = rel_type.lower().replace('_', '-')
    canonical = RELATION_MAP.get(normalized)
    if canonical:
        return canonical

    normalized = rel_type.lower().replace('-', '_')
    canonical = RELATION_MAP.get(normalized)
    if canonical:
        return canonical

    # Default fallback
    return "related-to"


def fix_sentence(sentence):
    """Fix a single sentence's entities and relations.
    Returns (fixed_sentence, stats_dict).
    """
    text = sentence["text"]
    entities = sentence.get("entities", [])
    relations = sentence.get("relations", [])

    stats = {
        "entities_total": len(entities),
        "spans_already_correct": 0,
        "spans_fixed": 0,
        "entities_dropped": 0,
        "relations_total": len(relations),
        "relations_remapped": 0,
        "relations_dropped": 0,
    }

    # --- Fix entity spans ---
    fixed_entities = []
    surviving_ids = set()

    for ent in entities:
        name = ent.get("name", "")
        original_span = ent.get("span", [])

        # Handle malformed spans
        if not name or not isinstance(original_span, list) or len(original_span) != 2:
            # Try to find it anyway if we have a name
            if name:
                result = find_entity_in_text(text, name, [0, 0])
                if result:
                    fixed_ent = dict(ent)
                    fixed_ent["span"] = [result[0], result[1]]
                    fixed_entities.append(fixed_ent)
                    surviving_ids.add(ent.get("id"))
                    stats["spans_fixed"] += 1
                else:
                    stats["entities_dropped"] += 1
            else:
                stats["entities_dropped"] += 1
            continue

        # Check if current span is already correct
        s, e = original_span
        # Handle non-integer span values
        try:
            s, e = int(s), int(e)
        except (ValueError, TypeError):
            result = find_entity_in_text(text, name, [0, 0])
            if result:
                fixed_ent = dict(ent)
                fixed_ent["span"] = [result[0], result[1]]
                fixed_entities.append(fixed_ent)
                surviving_ids.add(ent.get("id"))
                stats["spans_fixed"] += 1
            else:
                stats["entities_dropped"] += 1
            continue

        if 0 <= s and e <= len(text) and text[s:e].lower() == name.lower():
            stats["spans_already_correct"] += 1
            fixed_entities.append(ent)
            surviving_ids.add(ent["id"])
            continue

        # Also check if current span matches the stripped name
        stripped = strip_parenthetical(name)
        if stripped.lower() != name.lower() and 0 <= s and e <= len(text):
            if text[s:e].lower() == stripped.lower():
                stats["spans_already_correct"] += 1
                fixed_entities.append(ent)
                surviving_ids.add(ent["id"])
                continue

        # Try to find correct span (use sanitized [s, e] ints)
        result = find_entity_in_text(text, name, [s, e])
        if result:
            new_s, new_e = result
            fixed_ent = dict(ent)
            fixed_ent["span"] = [new_s, new_e]
            fixed_entities.append(fixed_ent)
            surviving_ids.add(ent["id"])
            stats["spans_fixed"] += 1
        else:
            stats["entities_dropped"] += 1

    # --- Fix relations ---
    fixed_relations = []
    for rel in relations:
        src = rel.get("source")
        tgt = rel.get("target")

        # Skip malformed relations (list IDs, missing fields, etc.)
        if isinstance(src, list) or isinstance(tgt, list) or src is None or tgt is None:
            stats["relations_dropped"] += 1
            continue

        # Drop relations referencing removed entities
        if src not in surviving_ids or tgt not in surviving_ids:
            stats["relations_dropped"] += 1
            continue

        original_type = rel.get("type", "")
        canonical_type = normalize_relation_type(original_type)
        fixed_rel = dict(rel)
        fixed_rel["type"] = canonical_type

        if canonical_type != original_type:
            stats["relations_remapped"] += 1

        fixed_relations.append(fixed_rel)

    # Build fixed sentence
    fixed = dict(sentence)
    fixed["entities"] = fixed_entities
    fixed["relations"] = fixed_relations

    return fixed, stats


def process_domain(input_path, output_path, dry_run=False):
    """Process a single domain's JSONL file."""
    domain_name = os.path.basename(os.path.dirname(input_path))
    print(f"\n{'='*60}")
    print(f"Processing: {domain_name}")
    print(f"  Input:  {input_path}")
    if not dry_run:
        print(f"  Output: {output_path}")
    print(f"{'='*60}")

    totals = Counter()
    sentences_kept = 0
    sentences_dropped = 0
    rel_type_before = Counter()
    rel_type_after = Counter()
    unmapped_types = Counter()

    output_lines = []

    with open(input_path) as f:
        for line_num, line in enumerate(f):
            try:
                sentence = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Track original relation types
            for r in sentence.get("relations", []):
                rel_type_before[r["type"]] += 1

            fixed, stats = fix_sentence(sentence)

            for key, val in stats.items():
                totals[key] += val

            # Track new relation types
            for r in fixed.get("relations", []):
                rel_type_after[r["type"]] += 1

            # Track unmapped (fell through to default)
            for r in sentence.get("relations", []):
                orig = r.get("type")
                if orig and orig not in RELATION_MAP and orig.lower() not in RELATION_MAP:
                    unmapped_types[orig] += 1

            # Drop sentences with 0 entities
            if len(fixed["entities"]) > 0:
                sentences_kept += 1
                output_lines.append(json.dumps(fixed, ensure_ascii=False))
            else:
                sentences_dropped += 1

    # Print stats
    total_ents = totals["entities_total"]
    print(f"\n  Sentences: {sentences_kept + sentences_dropped} total → {sentences_kept} kept, {sentences_dropped} dropped (0 entities)")
    print(f"\n  Entities: {total_ents} total")
    print(f"    Already correct: {totals['spans_already_correct']} ({100*totals['spans_already_correct']/max(total_ents,1):.1f}%)")
    print(f"    Spans fixed:     {totals['spans_fixed']} ({100*totals['spans_fixed']/max(total_ents,1):.1f}%)")
    print(f"    Dropped:         {totals['entities_dropped']} ({100*totals['entities_dropped']/max(total_ents,1):.1f}%)")

    total_rels = totals["relations_total"]
    print(f"\n  Relations: {total_rels} total")
    print(f"    Remapped:  {totals['relations_remapped']} ({100*totals['relations_remapped']/max(total_rels,1):.1f}%)")
    print(f"    Dropped:   {totals['relations_dropped']} ({100*totals['relations_dropped']/max(total_rels,1):.1f}%)")

    print(f"\n  Relation types before: {len(rel_type_before)} unique → after: {len(rel_type_after)} unique")
    print(f"  After distribution: {dict(rel_type_after)}")

    if unmapped_types:
        top_unmapped = unmapped_types.most_common(15)
        print(f"\n  Top unmapped relation types (defaulted to 'related-to'):")
        for t, c in top_unmapped:
            print(f"    {t}: {c}")

    # Write output
    if not dry_run:
        with open(output_path, 'w') as f:
            for line in output_lines:
                f.write(line + '\n')
        print(f"\n  ✓ Written to {output_path}")

    return totals


DOMAIN_CONFIGS = {
    "cs": {
        "input": "datasets/SPHERE/computer science/computer_science.jsonl",
        "output": "datasets/SPHERE/computer science/computer_science_cleaned.jsonl",
    },
    "physics": {
        "input": "datasets/SPHERE/physics/annotated_physics_sentences.jsonl",
        "output": "datasets/SPHERE/physics/annotated_physics_sentences_cleaned.jsonl",
    },
    "biology": {
        "input": "datasets/SPHERE/biology/annotated_biology_sentences.jsonl",
        "output": "datasets/SPHERE/biology/annotated_biology_sentences_cleaned.jsonl",
    },
    "materials": {
        "input": "datasets/SPHERE/material science/annotated_materials_sentences.jsonl",
        "output": "datasets/SPHERE/material science/annotated_materials_sentences_cleaned.jsonl",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Fix SPHERE dataset spans and relation types")
    parser.add_argument("--domain", choices=["cs", "physics", "biology", "materials", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, don't write output files")
    parser.add_argument("--base-dir", default=None, help="Base directory (default: auto-detect from script location)")
    args = parser.parse_args()

    # Auto-detect base directory
    if args.base_dir:
        base = args.base_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base = os.path.dirname(script_dir)  # ha-gnn/

    domains = list(DOMAIN_CONFIGS.keys()) if args.domain == "all" else [args.domain]

    grand_totals = Counter()
    for domain in domains:
        cfg = DOMAIN_CONFIGS[domain]
        input_path = os.path.join(base, cfg["input"])
        output_path = os.path.join(base, cfg["output"])

        if not os.path.exists(input_path):
            print(f"\n⚠ Skipping {domain}: {input_path} not found")
            continue

        totals = process_domain(input_path, output_path, dry_run=args.dry_run)
        for k, v in totals.items():
            grand_totals[k] += v

    # Grand summary
    if len(domains) > 1:
        print(f"\n{'='*60}")
        print("GRAND TOTALS")
        print(f"{'='*60}")
        total_ents = grand_totals["entities_total"]
        total_rels = grand_totals["relations_total"]
        print(f"  Entities: {total_ents}")
        print(f"    Already correct: {grand_totals['spans_already_correct']} ({100*grand_totals['spans_already_correct']/max(total_ents,1):.1f}%)")
        print(f"    Spans fixed:     {grand_totals['spans_fixed']} ({100*grand_totals['spans_fixed']/max(total_ents,1):.1f}%)")
        print(f"    Dropped:         {grand_totals['entities_dropped']} ({100*grand_totals['entities_dropped']/max(total_ents,1):.1f}%)")
        print(f"  Relations: {total_rels}")
        print(f"    Remapped:  {grand_totals['relations_remapped']} ({100*grand_totals['relations_remapped']/max(total_rels,1):.1f}%)")
        print(f"    Dropped:   {grand_totals['relations_dropped']} ({100*grand_totals['relations_dropped']/max(total_rels,1):.1f}%)")


if __name__ == "__main__":
    main()
