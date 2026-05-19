#!/usr/bin/env python3
"""Unified inference CLI for HGNet entity and relation extraction."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LABEL_MAPS = os.path.join(REPO_ROOT, "datasets", "SPHERE", "merged", "label_maps.json")
DEFAULT_NER_CHECKPOINT = os.path.join(REPO_ROOT, "z_nerd_sphere_model.pth")
DEFAULT_RE_CHECKPOINT = os.path.join(REPO_ROOT, "model_cache", "ha_gnn_best_model.pth")
DEFAULT_MODEL_NAME = "allenai/scibert_scivocab_uncased"
ZNERD_PATH = os.path.join(REPO_ROOT, "NER", "proposed_solution", "Z-NERD", "znerd.py")
HAGNN_PATH = os.path.join(REPO_ROOT, "RE", "proposed_solutions", "HA-GNN", "ha-gnn.py")
NO_RELATION = "No-Relation"
torch = None
F = None
AutoTokenizer = None


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_ml_dependencies():
    global torch, F, AutoTokenizer
    if torch is not None:
        return torch, F, AutoTokenizer
    try:
        import torch as torch_module
        import torch.nn.functional as functional_module
        from transformers import AutoTokenizer as tokenizer_cls
    except ImportError as exc:
        fail(f"Missing ML dependency: {exc}. Install project dependencies with: pip install -r requirements.txt")
    torch = torch_module
    F = functional_module
    AutoTokenizer = tokenizer_cls
    return torch, F, AutoTokenizer


def import_module_from_path(module_name: str, path: str):
    if not os.path.exists(path):
        fail(f"Cannot find module file: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        fail(f"Cannot import module from: {path}")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
    return module


def load_label_maps(label_maps_path: str):
    if not os.path.exists(label_maps_path):
        fail(f"Label map file not found: {label_maps_path}")
    with open(label_maps_path, "r", encoding="utf-8") as f:
        label_maps = json.load(f)

    if "entity_map" in label_maps:
        entity_map = {label: int(idx) for label, idx in label_maps["entity_map"].items()}
        entity_types = sorted(entity_map.keys())
    else:
        entity_types = sorted(label_maps.get("entity_types", []))
        entity_map = {label: idx for idx, label in enumerate(entity_types)}

    if "relation_map" in label_maps:
        relation_map = {label: int(idx) for label, idx in label_maps["relation_map"].items()}
    else:
        relation_types = set(label_maps.get("relation_types", []))
        relation_types.add(NO_RELATION)
        relation_map = {label: idx for idx, label in enumerate(sorted(relation_types))}

    if NO_RELATION not in relation_map:
        relation_map[NO_RELATION] = len(relation_map)
    return entity_types, entity_map, relation_map


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        fail("CUDA was requested but is not available.")
    if device_arg == "mps" and not torch.backends.mps.is_available():
        fail("MPS was requested but is not available.")
    return torch.device(device_arg)


def load_spacy_model():
    try:
        import spacy
    except ImportError as exc:
        fail(f"spaCy is not installed: {exc}")
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        fail("spaCy model 'en_core_web_sm' is missing. Run: python -m spacy download en_core_web_sm")


def split_text(text: str, nlp) -> List[Dict]:
    doc = nlp(text)
    spans = list(doc.sents) if doc.has_annotation("SENT_START") else [doc[:]]
    if not spans:
        spans = [doc[:]]

    sentences = []
    for sent in spans:
        tokens = [token.text for token in sent if not token.is_space]
        offsets = [(token.idx, token.idx + len(token.text)) for token in sent if not token.is_space]
        if not tokens:
            continue
        sentences.append(
            {
                "tokens": tokens,
                "offsets": offsets,
                "start_char": offsets[0][0],
                "end_char": offsets[-1][1],
                "text": text[offsets[0][0] : offsets[-1][1]],
            }
        )
    if not sentences and text.strip():
        stripped_start = len(text) - len(text.lstrip())
        stripped = text.strip()
        sentences.append(
            {
                "tokens": stripped.split(),
                "offsets": whitespace_token_offsets(text),
                "start_char": stripped_start,
                "end_char": stripped_start + len(stripped),
                "text": stripped,
            }
        )
    return sentences


def whitespace_token_offsets(text: str) -> List[Tuple[int, int]]:
    offsets = []
    start = None
    for idx, char in enumerate(text):
        if char.isspace():
            if start is not None:
                offsets.append((start, idx))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        offsets.append((start, len(text)))
    return offsets


def load_state_dict(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        fail(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def load_ner_model(znerd_module, model_name: str, checkpoint_path: str, num_labels: int, device: torch.device):
    state_dict = load_state_dict(checkpoint_path, device)
    model = znerd_module.ZNERD_Tagger_Model(model_name, num_labels=num_labels)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_re_model(hagnn_module, model_name: str, checkpoint_path: str, entity_map: Dict[str, int], relation_map: Dict[str, int], device: torch.device):
    state_dict = load_state_dict(checkpoint_path, torch.device("cpu"))
    use_caf_loss = isinstance(state_dict, dict) and "w_abs" in state_dict
    model = hagnn_module.HA_GNN(
        model_name=model_name,
        hidden_dim=256,
        num_entity_types=len(entity_map),
        num_relations=len(relation_map),
        num_layers=3,
        dropout=0.2,
        cache_dir=os.path.join(REPO_ROOT, "model_cache"),
        use_caf_loss=use_caf_loss,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def token_span_for_chars(token_offsets: Sequence[Tuple[int, int]], start_char: int, end_char: int) -> Optional[Tuple[int, int]]:
    indices = [
        idx
        for idx, (tok_start, tok_end) in enumerate(token_offsets)
        if max(tok_start, start_char) < min(tok_end, end_char)
    ]
    if not indices:
        return None
    return indices[0], indices[-1]


def close_entity(active, entities, text, token_offsets, probs):
    if active is None:
        return
    token_span = token_span_for_chars(token_offsets, active["start_char"], active["end_char"])
    if token_span is None:
        return
    entity_id = f"E{len(entities)}"
    prob_values = probs[active["start_token_idx"] : active["end_token_idx"] + 1]
    confidence = float(sum(prob_values) / max(len(prob_values), 1))
    entities.append(
        {
            "id": entity_id,
            "text": text[active["start_char"] : active["end_char"]],
            "type": active["type"],
            "start_char": active["start_char"],
            "end_char": active["end_char"],
            "sentence_index": active["sentence_index"],
            "token_start": token_span[0],
            "token_end": token_span[1],
            "confidence": confidence,
        }
    )


def decode_bio_entities(
    text: str,
    sentence_index: int,
    sentence_start: int,
    token_offsets: Sequence[Tuple[int, int]],
    subword_offsets: Sequence[Tuple[int, int]],
    pred_tags: Sequence[str],
    pred_probs: Sequence[float],
) -> List[Dict]:
    entities = []
    active = None

    for token_idx, (offset, tag, prob) in enumerate(zip(subword_offsets, pred_tags, pred_probs)):
        start, end = offset
        if start == end == 0:
            continue
        global_start = sentence_start + start
        global_end = sentence_start + end

        if tag == "O" or tag == "":
            close_entity(active, entities, text, token_offsets, pred_probs)
            active = None
            continue

        if "-" not in tag:
            close_entity(active, entities, text, token_offsets, pred_probs)
            active = None
            continue

        prefix, entity_type = tag.split("-", 1)
        starts_new = prefix == "B" or active is None or active["type"] != entity_type
        if starts_new:
            close_entity(active, entities, text, token_offsets, pred_probs)
            active = {
                "type": entity_type,
                "start_char": global_start,
                "end_char": global_end,
                "start_token_idx": token_idx,
                "end_token_idx": token_idx,
                "sentence_index": sentence_index,
            }
        else:
            active["end_char"] = global_end
            active["end_token_idx"] = token_idx

    close_entity(active, entities, text, token_offsets, pred_probs)
    return entities


def run_ner(text: str, sentences: List[Dict], tokenizer, model, tag_to_id: Dict[str, int], id_to_tag: Dict[int, str], max_len: int, device: torch.device):
    all_entities = []
    for sent_idx, sentence in enumerate(sentences):
        encoded = tokenizer(
            sentence["text"],
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")[0].tolist()]
        inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
            probs = F.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)[0].tolist()
            pred_probs = torch.max(probs, dim=-1).values[0].tolist()

        pred_tags = [id_to_tag.get(pred_id, "O") for pred_id in pred_ids]
        sentence_entities = decode_bio_entities(
            text=text,
            sentence_index=sent_idx,
            sentence_start=sentence["start_char"],
            token_offsets=sentence["offsets"],
            subword_offsets=offsets,
            pred_tags=pred_tags,
            pred_probs=pred_probs,
        )
        all_entities.extend(sentence_entities)

    for idx, entity in enumerate(all_entities):
        entity["id"] = f"E{idx}"
    return all_entities


def build_re_doc(sentences: List[Dict], entities: List[Dict], entity_map: Dict[str, int]):
    ner_by_sentence = [[] for _ in sentences]
    sentence_token_offsets = []
    running_offset = 0
    for sentence in sentences:
        sentence_token_offsets.append(running_offset)
        running_offset += len(sentence["tokens"])

    node_entity_ids = []
    for entity in entities:
        sent_idx = entity["sentence_index"]
        if sent_idx >= len(sentences) or entity["type"] not in entity_map:
            continue
        global_start = sentence_token_offsets[sent_idx] + entity["token_start"]
        global_end = sentence_token_offsets[sent_idx] + entity["token_end"]
        ner_by_sentence[sent_idx].append([global_start, global_end, entity["type"]])
        node_entity_ids.append(entity["id"])

    return (
        {
            "doc_id": "hgnet-inference",
            "sentences": [sentence["tokens"] for sentence in sentences],
            "ner": ner_by_sentence,
            "relations": [[] for _ in sentences],
        },
        node_entity_ids,
    )


def filter_graph_token_edges(graph, final_num_tokens: int):
    for edge_type in list(graph.edge_index_dict.keys()):
        if "token" not in edge_type:
            continue
        edge_index = graph[edge_type].edge_index
        if edge_index.numel() == 0:
            continue
        src_node_type, _, dst_node_type = edge_type
        src_mask = edge_index[0] < graph.num_nodes_dict[src_node_type]
        dst_mask = edge_index[1] < graph.num_nodes_dict[dst_node_type]
        if src_node_type == "token":
            src_mask &= edge_index[0] < final_num_tokens
        if dst_node_type == "token":
            dst_mask &= edge_index[1] < final_num_tokens
        graph[edge_type].edge_index = edge_index[:, src_mask & dst_mask]


def run_re(text: str, sentences: List[Dict], entities: List[Dict], tokenizer, model, hagnn_module, entity_map: Dict[str, int], relation_map: Dict[str, int], max_len: int, device: torch.device):
    if len(entities) < 2:
        return []

    re_doc, node_entity_ids = build_re_doc(sentences, entities, entity_map)
    if not node_entity_ids:
        return []

    try:
        graph = hagnn_module.build_graph_from_doc(re_doc, entity_map)
    except Exception as exc:
        fail(f"Failed to build HA-GNN graph for inference: {exc}")

    if graph.num_nodes == 0 or "entity" not in graph.node_types or graph["entity"].num_nodes < 2:
        return []

    graph = graph.to(device)
    flat_tokens = [token for sentence in re_doc["sentences"] for token in sentence]
    if not flat_tokens:
        return []

    token_inputs = tokenizer(
        flat_tokens,
        return_tensors="pt",
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_len,
    ).to(device)
    filter_graph_token_edges(graph, token_inputs["input_ids"].shape[1])

    rev_relation_map = {idx: label for label, idx in relation_map.items()}
    no_relation_idx = relation_map[NO_RELATION]
    relations = []
    with torch.no_grad():
        logits, pair_indices, _, _ = model(graph, token_inputs, relation_map)
        if logits.size(0) == 0:
            return []
        probs = F.softmax(logits, dim=-1)
        confidences, pred_ids = torch.max(probs, dim=-1)

    for pair, pred_id, confidence in zip(pair_indices.cpu().tolist(), pred_ids.cpu().tolist(), confidences.cpu().tolist()):
        if pred_id == no_relation_idx:
            continue
        head_idx, tail_idx = pair
        if head_idx >= len(node_entity_ids) or tail_idx >= len(node_entity_ids):
            continue
        relations.append(
            {
                "type": rev_relation_map.get(pred_id, f"relation_{pred_id}"),
                "head": node_entity_ids[head_idx],
                "tail": node_entity_ids[tail_idx],
                "confidence": float(confidence),
            }
        )
    return relations


def parse_args():
    parser = argparse.ArgumentParser(description="Run HGNet entity and relation extraction on a text chunk.")
    parser.add_argument("--task", choices=["ner", "re", "joint"], required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Input text chunk.")
    input_group.add_argument("--input_file", type=str, help="Path to a UTF-8 text file.")
    parser.add_argument("--ner_checkpoint", type=str, default=DEFAULT_NER_CHECKPOINT)
    parser.add_argument("--re_checkpoint", type=str, default=DEFAULT_RE_CHECKPOINT)
    parser.add_argument("--label_maps", type=str, default=DEFAULT_LABEL_MAPS)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output", type=str, help="Optional path to write the JSON result.")
    return parser.parse_args()


def read_input_text(args) -> str:
    if args.text is not None:
        return args.text
    if not os.path.exists(args.input_file):
        fail(f"Input file not found: {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as f:
        return f.read()


def validate_runtime_paths(args) -> None:
    if not os.path.exists(args.label_maps):
        fail(f"Label map file not found: {args.label_maps}")
    if not os.path.exists(args.ner_checkpoint):
        fail(f"Checkpoint not found: {args.ner_checkpoint}")
    if args.task in ("re", "joint") and not os.path.exists(args.re_checkpoint):
        fail(f"Checkpoint not found: {args.re_checkpoint}")


def main():
    args = parse_args()
    text = read_input_text(args)
    validate_runtime_paths(args)
    load_ml_dependencies()
    device = resolve_device(args.device)
    nlp = load_spacy_model()
    sentences = split_text(text, nlp)
    if not sentences:
        fail("Input text did not contain any tokens.")

    entity_types, entity_map, relation_map = load_label_maps(args.label_maps)
    znerd_module = import_module_from_path("hgnet_znerd", ZNERD_PATH)
    tag_to_id, id_to_tag = znerd_module.create_custom_label_maps(entity_types)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ner_model = load_ner_model(znerd_module, args.model_name, args.ner_checkpoint, len(tag_to_id), device)
    entities = run_ner(text, sentences, tokenizer, ner_model, tag_to_id, id_to_tag, args.max_len, device)

    relations = []
    if args.task in ("re", "joint"):
        hagnn_module = import_module_from_path("hgnet_hagnn", HAGNN_PATH)
        hagnn_module.nlp = nlp
        re_model = load_re_model(hagnn_module, args.model_name, args.re_checkpoint, entity_map, relation_map, device)
        relations = run_re(text, sentences, entities, tokenizer, re_model, hagnn_module, entity_map, relation_map, args.max_len, device)

    output = {
        "text": text,
        "entities": entities,
        "relations": relations,
        "metadata": {
            "task": args.task,
            "model_name": args.model_name,
            "ner_checkpoint": args.ner_checkpoint,
            "re_checkpoint": args.re_checkpoint if args.task in ("re", "joint") else None,
            "label_maps": args.label_maps,
            "device": str(device),
        },
    }

    output_json = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
            f.write("\n")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
