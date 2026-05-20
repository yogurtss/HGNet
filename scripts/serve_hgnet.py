#!/usr/bin/env python3
"""HTTP deployment server for HGNet NER and relation extraction."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import infer_hgnet as infer


class ChunkRequest(BaseModel):
    chunk: str = Field(..., min_length=1)


class EntityInput(BaseModel):
    id: Optional[str] = None
    text: Optional[str] = None
    type: str
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., gt=0)
    sentence_index: Optional[int] = None
    token_start: Optional[int] = Field(default=None, ge=0)
    token_end: Optional[int] = Field(default=None, ge=0)
    confidence: Optional[float] = None


class RelationsRequest(BaseModel):
    chunk: str = Field(..., min_length=1)
    entities: List[EntityInput]


class HGNetService:
    def __init__(self, args: argparse.Namespace):
        infer.load_ml_dependencies()
        self.device = infer.resolve_device(args.device)
        self.max_len = args.max_len
        self.model_name = args.model_name
        self.ner_checkpoint = args.ner_checkpoint
        self.re_checkpoint = args.re_checkpoint
        self.label_maps = args.label_maps

        self.nlp = infer.load_spacy_model()
        self.entity_types, self.entity_map, self.relation_map = infer.load_label_maps(args.label_maps)
        self.znerd_module = infer.import_module_from_path("hgnet_znerd_server", infer.ZNERD_PATH)
        self.hagnn_module = infer.import_module_from_path("hgnet_hagnn_server", infer.HAGNN_PATH)
        self.hagnn_module.nlp = self.nlp

        self.tag_to_id, self.id_to_tag = self.znerd_module.create_custom_label_maps(self.entity_types)
        self.tokenizer = infer.AutoTokenizer.from_pretrained(args.model_name)
        self.ner_model = infer.load_ner_model(
            self.znerd_module,
            args.model_name,
            args.ner_checkpoint,
            len(self.tag_to_id),
            self.device,
        )
        self.re_model = infer.load_re_model(
            self.hagnn_module,
            args.model_name,
            args.re_checkpoint,
            self.entity_map,
            self.relation_map,
            self.device,
        )

    def split(self, chunk: str) -> List[Dict]:
        sentences = infer.split_text(chunk, self.nlp)
        if not sentences:
            raise HTTPException(status_code=422, detail="Input chunk did not contain any tokens.")
        return sentences

    def predict_entities(self, chunk: str, sentences: Optional[List[Dict]] = None) -> List[Dict]:
        sentences = sentences or self.split(chunk)
        return infer.run_ner(
            text=chunk,
            sentences=sentences,
            tokenizer=self.tokenizer,
            model=self.ner_model,
            tag_to_id=self.tag_to_id,
            id_to_tag=self.id_to_tag,
            max_len=self.max_len,
            device=self.device,
        )

    def predict_relations(self, chunk: str, entities: List[Dict], sentences: Optional[List[Dict]] = None) -> List[Dict]:
        sentences = sentences or self.split(chunk)
        try:
            return infer.run_re(
                text=chunk,
                sentences=sentences,
                entities=entities,
                tokenizer=self.tokenizer,
                model=self.re_model,
                hagnn_module=self.hagnn_module,
                entity_map=self.entity_map,
                relation_map=self.relation_map,
                max_len=self.max_len,
                device=self.device,
            )
        except SystemExit as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def normalize_entities(self, chunk: str, sentences: List[Dict], entities: List[EntityInput]) -> List[Dict]:
        normalized = []
        for idx, entity in enumerate(entities):
            sentence_index, token_span = self._resolve_entity_token_span(
                sentences,
                entity.start_char,
                entity.end_char,
                entity.sentence_index,
                entity.token_start,
                entity.token_end,
            )
            if token_span is None or sentence_index is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"Entity {entity.id or idx} could not be aligned to the chunk tokens.",
                )
            token_start, token_end = token_span
            normalized.append(
                {
                    "id": entity.id or f"E{idx}",
                    "text": entity.text if entity.text is not None else chunk[entity.start_char : entity.end_char],
                    "type": entity.type,
                    "start_char": entity.start_char,
                    "end_char": entity.end_char,
                    "sentence_index": sentence_index,
                    "token_start": token_start,
                    "token_end": token_end,
                    "confidence": entity.confidence,
                }
            )
        return normalized

    def _resolve_entity_token_span(
        self,
        sentences: List[Dict],
        start_char: int,
        end_char: int,
        sentence_index: Optional[int],
        token_start: Optional[int],
        token_end: Optional[int],
    ) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
        if end_char <= start_char:
            raise HTTPException(status_code=422, detail="Entity end_char must be greater than start_char.")

        if sentence_index is not None and token_start is not None and token_end is not None:
            if sentence_index >= len(sentences):
                return None, None
            if token_end < token_start or token_end >= len(sentences[sentence_index]["tokens"]):
                return None, None
            return sentence_index, (token_start, token_end)

        candidate_indices = range(len(sentences)) if sentence_index is None else [sentence_index]
        for sent_idx in candidate_indices:
            if sent_idx >= len(sentences):
                continue
            offsets = sentences[sent_idx]["offsets"]
            token_span = infer.token_span_for_chars(offsets, start_char, end_char)
            if token_span is not None:
                return sent_idx, token_span
        return None, None


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="HGNet Inference Server", version="1.0.0")
    service = HGNetService(args)
    app.state.hgnet = service

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "device": str(service.device),
            "model_name": service.model_name,
        }

    @app.post("/entities")
    def entities(request: ChunkRequest):
        sentences = service.split(request.chunk)
        predicted_entities = service.predict_entities(request.chunk, sentences)
        return {
            "chunk": request.chunk,
            "entities": predicted_entities,
            "relations": [],
        }

    @app.post("/extract")
    def extract(request: ChunkRequest):
        sentences = service.split(request.chunk)
        predicted_entities = service.predict_entities(request.chunk, sentences)
        predicted_relations = service.predict_relations(request.chunk, predicted_entities, sentences)
        return {
            "chunk": request.chunk,
            "entities": predicted_entities,
            "relations": predicted_relations,
        }

    @app.post("/relations")
    def relations(request: RelationsRequest):
        sentences = service.split(request.chunk)
        normalized_entities = service.normalize_entities(request.chunk, sentences, request.entities)
        predicted_relations = service.predict_relations(request.chunk, normalized_entities, sentences)
        return {
            "chunk": request.chunk,
            "entities": normalized_entities,
            "relations": predicted_relations,
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve trained HGNet NER and RE models over HTTP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ner_checkpoint", default=infer.DEFAULT_NER_CHECKPOINT)
    parser.add_argument("--re_checkpoint", default=infer.DEFAULT_RE_CHECKPOINT)
    parser.add_argument("--label_maps", default=infer.DEFAULT_LABEL_MAPS)
    parser.add_argument("--model_name", default=infer.DEFAULT_MODEL_NAME)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    for path_name in ("ner_checkpoint", "re_checkpoint", "label_maps"):
        path = getattr(args, path_name)
        if not os.path.exists(path):
            raise SystemExit(f"{path_name} not found: {path}")


def main() -> None:
    args = parse_args()
    validate_paths(args)
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
