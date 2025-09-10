import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv
from tqdm import tqdm
import spacy
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import random
import argparse
from collections import Counter
import pandas as pd
import networkx as nx
import re
import matplotlib.pyplot as plt

# --- Configuration and Global Settings ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Data Loading and Preprocessing ---
class SciERDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_label_maps(data_samples):
    entity_types, relation_types = set(), {"No-Relation"}
    for doc in data_samples:
        for sent_ner in doc['ner']:
            for entity in sent_ner:
                entity_types.add(entity[2])
        for sent_relations in doc['relations']:
            for relation in sent_relations:
                relation_types.add(relation[4])
    entity_map = {label: i for i, label in enumerate(sorted(list(entity_types)))}
    relation_map = {label: i for i, label in enumerate(sorted(list(relation_types)))}
    return entity_map, relation_map

# --- Graph Construction ---
nlp = None
def init_spacy():
    global nlp
    if nlp is None:
        print("Loading spaCy model...")
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading 'en_core_web_sm' model...")
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

def calculate_hierarchical_separation_loss(A_parent, num_entities):
    """
    Hierarchical Separation Loss: Penalizes shortcuts in hierarchy
    L_separation = Σ(A²)_uw · (A)_uw - penalizes when both 1-step and 2-step paths are strong
    """
    A_squared = torch.matmul(A_parent, A_parent)
    separation_loss = torch.sum(A_squared * A_parent)
    return separation_loss

def calculate_acyclic_loss(A_parent, num_entities):
    """
    Differentiable acyclicity constraint using matrix exponential trace
    Based on: tr(exp(A ∘ A)) - d where A is the parent probability matrix
    """
    A_hadamard_sq = A_parent * A_parent  # Hadamard product
    try:
        matrix_exp = torch.matrix_exp(A_hadamard_sq)
        acyclic_loss = torch.trace(matrix_exp) - num_entities
        return F.relu(acyclic_loss)  # Only penalize when > 0
    except Exception:
        return torch.sum(A_hadamard_sq) * 0.1

def calculate_caf_loss(entity_reps, pair_indices, labels, relation_map, w_abs, caf_gamma=1.0, caf_delta=1.0):
    """
    Continuum Abstraction Field Loss with three components:
    1. Pairwise Ranking Loss
    2. Field Anchoring Loss  
    """
    part_of_idx = relation_map.get("Part-Of")
    if part_of_idx is None:
        return torch.tensor(0.0, device=entity_reps.device)

    # Find true Part-Of relations
    true_part_of_mask = (labels == part_of_idx)
    if not torch.any(true_part_of_mask):
        return torch.tensor(0.0, device=entity_reps.device)

    part_of_pairs = pair_indices[true_part_of_mask]
    if len(part_of_pairs) == 0:
        return torch.tensor(0.0, device=entity_reps.device)
    
    try:
        # 1. Pairwise Ranking Loss
        h_c = entity_reps[part_of_pairs[:, 0]]  # children (heads in Part-Of)
        h_p = entity_reps[part_of_pairs[:, 1]]  # parents (tails in Part-Of)
        
        score_diff = torch.matmul(h_c - h_p, w_abs)  
        l_ranking = torch.mean(F.relu(score_diff + caf_delta))
        
        # 2. Field Anchoring Loss
        G = nx.DiGraph(part_of_pairs.cpu().numpy().tolist())
        all_nodes = list(G.nodes())
        
        if len(all_nodes) == 0:
            return torch.tensor(0.0, device=entity_reps.device)
        
        roots = [n for n in all_nodes if G.in_degree(n) == 0]
        leaves = [n for n in all_nodes if G.out_degree(n) == 0]
        
        l_anchor = torch.tensor(0.0, device=entity_reps.device)
        
        if roots and len(roots) < entity_reps.size(0):
            root_reps = entity_reps[torch.tensor(roots, device=entity_reps.device)]
            root_scores = torch.matmul(root_reps, w_abs)
            l_anchor += torch.mean((root_scores - 1.0) ** 2)
        
        if leaves and len(leaves) < entity_reps.size(0):
            leaf_reps = entity_reps[torch.tensor(leaves, device=entity_reps.device)]
            leaf_scores = torch.matmul(leaf_reps, w_abs)
            l_anchor += torch.mean((leaf_scores - 0.0) ** 2)
        
        caf_loss = l_ranking + (caf_gamma * l_anchor)
        
        if torch.isnan(caf_loss) or torch.isinf(caf_loss):
            return torch.tensor(0.0, device=entity_reps.device)
            
        return caf_loss
            
    except Exception as e:
        return torch.tensor(0.0, device=entity_reps.device)

def build_graph_from_doc(doc, entity_map):
    init_spacy()
    graph = HeteroData()
    all_tokens = [token for sent in doc['sentences'] for token in sent]
    if not all_tokens: return graph
    graph['token'].x = torch.arange(len(all_tokens)).unsqueeze(1).float()

    entity_nodes, entity_to_node_idx, entity_idx_counter = [], {}, 0
    entity_to_sentence_map = []
    entity_type_ids = []
    entity_start_tokens = []
    entities_by_sent = [[] for _ in range(len(doc['sentences']))]
    entity_texts = []
    entity_token_spans = []

    token_offset_by_sent = [0]
    for sent_idx, sent in enumerate(doc['sentences']):
        token_offset_by_sent.append(token_offset_by_sent[-1] + len(sent))

    for i, sent_ner in enumerate(doc['ner']):
        sent_start_offset = token_offset_by_sent[i]
        for entity in sent_ner:
            start, end, entity_type = entity
            entity_id = f"{i}-{start}-{end}"
            node_idx = entity_idx_counter
            
            entity_nodes.append([entity_map[entity_type]])
            entity_to_node_idx[entity_id] = node_idx
            entity_to_sentence_map.append(i)
            entity_type_ids.append(entity_map[entity_type])
            entity_start_tokens.append(start)
            
            if start <= end and end < len(all_tokens):
                entity_text = " ".join(all_tokens[start:end+1])
            else:
                entity_text = f"entity_{node_idx}"
            
            entity_texts.append(entity_text)
            entity_token_spans.append((start, end))

            local_start = start - sent_start_offset
            local_end = end - sent_start_offset
            entities_by_sent[i].append({
                'node_idx': node_idx,  
                'start': local_start,
                'end': local_end,
                'global_start': start,
                'global_end': end
            })
            
            entity_idx_counter += 1

    if not entity_nodes: return HeteroData()
    graph['entity'].x = torch.tensor(entity_nodes, dtype=torch.long)
    graph['entity'].type_id = torch.tensor(entity_type_ids, dtype=torch.long)
    graph.entity_to_sentence_map = torch.tensor(entity_to_sentence_map, dtype=torch.long)
    graph['entity'].start_token = torch.tensor(entity_start_tokens, dtype=torch.long)
    graph.entity_texts = entity_texts
    graph.entity_token_spans = torch.tensor(entity_token_spans, dtype=torch.long)
    graph['sentence'].x = torch.arange(len(doc['sentences'])).unsqueeze(1).float()

    edge_builders = {k: [] for k in ['seq_src', 'seq_dst', 'syn_src', 'syn_dst', 'ent_mem_src', 'ent_mem_dst', 'tok_sent_src', 'tok_sent_dst']}
    sdp_paths = {}
    
    spacy_docs_by_sent = [nlp(" ".join(sent)) for sent in doc['sentences']]
    graph.spacy_docs_by_sent = spacy_docs_by_sent
    graph.sentences_tokens = doc['sentences']

    token_offset = 0
    for sent_idx, sent in enumerate(doc['sentences']):
        spacy_doc = spacy_docs_by_sent[sent_idx]
        if len(spacy_doc) != len(sent):
            token_offset += len(sent)
            continue
        dep_graph = nx.Graph()
        for token in spacy_doc:
            if token.head.i != token.i:
                dep_graph.add_edge(token.i, token.head.i)

        sent_entities = entities_by_sent[sent_idx]
        if len(sent_entities) >= 2:
            for i in range(len(sent_entities)):
                for j in range(i + 1, len(sent_entities)):
                    ent1, ent2 = sent_entities[i], sent_entities[j]
                    try:
                        path = nx.shortest_path(dep_graph, source=ent1['start'], target=ent2['start'])
                        global_path = [p + token_offset for p in path]
                        sdp_paths[(ent1['node_idx'], ent2['node_idx'])] = global_path
                        sdp_paths[(ent2['node_idx'], ent1['node_idx'])] = global_path
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass

        for i, token in enumerate(spacy_doc):
            global_idx = token_offset + i
            if i < len(sent) - 1:
                edge_builders['seq_src'].append(global_idx)
                edge_builders['seq_dst'].append(global_idx + 1)
            if token.head.i != token.i:
                edge_builders['syn_src'].append(global_idx)
                edge_builders['syn_dst'].append(token_offset + token.head.i)
            edge_builders['tok_sent_src'].append(global_idx)
            edge_builders['tok_sent_dst'].append(sent_idx)
        token_offset += len(sent)

    for entity_idx, (start, end) in enumerate(entity_token_spans):
        for token_idx in range(start, end+1):
            if token_idx < len(all_tokens):
                edge_builders['ent_mem_src'].append(entity_idx)
                edge_builders['ent_mem_dst'].append(token_idx)

    if edge_builders['seq_src']: graph['token', 'seq', 'token'].edge_index = torch.tensor([edge_builders['seq_src'] + edge_builders['seq_dst'], edge_builders['seq_dst'] + edge_builders['seq_src']], dtype=torch.long)
    if edge_builders['syn_src']: graph['token', 'syn', 'token'].edge_index = torch.tensor([edge_builders['syn_src'] + edge_builders['syn_dst'], edge_builders['syn_dst'] + edge_builders['syn_src']], dtype=torch.long)
    if edge_builders['ent_mem_src']: graph['entity', 'has', 'token'].edge_index = torch.tensor([edge_builders['ent_mem_src'], edge_builders['ent_mem_dst']], dtype=torch.long)
    if edge_builders['tok_sent_src']: graph['token', 'in', 'sentence'].edge_index = torch.tensor([edge_builders['tok_sent_src'], edge_builders['tok_sent_dst']], dtype=torch.long)
    graph.entity_to_node_idx = entity_to_node_idx
    graph.relations = doc['relations']
    graph.sdp_paths = sdp_paths
    return graph

def calculate_class_weights(dataloader_data, relation_map):
    counts = Counter()
    no_relation_label = relation_map["No-Relation"]
    for doc in dataloader_data:
        entities_in_doc = sum([len(ner) for ner in doc['ner']])
        total_pairs = entities_in_doc * (entities_in_doc - 1)
        true_rels = 0
        for sent_relations in doc['relations']:
            for rel in sent_relations:
                if rel[4] in relation_map:
                    counts[relation_map[rel[4]]] += 1
                    true_rels += 1
        counts[no_relation_label] += total_pairs - true_rels

    total_samples = sum(counts.values()) if sum(counts.values()) > 0 else 1
    alpha = torch.tensor([counts.get(i, 0) / total_samples for i in range(len(relation_map))])
    return 1 - alpha

# --- Model and Loss Function ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt)**self.gamma * BCE_loss
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            alpha_t = self.alpha[targets.data.view(-1)]
            F_loss = alpha_t * F_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class HeteroGNNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_types):
        super().__init__()
        self.convs = nn.ModuleDict({str(et): GATConv((-1, -1), hidden_dim, add_self_loops=False) for et in edge_types})
        self.bns = nn.ModuleDict({ntype: nn.BatchNorm1d(hidden_dim) for ntype in ['token', 'entity', 'sentence']})

    def forward(self, x_dict, edge_index_dict):
        out_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if x_dict[src_type].size(0) == 0 or edge_index.numel() == 0:
                continue
            out = self.convs[str(edge_type)]((x_dict[src_type], x_dict[dst_type]), edge_index)
            out_dict[dst_type] = out + out_dict.get(dst_type, 0)
        for ntype, x in out_dict.items():
            if x_dict[ntype].size(0) > 0 and x.size(0) > 0:
                x_dict[ntype] = self.bns[ntype](x) + x_dict[ntype]
        return x_dict

class LatentRelationPredictor(nn.Module):
    def __init__(self, input_dim, num_relations):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_relations)
        )
    def forward(self, head_reps, tail_reps):
        concatenated_reps = torch.cat([head_reps, tail_reps], dim=-1)
        logits = self.mlp(concatenated_reps)
        return F.softmax(logits, dim=-1)

class HierarchicalProbabilisticMessagePassing(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_up = nn.Linear(hidden_dim, hidden_dim)
        self.W_down = nn.Linear(hidden_dim, hidden_dim)
        self.W_peer = nn.Linear(hidden_dim, hidden_dim)
        self.UpdateMLP = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU()
        )

    def forward(self, entity_reps, P_uv_all, pair_indices, relation_map):
        num_entities = entity_reps.size(0)
        
        # Create full probability matrices
        P_parent = torch.zeros(num_entities, num_entities, device=entity_reps.device)
        P_peer = torch.zeros(num_entities, num_entities, device=entity_reps.device)
        
        parent_rel_idx = relation_map.get('Part-Of')
        peer_rel_idx = relation_map.get('Peer-Of')

        if parent_rel_idx is not None:
            P_parent[pair_indices[:, 0], pair_indices[:, 1]] = P_uv_all[:, parent_rel_idx]
        if peer_rel_idx is not None:
            P_peer[pair_indices[:, 0], pair_indices[:, 1]] = P_uv_all[:, peer_rel_idx]

        m_parents = torch.matmul(P_parent.T, self.W_up(entity_reps))
        m_children = torch.matmul(P_parent, self.W_down(entity_reps))
        m_peers = torch.matmul(P_peer, self.W_peer(entity_reps))

        updated_reps = self.UpdateMLP(torch.cat([entity_reps, m_parents, m_children, m_peers], dim=-1))
        return updated_reps

class HA_GNN(nn.Module):
    def __init__(self, model_name, hidden_dim, num_entity_types, num_relations, num_layers=2, dropout=0.5, cache_dir='./', use_caf_loss=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        encoder_dim = self.encoder.config.hidden_size
        self.hidden_dim = hidden_dim
        
        self.type_emb_dim = 50
        self.dist_emb_dim = 50
        self.num_dist_bins = 10
        self.num_nlp_cues = 8

        self.entity_emb = nn.Embedding(num_entity_types, encoder_dim)
        node_types = ['token', 'entity', 'sentence']
        edge_types = [('token', 'seq', 'token'), ('token', 'syn', 'token'), ('entity', 'has', 'token'), ('token', 'in', 'sentence')]
        all_edge_types = edge_types + [(d, f'rev_{r}', s) for s, r, d in edge_types]
        self.lin_dict = nn.ModuleDict({nt: nn.Linear(encoder_dim, hidden_dim) for nt in node_types})
        self.gnn_layers = nn.ModuleList([HeteroGNNLayer(hidden_dim, all_edge_types) for _ in range(num_layers)])

        self.latent_relation_predictor = LatentRelationPredictor(hidden_dim, num_relations)
        self.hierarchical_message_passing = HierarchicalProbabilisticMessagePassing(hidden_dim)

        self.path_encoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.sdp_encoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.final_type_emb = nn.Embedding(num_entity_types, self.type_emb_dim)
        self.distance_emb = nn.Embedding(self.num_dist_bins, self.dist_emb_dim)

        classifier_input_dim = (self.hidden_dim * 2) + self.hidden_dim + self.hidden_dim + (self.type_emb_dim * 2) + self.dist_emb_dim + self.num_nlp_cues
        self.relation_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, num_relations)
        )
        self.use_caf_loss = use_caf_loss
        if use_caf_loss:
            self.w_abs = nn.Parameter(torch.randn(hidden_dim))
        else:
            self.w_abs = None

    def _bin_distance(self, distances):
        bins = torch.tensor([-30, -10, -5, -2, 0, 2, 5, 10, 30], device=distances.device)
        binned_distances = torch.bucketize(distances, bins)
        return binned_distances
    
    def _extract_hierarchical_cues(self, head_idx, tail_idx, graph):
        cues = [0.0] * self.num_nlp_cues
        head_sent_idx = graph.entity_to_sentence_map[head_idx].item()
        tail_sent_idx = graph.entity_to_sentence_map[tail_idx].item()
        if head_sent_idx != tail_sent_idx: return cues
        sent_idx = head_sent_idx
        sent_tokens = graph.sentences_tokens[sent_idx]
        h_span = graph.entity_token_spans[head_idx]
        t_span = graph.entity_token_spans[tail_idx]
        sent_start_offset = sum(len(graph.sentences_tokens[i]) for i in range(sent_idx))
        h_local_start, h_local_end = h_span[0] - sent_start_offset, h_span[1] - sent_start_offset
        t_local_start, t_local_end = t_span[0] - sent_start_offset, t_span[1] - sent_start_offset
        if h_local_end >= len(sent_tokens) or t_local_end >= len(sent_tokens) or h_local_start < 0 or t_local_start < 0: return cues
        h_text = " ".join(sent_tokens[h_local_start:h_local_end+1]).lower()
        t_text = " ".join(sent_tokens[t_local_start:t_local_end+1]).lower()
        if h_local_start <= t_local_end and t_local_start <= h_local_end: text_between = ""
        else:
            start, end = min(h_local_end+1, t_local_end+1), max(h_local_start, t_local_start)
            if start < end and start < len(sent_tokens) and end <= len(sent_tokens): text_between = " ".join(sent_tokens[start:end]).lower()
            else: text_between = ""
        hearst_pattern = r'\b(such as|including|especially)\b'
        subclass_keyword = r'\b(is a|are a|a type of|a kind of|an instance of)\b'
        part_of_keyword = r'\b(part of|component of|member of|module of)\b'
        part_of_reverse_keyword = r'\b(consists of|contains|includes|composed of|has a)\b'
        if h_local_start > t_local_end:
            if re.search(hearst_pattern, text_between): cues[0] = 1.0
            if re.search(subclass_keyword, text_between): cues[1] = 1.0
            if re.search(part_of_reverse_keyword, text_between): cues[5] = 1.0
        else:
            if re.search(part_of_keyword, text_between): cues[4] = 1.0
        if h_text in t_text and h_text != t_text: cues[3] = 1.0
        if t_text in h_text and h_text != t_text: cues[6] = 1.0
        if h_text == t_text: cues[7] = 1.0
        return cues

    def forward(self, graph, token_inputs, relation_map):
        for src, rel, dst in list(graph.edge_types):
            graph[dst, f'rev_{rel}', src].edge_index = graph[src, rel, dst].edge_index.flip([0])
        
        token_outputs = self.encoder(**token_inputs).last_hidden_state
        graph['token'].x = token_outputs.squeeze(0)
        device = token_inputs['input_ids'].device
        
        sent_x = torch.zeros(graph.num_nodes_dict['sentence'], self.encoder.config.hidden_size, device=device)
        if ('token', 'in', 'sentence') in graph.edge_index_dict and graph['token', 'in', 'sentence'].edge_index.numel() > 0:
            tok_in_sent_edge_index = graph['token', 'in', 'sentence'].edge_index
            token_feats = graph['token'].x[tok_in_sent_edge_index[0]]
            sent_indices = tok_in_sent_edge_index[1]
            sent_x.index_add_(0, sent_indices, token_feats)
            counts = torch.bincount(sent_indices, minlength=sent_x.size(0)).float().clamp(min=1).unsqueeze(1)
            sent_x /= counts

        x_dict = {'token': graph['token'].x, 'entity': self.entity_emb(graph['entity'].x.squeeze(-1)), 'sentence': sent_x}
        for ntype, x in x_dict.items():  
            if x.size(0) > 0: x_dict[ntype] = self.lin_dict[ntype](x).relu()
        for conv in self.gnn_layers:  
            x_dict = conv(x_dict, graph.edge_index_dict)
        
        entity_reps = x_dict['entity']
        token_reps = x_dict['token']

        all_pair_indices = []
        entity_to_sent = graph.entity_to_sentence_map.to(entity_reps.device)
        for sent_idx in torch.unique(entity_to_sent):
            entities_in_sent_indices = torch.where(entity_to_sent == sent_idx)[0]
            if len(entities_in_sent_indices) >= 2:
                combinations = torch.combinations(entities_in_sent_indices, r=2)
                all_pair_indices.append(combinations)
                all_pair_indices.append(combinations.flip(1))

        if not all_pair_indices:
            return torch.empty((0, self.relation_classifier[-1].out_features), device=device), torch.empty((0, 2), dtype=torch.long, device=device), entity_reps, None

        pair_indices = torch.cat(all_pair_indices, dim=0)
        head_indices, tail_indices = pair_indices[:, 0], pair_indices[:, 1]
        
        head_reps_latent, tail_reps_latent = entity_reps[head_indices], entity_reps[tail_indices]
        P_uv = self.latent_relation_predictor(head_reps_latent, tail_reps_latent)

        # Hierarchical Message Passing
        updated_entity_reps = self.hierarchical_message_passing(entity_reps, P_uv, pair_indices, relation_map)
        head_reps_updated, tail_reps_updated = updated_entity_reps[head_indices], updated_entity_reps[tail_indices]

        # Path Encoder
        sdp_features_list = []
        for h, t in pair_indices:
            path_indices = graph.sdp_paths.get((h.item(), t.item()))
            if path_indices:
                valid_path_indices = [idx for idx in path_indices if idx < token_reps.size(0)]
                if valid_path_indices:
                    sdp_features_list.append(torch.mean(token_reps[valid_path_indices], dim=0))
                else:
                    sdp_features_list.append(torch.zeros(self.hidden_dim, device=device))
            else:
                sdp_features_list.append(torch.zeros(self.hidden_dim, device=device))
        
        encoded_sdp_features = self.sdp_encoder(torch.stack(sdp_features_list))

        # Direct NLP Features
        head_type_ids = graph['entity'].type_id[head_indices]
        tail_type_ids = graph['entity'].type_id[tail_indices]
        head_type_embs = self.final_type_emb(head_type_ids)
        tail_type_embs = self.final_type_emb(tail_type_ids)
        head_starts = graph['entity'].start_token[head_indices]
        tail_starts = graph['entity'].start_token[tail_indices]
        distances = head_starts - tail_starts
        binned_distances = self._bin_distance(distances)
        distance_embs = self.distance_emb(binned_distances)
        nlp_cues = [self._extract_hierarchical_cues(h, t, graph) for h, t in pair_indices]
        nlp_cues_tensor = torch.tensor(nlp_cues, dtype=torch.float, device=device)

        final_reps = torch.cat([
            head_reps_updated, tail_reps_updated, encoded_sdp_features,
            head_type_embs, tail_type_embs, distance_embs,
            nlp_cues_tensor
        ], dim=1)

        logits = self.relation_classifier(final_reps)
        return logits, pair_indices, updated_entity_reps, P_uv

# --- Training Loop ---
def train_and_eval(model, dataloader, entity_map, relation_map, tokenizer, device, optimizer=None, criterion=None, scheduler=None, acyclic_loss_weight=0.0, caf_gamma=1.0, caf_delta=1.0, caf_loss_weight=0.0, separation_loss_weight=0.0):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    total_acyclic_loss, total_caf_loss, total_separation_loss = 0, 0, 0
    desc = "Training" if is_training else "Evaluating"

    rev_relation_map = {v: k for k, v in relation_map.items()}

    iterable = tqdm(dataloader, desc=desc, leave=False)
    for doc in iterable:
        if is_training: optimizer.zero_grad()
        try:
            graph = build_graph_from_doc(doc, entity_map)
            if graph.num_nodes == 0 or 'entity' not in graph.node_types or graph['entity'].num_nodes < 2: continue
            graph = graph.to(device)
        except Exception as e:
            continue

        flat_tokens = [token for sent in doc['sentences'] for token in sent]
        if not flat_tokens: continue
        token_inputs = tokenizer(flat_tokens, return_tensors='pt', is_split_into_words=True, padding='max_length', truncation=True, max_length=512).to(device)
        
        final_num_tokens = token_inputs['input_ids'].shape[1]
        for edge_type in list(graph.edge_index_dict.keys()):
            if 'token' in edge_type:
                edge_index = graph[edge_type].edge_index
                if edge_index.numel() > 0:
                    src_node_type, _, dst_node_type = edge_type
                    src_mask = edge_index[0] < graph.num_nodes_dict[src_node_type]
                    dst_mask = edge_index[1] < graph.num_nodes_dict[dst_node_type]
                    if src_node_type == 'token': src_mask &= edge_index[0] < final_num_tokens
                    if dst_node_type == 'token': dst_mask &= edge_index[1] < final_num_tokens
                    mask = src_mask & dst_mask
                    graph[edge_type].edge_index = edge_index[:, mask]

        with torch.set_grad_enabled(is_training):
            logits, pair_indices, entity_reps, P_uv = model(graph, token_inputs, relation_map)
            
            if logits.size(0) == 0: continue
            
            true_relations = { (graph.entity_to_node_idx[f"{s_idx}-{rel[0]}-{rel[1]}"], graph.entity_to_node_idx[f"{s_idx}-{rel[2]}-{rel[3]}"]): relation_map.get(rel[4])
                             for s_idx, s_rels in enumerate(doc['relations']) for rel in s_rels
                             if f"{s_idx}-{rel[0]}-{rel[1]}" in graph.entity_to_node_idx and f"{s_idx}-{rel[2]}-{rel[3]}" in graph.entity_to_node_idx and rel[4] in relation_map}
            
            labels = torch.tensor([true_relations.get((i.item(), j.item()), relation_map["No-Relation"]) for i, j in pair_indices], dtype=torch.long, device=device)

            if is_training:
                classification_loss = criterion(logits, labels)
                
                acyclic_loss, separation_loss, caf_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                
                if acyclic_loss_weight > 0 and 'Part-Of' in relation_map and P_uv is not None and P_uv.numel() > 0:
                    num_entities = entity_reps.size(0)
                    A_parent = torch.zeros(num_entities, num_entities, device=device)
                    parent_idx = relation_map.get('Part-Of')
                    A_parent[pair_indices[:, 0], pair_indices[:, 1]] = P_uv[:, parent_idx]
                    acyclic_loss = calculate_acyclic_loss(A_parent, num_entities)
                
                if separation_loss_weight > 0 and 'Part-Of' in relation_map and P_uv is not None and P_uv.numel() > 0:
                    num_entities = entity_reps.size(0)
                    A_parent = torch.zeros(num_entities, num_entities, device=device)
                    parent_idx = relation_map.get('Part-Of')
                    A_parent[pair_indices[:, 0], pair_indices[:, 1]] = P_uv[:, parent_idx]
                    separation_loss = calculate_hierarchical_separation_loss(A_parent, num_entities)

                if caf_loss_weight > 0 and model.w_abs is not None:
                    caf_loss = calculate_caf_loss(entity_reps, pair_indices, labels, relation_map, model.w_abs.detach().clone(), caf_gamma, caf_delta)
                
                loss = classification_loss + (acyclic_loss_weight * acyclic_loss) + (caf_loss_weight * caf_loss) + (separation_loss_weight * separation_loss)
                
                if torch.isnan(loss) or torch.isinf(loss): continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                total_acyclic_loss += acyclic_loss.item()
                total_caf_loss += caf_loss.item()
                total_separation_loss += separation_loss.item()
                
                iterable.set_postfix(cls_loss=f"{classification_loss.item():.4f}", acy_loss=f"{acyclic_loss.item():.4f}", sep_loss=f"{separation_loss.item():.4f}", caf_loss=f"{caf_loss.item():.4f}")
            else:
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    if is_training:
        avg_loss = total_loss / (len(iterable) + 1e-9)
        avg_acyclic_loss = total_acyclic_loss / (len(iterable) + 1e-9)
        avg_caf_loss = total_caf_loss / (len(iterable) + 1e-9)
        avg_separation_loss = total_separation_loss / (len(iterable) + 1e-9)
        print(f"\nAverage losses - Total: {avg_loss:.4f}, Acyclic: {avg_acyclic_loss:.4f}, Separation: {avg_separation_loss:.4f}, CAF: {avg_caf_loss:.4f}")
        return avg_loss
    
    no_rel_idx = relation_map["No-Relation"]
    filtered_preds = [p for p, l in zip(all_preds, all_labels) if l != no_rel_idx]
    filtered_labels = [l for l in all_labels if l != no_rel_idx]
    if not filtered_labels: return 0.0, 0.0, 0.0, 0.0, {}, []

    positive_labels = sorted([i for i in relation_map.values() if i != no_rel_idx])
    p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(filtered_labels, filtered_preds, average=None, zero_division=0, labels=positive_labels)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(filtered_labels, filtered_preds, average='macro', zero_division=0, labels=positive_labels)
    acc = np.mean(np.array(filtered_preds) == np.array(filtered_labels)) if filtered_labels else 0.0
    
    f1_per_class_dict = {label: f1 for label, f1 in zip(positive_labels, f1_per_class)}
    pr_stats = [{'Relation Type': rev_relation_map.get(label_idx, "Unknown"), 'Precision': f"{p:.4f}", 'Recall': f"{r:.4f}", 'F1-Score': f"{f1:.4f}"}  
                for label_idx, p, r, f1 in zip(positive_labels, p_per_class, r_per_class, f1_per_class)]
    return p_macro, r_macro, f1_macro, acc, f1_per_class_dict, pr_stats

def plot_abstraction_field(model, dataset, entity_map, relation_map, tokenizer, device, num_samples=10):
    model.eval()
    entity_reps_list, entity_texts_list, abstraction_scores_list = [], [], []

    sample_docs = random.sample(dataset.data, min(num_samples, len(dataset.data)))
    
    with torch.no_grad():
        for doc in tqdm(sample_docs, desc="Collecting embeddings for plot"):
            try:
                graph = build_graph_from_doc(doc, entity_map)
                if graph.num_nodes == 0 or 'entity' not in graph.node_types or graph['entity'].num_nodes == 0: continue
                graph = graph.to(device)
            except: continue
            
            flat_tokens = [token for sent in doc['sentences'] for token in sent]
            if not flat_tokens: continue
            token_inputs = tokenizer(flat_tokens, return_tensors='pt', is_split_into_words=True, padding='max_length', truncation=True, max_length=512).to(device)

            logits, _, entity_reps, _ = model(graph, token_inputs, relation_map)
            
            if model.use_caf_loss and model.w_abs is not None:
                abstraction_scores = torch.matmul(entity_reps, model.w_abs.detach().clone().unsqueeze(-1)).squeeze(-1).cpu().numpy()
                entity_reps_list.append(entity_reps.cpu().numpy())
                entity_texts_list.extend(graph.entity_texts)
                abstraction_scores_list.extend(abstraction_scores)

    if not entity_reps_list:
        print("Could not collect any entity embeddings. Aborting plot.")
        return

    all_reps = np.vstack(entity_reps_list)
    all_scores = np.array(abstraction_scores_list)

    # Normalize scores to [0, 1] for better visualization
    if all_scores.size > 0:
        min_score = np.min(all_scores)
        max_score = np.max(all_scores)
        if max_score > min_score:
            normalized_scores = (all_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(all_scores)
    else:
        normalized_scores = np.zeros_like(all_scores)

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Plot 1: Abstraction Score Distribution
    ax1.hist(normalized_scores, bins=20, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Abstraction Scores', fontsize=16)
    ax1.set_xlabel('Normalized Abstraction Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(axis='y', linestyle='--')

    # Plot 2: Abstraction Field Visualization (t-SNE or PCA for 2D plot)
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(all_reps)-1))
        reps_2d = tsne.fit_transform(all_reps)
        ax2.scatter(reps_2d[:, 0], reps_2d[:, 1], c=normalized_scores, cmap='viridis', alpha=0.7)
        cbar = fig.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Normalized Abstraction Score', rotation=270, labelpad=15)
        ax2.set_title('Entity Embeddings on Abstraction Continuum (t-SNE)', fontsize=16)
        
        # Add labels for a few sample points
        sorted_indices = np.argsort(normalized_scores)
        low_idx = sorted_indices[:3]
        high_idx = sorted_indices[-3:]
        
        for idx in np.concatenate([low_idx, high_idx]):
            ax2.annotate(entity_texts_list[idx], (reps_2d[idx, 0], reps_2d[idx, 1]), fontsize=8, alpha=0.8, ha='right')
        
    except ImportError:
        ax2.text(0.5, 0.5, "t-SNE not available. Cannot plot.", transform=ax2.transAxes, ha='center', va='center')

    fig.suptitle('Visualizing the Learned Abstraction Field', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("abstraction_field_plot.png")
    plt.show()

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hierarchical DHGNN for scientific relation extraction")
    parser.add_argument('--data_dir', type=str, default='../../../datasets/SciER/', help='Directory for dataset files')
    parser.add_argument('--train_file', type=str, default='train.json', help='Training data file')
    parser.add_argument('--dev_file', type=str, default='dev.json', help='Development data file')
    parser.add_argument('--test_file', type=str, default='test.json', help='Test data file')
    parser.add_argument('--model_name', type=str, default='allenai/scibert_scivocab_uncased', help='Pre-trained transformer')
    parser.add_argument('--hidden_dim', type=int, default=256, help='GNN hidden dimension')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--cache_dir', type=str, default='./model_cache', help='Cache directory for transformers')
    parser.add_argument('--gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss')
    parser.add_argument('--use_acyclic_loss', action='store_true', help='Enable acyclic loss for Part-Of relations')
    parser.add_argument('--acyclic_loss_weight', type=float, default=0.0, help='Weight for acyclic loss')
    parser.add_argument('--use_caf_loss', action='store_true', help='Enable CAF loss')
    parser.add_argument('--caf_gamma', type=float, default=1.0, help='Gamma for CAF anchor loss')
    parser.add_argument('--caf_delta', type=float, default=1.0, help='Margin for CAF ranking loss')
    parser.add_argument('--caf_loss_weight', type=float, default=0.0, help='Weight for CAF loss')
    parser.add_argument('--use_separation_loss', action='store_true', help='Enable hierarchical separation loss')
    parser.add_argument('--separation_loss_weight', type=float, default=0.0, help='Weight for separation loss')
    args = parser.parse_args()

    if args.use_acyclic_loss and args.acyclic_loss_weight == 0.0: args.acyclic_loss_weight = 0.1
    if args.use_caf_loss and args.caf_loss_weight == 0.0: args.caf_loss_weight = 0.1
    if args.use_separation_loss and args.separation_loss_weight == 0.0: args.separation_loss_weight = 0.05
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.cache_dir, exist_ok=True)
    train_path, dev_path, test_path = os.path.join(args.data_dir, args.train_file), os.path.join(args.data_dir, args.dev_file), os.path.join(args.data_dir, args.test_file)

    print("Loading datasets...")
    train_dataset = SciERDataset(train_path)
    dev_dataset = SciERDataset(dev_path) if os.path.exists(dev_path) else None
    test_dataset = SciERDataset(test_path) if os.path.exists(test_path) else None

    print("Creating label maps...")
    entity_map, relation_map = get_label_maps(train_dataset.data)
    rev_relation_map = {v: k for k, v in relation_map.items()}
    print(f"Found {len(entity_map)} entity types and {len(relation_map)} relation types")
    print(f"Relation types: {list(relation_map.keys())}")
    
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    print("Initializing model with hierarchical losses...")
    model = HA_GNN(
        model_name=args.model_name, hidden_dim=args.hidden_dim, num_entity_types=len(entity_map),
        num_relations=len(relation_map), cache_dir=args.cache_dir, use_caf_loss=(args.caf_loss_weight > 0)
    ).to(device)
    
    print("Calculating class weights for Focal Loss...")
    class_alphas = calculate_class_weights(train_dataset.data, relation_map).to(device)
    criterion = FocalLoss(alpha=class_alphas, gamma=args.gamma)
    print(f"Using Focal Loss with gamma={args.gamma}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(train_dataset)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_training_steps), num_training_steps=num_training_steps)

    print(f"\nStarting training with:")
    print(f"  Acyclic loss weight: {args.acyclic_loss_weight}")
    print(f"  Separation loss weight: {args.separation_loss_weight}")
    print(f"  CAF loss weight: {args.caf_loss_weight}")
    print("="*60)

    for epoch in range(args.epochs):
        train_loss = train_and_eval(
            model, train_dataset, entity_map, relation_map, tokenizer, device, 
            optimizer=optimizer, criterion=criterion, scheduler=scheduler,
            acyclic_loss_weight=args.acyclic_loss_weight,
            caf_gamma=args.caf_gamma, caf_delta=args.caf_delta, caf_loss_weight=args.caf_loss_weight,
            separation_loss_weight=args.separation_loss_weight
        )
        print(f"\nEpoch {epoch+1}/{args.epochs} - Training Loss: {train_loss:.4f}")
        
        with torch.no_grad():
            print("\n--- Evaluating on Dev Set ---")
            if dev_dataset:
                p, r, f1, acc, _, pr_stats = train_and_eval(model, dev_dataset, entity_map, relation_map, tokenizer, device, separation_loss_weight=args.separation_loss_weight)
                print(f"Dev Set --> Macro-F1: {f1:.4f}, P: {p:.4f}, R: {r:.4f}, Acc on Positives: {acc:.4f}")
                df = pd.DataFrame(pr_stats)
                print("\n--- Precision, Recall, and F1-Scores per Relation Type (Dev Set) ---")
                print(df.to_string(index=False) if not df.empty else "No positive relations to report on.")
        print("-" * 60)
    
    if args.use_caf_loss:
        print("Training complete. Generating abstraction field plot...")
        plot_abstraction_field(model, test_dataset, entity_map, relation_map, tokenizer, device, num_samples=20)
    
    if test_dataset:
        print("\nFinal Test Evaluation:")
        with torch.no_grad():
            p, r, f1, acc, _, pr_stats = train_and_eval(model, test_dataset, entity_map, relation_map, tokenizer, device, separation_loss_weight=args.separation_loss_weight)
            print(f"Test Set: Macro-F1={f1:.4f}, P={p:.4f}, R={r:.4f}, Acc={acc:.4f}")
            df = pd.DataFrame(pr_stats)
            print("\nPer-relation performance:")
            print(df.to_string(index=False) if not df.empty else "No positive relations found")