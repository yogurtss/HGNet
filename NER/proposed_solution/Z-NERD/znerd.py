import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import json
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import argparse # Added for command-line arguments
import pandas as pd
import math

# ======================================================================================
# SECTION -1: ENVIRONMENT CONFIGURATION
# ======================================================================================
CACHE_DIR = os.path.join(os.getcwd(), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"--- Forcing all caches to use local directory: {CACHE_DIR} ---")
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'huggingface', 'models')

# ======================================================================================
# SECTION 0: CONFIGURATION
# ======================================================================================
MODEL_NAME = "allenai/scibert_scivocab_uncased"
FINAL_MODEL_SAVE_PATH = "z_nerd_tagger_scierc_model.pth"
PLOT_SAVE_PATH = "divergent_vector_norms_plot_scierc.png"
PICKLE_SAVE_PATH = "divergent_vector_norms_data_scierc.pkl"

# --- Training Hyperparameters ---
EPOCHS = 10
LEARNING_RATE = 3e-5
MAX_LEN = 256 # Increased for potentially longer scientific sentences
BATCH_SIZE = 16

# ======================================================================================
# SECTION 1: THE Z-NERD TAGGER ARCHITECTURE (Orthogonal Decomposition + TCQK)
# ======================================================================================

class MultiScaleTCQKAttention(nn.Module):
    """
    Implements the Multi-Scale Temporal Convolutional Queries & Keys (TCQK)
    self-attention mechanism. This block is designed to be placed after the main
    encoder to process enriched embeddings.
    """
    def __init__(self, embed_dim, num_heads=12, kernel_sizes=[1, 3, 5, 7]):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        assert num_heads % len(kernel_sizes) == 0, "Number of heads must be divisible by number of kernel groups"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kernel_sizes = kernel_sizes
        self.num_groups = len(kernel_sizes)
        self.heads_per_group = num_heads // self.num_groups

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.conv_layers = nn.ModuleList()
        for k_size in kernel_sizes:
            # Each group of heads gets its own convolution type
            conv = nn.Conv1d(
                in_channels=self.head_dim,
                out_channels=self.head_dim,
                kernel_size=k_size,
                padding='same' # Preserves sequence length
            )
            self.conv_layers.append(conv)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Add residual connection
        residual = x

        # 1. Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape and permute for multi-head processing
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, nH, S, H_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, nH, S, H_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, nH, S, H_dim)

        # 3. Apply multi-scale convolutions to Q and K
        q_conv_list, k_conv_list = [], []
        for i in range(self.num_groups):
            start_head = i * self.heads_per_group
            end_head = (i + 1) * self.heads_per_group
            
            q_group = q[:, start_head:end_head] # (B, h_per_g, S, H_dim)
            k_group = k[:, start_head:end_head] # (B, h_per_g, S, H_dim)

            # Reshape for Conv1D: (N, C_in, L) -> (B * h_per_g, H_dim, S)
            q_group_reshaped = q_group.reshape(-1, seq_len, self.head_dim).permute(0, 2, 1)
            k_group_reshaped = k_group.reshape(-1, seq_len, self.head_dim).permute(0, 2, 1)

            # Apply convolution
            q_conv = self.conv_layers[i](q_group_reshaped)
            k_conv = self.conv_layers[i](k_group_reshaped)

            # Reshape back to original: (B, h_per_g, S, H_dim)
            q_conv_list.append(q_conv.permute(0, 2, 1).view(batch_size, self.heads_per_group, seq_len, self.head_dim))
            k_conv_list.append(k_conv.permute(0, 2, 1).view(batch_size, self.heads_per_group, seq_len, self.head_dim))

        q = torch.cat(q_conv_list, dim=1)
        k = torch.cat(k_conv_list, dim=1)

        # 4. Standard scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            # Expand mask to fit attention scores shape
            mask = attention_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        # 5. Concatenate heads and project
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)
        
        # Apply dropout, residual connection, and layer norm
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        
        return output

class ZNERD_Tagger_Model(nn.Module):
    """
    This model implements the full Z-NERD methodology:
    1. Orthogonal Semantic Decomposition (OSD) to get a divergent vector.
    2. A Multi-Scale TCQK Attention block to process the OSD-enriched embeddings.
    """
    def __init__(self, model_name, num_labels):
        super(ZNERD_Tagger_Model, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Dimension of concatenated features (original embedding + divergent vector)
        feature_dim = self.bert.config.hidden_size * 2
        
        # NEW: The TCQK block processes the combined features from OSD
        # Using 12 heads to match SciBERT-base's default, and 4 kernel groups
        self.tcqk_attention = MultiScaleTCQKAttention(
            embed_dim=feature_dim, 
            num_heads=12, 
            kernel_sizes=[1, 3, 5, 7]
        )
        
        # The final classifier takes the output of the TCQK block
        self.classifier = nn.Linear(feature_dim, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # --- Stage 1: Get base contextual embeddings ---
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # Shape: (batch, seq_len, hidden_size)

        # --- Stage 2: ORTHOGONAL SEMANTIC DECOMPOSITION ---
        E_t_minus_1 = nn.functional.pad(sequence_output, (0, 0, 1, 0))[:, :-1]
        E_t = sequence_output
        delta_E_t = E_t - E_t_minus_1
        dot_product = torch.sum(delta_E_t * E_t_minus_1, dim=-1, keepdim=True)
        norm_sq_E_t_minus_1 = torch.sum(E_t_minus_1**2, dim=-1, keepdim=True)
        projection_scalar = dot_product / (norm_sq_E_t_minus_1 + 1e-8)
        v_sustaining = projection_scalar * E_t_minus_1
        v_divergent = delta_E_t - v_sustaining
        
        combined_features = torch.cat((sequence_output, v_divergent), dim=-1)
        
        # --- Stage 3: TCQK ATTENTION MECHANISM ---
        # The enriched features are passed through the TCQK block
        refined_features = self.tcqk_attention(combined_features, attention_mask=attention_mask)
        
        refined_features = self.dropout(refined_features)
        logits = self.classifier(refined_features)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.classifier.out_features)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        
        # We still return v_divergent for analysis purposes
        return type('obj', (object,), {'loss': loss, 'logits': logits, 'velocity': v_divergent})

# ======================================================================================
# SECTION 1.5: SCIERC DATA ADAPTER
# ======================================================================================
def load_and_convert_scierc(file_path):
    """
    Loads a SciERC dataset file and converts it.
    This FINAL version correctly handles DOCUMENT-LEVEL entity indexing
    by calculating a running token offset.
    """
    print(f"Loading and converting SciERC data from: {file_path}")
    converted_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        scierc_docs = json.load(f)

    for doc in tqdm(scierc_docs, desc="Converting documents"):
        token_offset = 0
        # NEW: Flatten all entity annotations for the document into a single list.
        all_doc_entities = [entity for sent_ner in doc['ner'] for entity in sent_ner]

        for sent_idx, tokens in enumerate(doc['sentences']):
            # --- Part 1: Reconstruct sentence and find character offsets ---
            text = ""
            token_char_starts = []
            for token in tokens:
                token_char_starts.append(len(text))
                text += token + " "
            text = text.strip()
            
            current_sent_len = len(tokens)
            # Define the global token range for the current sentence
            sent_global_start = token_offset
            sent_global_end = token_offset + current_sent_len

            # --- Part 2: Find entities that belong to this sentence and convert their indices ---
            entities_in_this_sentence = []
            for entity in all_doc_entities:
                global_start_idx, global_end_idx, entity_type = entity

                # Check if the entity's start index falls within the current sentence's global range
                if sent_global_start <= global_start_idx < sent_global_end:
                    # Convert document-level indices to local, sentence-level indices
                    local_start_idx = global_start_idx - token_offset
                    local_end_idx = global_end_idx - token_offset

                    # Robustness check on the NEW local indices
                    if local_start_idx >= current_sent_len or local_end_idx > current_sent_len:
                        print(f"\n⚠️ Warning: Skipping entity post-conversion. Local indices [{local_start_idx}, {local_end_idx}] are out of bounds for sentence with {current_sent_len} tokens.")
                        continue

                    # Convert local token indices to character spans
                    char_start = token_char_starts[local_start_idx]
                    char_end = token_char_starts[local_end_idx - 1] + len(tokens[local_end_idx - 1])
                    
                    entities_in_this_sentence.append({
                        "span": [char_start, char_end],
                        "type": entity_type
                    })

            converted_data.append({
                "text": text,
                "entities": entities_in_this_sentence
            })
            
            # --- Part 3: Update the offset for the next sentence ---
            token_offset += current_sent_len
            
    return converted_data

def get_entity_types_from_data(data_samples):
    """Extracts all unique entity types from the converted data."""
    entity_types = set()
    for sample in data_samples:
        for entity in sample.get('entities', []):
            entity_types.add(entity['type'])
    return sorted(list(entity_types))

# ======================================================================================
# SECTION 2: CUSTOM DATA PIPELINE
# ======================================================================================
def create_custom_label_maps(entity_types):
    tag_to_id = {'O': 0}
    for etype in entity_types:
        tag_to_id[f'B-{etype}'] = len(tag_to_id); tag_to_id[f'I-{etype}'] = len(tag_to_id)
    return tag_to_id, {v: k for k, v in tag_to_id.items()}

class CustomNERDataset(Dataset):
    def __init__(self, data, tokenizer, tag_to_id, max_len):
        self.tokenizer = tokenizer; self.tag_to_id = tag_to_id; self.max_len = max_len
        self.all_tokenized_inputs = []
        for sentence_data in tqdm(data, desc="Preprocessing data"):
            text = sentence_data['text']; annotations = sentence_data.get('entities', [])
            tokenized = tokenizer(text, max_length=self.max_len, padding="max_length", truncation=True, return_offsets_mapping=True)
            labels = [self.tag_to_id['O']] * self.max_len
            
            is_start_boundary = [False] * self.max_len
            is_end_boundary = [False] * self.max_len
            
            for entity in annotations:
                start, end = entity['span']; etype = entity.get('type')
                if not etype: continue
                b_tag = self.tag_to_id.get(f"B-{etype}"); i_tag = self.tag_to_id.get(f"I-{etype}")
                if b_tag is None or i_tag is None: continue
                
                # Find tokens that fall within the character span
                entity_tokens = []
                for i, (os, oe) in enumerate(tokenized['offset_mapping']):
                    if oe > 0 and max(os, start) < min(oe, end):
                        entity_tokens.append(i)

                if entity_tokens:
                    labels[entity_tokens[0]] = b_tag
                    is_start_boundary[entity_tokens[0]] = True
                    is_end_boundary[entity_tokens[-1]] = True
                    for idx in entity_tokens[1:]: labels[idx] = i_tag
            
            # Set labels for special tokens ([CLS], [SEP], [PAD]) to be ignored by the loss function
            for i, offset in enumerate(tokenized['offset_mapping']):
                if offset == (0, 0):
                    labels[i] = -100

            self.all_tokenized_inputs.append({
                'input_ids': torch.tensor(tokenized['input_ids']),
                'attention_mask': torch.tensor(tokenized['attention_mask']),
                'labels': torch.tensor(labels),
                'is_start_boundary': torch.tensor(is_start_boundary),
                'is_end_boundary': torch.tensor(is_end_boundary),
            })
    
    def __len__(self): return len(self.all_tokenized_inputs)
    def __getitem__(self, idx): return self.all_tokenized_inputs[idx]

# ======================================================================================
# SECTION 3: TRAINING & EVALUATION
# ======================================================================================
def train_and_evaluate(model, train_loader, val_loader, test_loader, id_to_tag, concept_types):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)
    best_val_f1 = -1.0

    stored_info = []
    start_vel_dict = {etype: [] for etype in concept_types}
    end_vel_dict = {etype: [] for etype in concept_types}
    non_entity_velocities = []
    batch_count = 0

    for epoch in range(EPOCHS):
        model.train(); total_loss = 0
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device); mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            is_start_boundary = batch['is_start_boundary'].to(device)
            is_end_boundary = batch['is_end_boundary'].to(device)
            
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward(); optimizer.step(); scheduler.step()
            total_loss += loss.item()

            # --- Store divergent vector norms for analysis ---
            with torch.no_grad():
                velocities = outputs.velocity

                # Start boundaries
                start_mask = is_start_boundary.bool()
                if start_mask.any():
                    s_vels = velocities[start_mask]
                    s_norms = torch.linalg.norm(s_vels, dim=-1).cpu().numpy()
                    s_labels = labels[start_mask].cpu().numpy()
                    for lab, norm in zip(s_labels, s_norms):
                        if lab != -100:
                            tag = id_to_tag.get(lab, 'O')
                            if tag.startswith('B-'):
                                etype = tag[2:]
                                if etype in start_vel_dict:
                                    start_vel_dict[etype].append(norm)
                # End boundaries
                end_mask = is_end_boundary.bool()
                if end_mask.any():
                    e_vels = velocities[end_mask]
                    e_norms = torch.linalg.norm(e_vels, dim=-1).cpu().numpy()
                    e_labels = labels[end_mask].cpu().numpy()
                    for lab, norm in zip(e_labels, e_norms):
                        if lab != -100:
                            tag = id_to_tag.get(lab, 'O')
                            if tag.startswith('B-') or tag.startswith('I-'):
                                etype = tag[2:]
                                if etype in end_vel_dict:
                                    end_vel_dict[etype].append(norm)

                # Non-entity (O tags)
                o_mask = (labels == 0)
                if o_mask.any():
                    o_vels = velocities[o_mask]
                    o_norms = torch.linalg.norm(o_vels, dim=-1).cpu().numpy()
                    non_entity_velocities.extend(o_norms)
            
            # Store data periodically for plotting
            batch_count += 1
            is_last_batch = (epoch == EPOCHS - 1 and batch_idx == len(train_loader) - 1)
            if batch_count % 50 == 0 or is_last_batch: # Store more frequently
                start_means = {etype: np.mean(start_vel_dict[etype]) if start_vel_dict[etype] else np.nan for etype in concept_types}
                end_means = {etype: np.mean(end_vel_dict[etype]) if end_vel_dict[etype] else np.nan for etype in concept_types}
                non_mean = np.mean(non_entity_velocities) if non_entity_velocities else np.nan
                stored_info.append({'batch_step': batch_count, 'start_means': start_means, 'end_means': end_means, 'non_mean': non_mean})
                # Reset collectors
                start_vel_dict = {etype: [] for etype in concept_types}; end_vel_dict = {etype: [] for etype in concept_types}; non_entity_velocities = []

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} finished. Avg Train Loss: {avg_train_loss:.4f}")
        
        val_metrics = evaluate(model, val_loader, id_to_tag, device, is_final_test=False)
        val_f1 = val_metrics['micro avg']['f1-score']
        print(f"Validation F1-Score: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)
            print(f"✅ New best model saved with F1: {best_val_f1:.4f} to {FINAL_MODEL_SAVE_PATH}")

    # --- Save analysis data and create plot ---
    with open(PICKLE_SAVE_PATH, 'wb') as f:
        pickle.dump({'stored_info': stored_info, 'concept_types': concept_types}, f)
    print(f"✅ Divergent vector norms data saved to {PICKLE_SAVE_PATH}")

    if stored_info:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(16, 9))
        steps = [info['batch_step'] for info in stored_info]
        colors = plt.cm.get_cmap('tab10', len(concept_types))
        
        for i, etype in enumerate(concept_types):
            start_m = [info['start_means'].get(etype, np.nan) for info in stored_info]
            plt.plot(steps, start_m, label=f'Start-{etype}', color=colors(i), marker='o', markersize=4, linestyle='-')
            end_m = [info['end_means'].get(etype, np.nan) for info in stored_info]
            plt.plot(steps, end_m, label=f'End-{etype}', color=colors(i), marker='x', markersize=5, linestyle=':')

        non_m = [info['non_mean'] for info in stored_info]
        plt.plot(steps, non_m, label='Non-Entity (O)', color='grey', linestyle='--', marker='s', markersize=4)
        
        plt.xlabel('Training Batch Step', fontsize=14)
        plt.ylabel('Average Divergent Vector Norm', fontsize=14)
        plt.title('Divergent Vector Norms at Entity Boundaries (SciERC)', fontsize=16)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0, fontsize=10)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(PLOT_SAVE_PATH, dpi=300)
        plt.close()
        print(f"✅ Plot saved to {PLOT_SAVE_PATH}")

    # --- Final evaluation on the test set ---
    print("\n--- Running Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(FINAL_MODEL_SAVE_PATH, map_location=device))
    test_metrics = evaluate(model, test_loader, id_to_tag, device, is_final_test=True)
    print("\n--- Final Test Performance Metrics on SciERC ---")
    report_df = pd.DataFrame(test_metrics).transpose()
    print(report_df.to_string(float_format="%.4f"))


def evaluate(model, data_loader, id_to_tag, device, is_final_test=False):
    model.eval(); all_preds, all_labels = [], []
    desc = "Final Test Evaluation" if is_final_test else "Validation"
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, leave=False):
            ids = batch['input_ids'].to(device); mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds = torch.argmax(logits, dim=-1)
            for i in range(labels.shape[0]):
                active_mask = labels[i] != -100
                true_labels = labels[i][active_mask]
                pred_labels = preds[i][active_mask]
                all_labels.extend([id_to_tag.get(l.item(), 'O') for l in true_labels])
                all_preds.extend([id_to_tag.get(p.item(), 'O') for p in pred_labels])
    entity_tags = [tag for tag in id_to_tag.values() if tag != 'O']
    return classification_report(all_labels, all_preds, labels=entity_tags, output_dict=True, zero_division=0)

# ======================================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ======================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Z-NERD Tagger on the SciER dataset.")
    parser.add_argument('--data_dir', type=str, default='../../../datasets/SciER/', help='Directory containing train.json, dev.json, and test.json.')
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, 'train.json')
    dev_path = os.path.join(args.data_dir, 'dev.json')
    test_path = os.path.join(args.data_dir, 'test.json')

    if not all(os.path.exists(p) for p in [train_path, dev_path, test_path]):
        print("❌ Error: Not all SciERC files (train.json, dev.json, test.json) found in the specified data directory.")
        print("Please download the dataset from https://github.com/allenai/scierc and place the files in the correct directory.")
        exit()

    # MODIFIED: Load data using the new converter
    train_set = load_and_convert_scierc(train_path)
    val_set = load_and_convert_scierc(dev_path)
    test_set = load_and_convert_scierc(test_path)
    
    # MODIFIED: Get entity types directly from training data
    concept_types = get_entity_types_from_data(train_set)
    print(f"\nFound {len(concept_types)} unique entity types in SciERC: {concept_types}")

    tag_to_id, id_to_tag = create_custom_label_maps(concept_types)
    num_labels = len(tag_to_id)
    print(f"Created {num_labels} unique BIO tags.")
    
    print(f"\nData loaded:")
    print(f"  Training sentences:   {len(train_set)}")
    print(f"  Validation sentences: {len(val_set)}")
    print(f"  Test sentences:       {len(test_set)}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ZNERD_Tagger_Model(MODEL_NAME, num_labels=num_labels)
    print("\n--- Z-NERD Tagger Model loaded with OSD and TCQK Attention. ---")
    
    train_dataset = CustomNERDataset(train_set, tokenizer, tag_to_id, MAX_LEN)
    val_dataset = CustomNERDataset(val_set, tokenizer, tag_to_id, MAX_LEN)
    test_dataset = CustomNERDataset(test_set, tokenizer, tag_to_id, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    train_and_evaluate(model, train_loader, val_loader, test_loader, id_to_tag, concept_types)