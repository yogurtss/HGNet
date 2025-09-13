import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertSelfAttention
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import time
from datetime import datetime

# ======================================================================================
# CONFIGURATION FOR SCALING ANALYSIS
# ======================================================================================
KB_FILE = "knowledge_base_v3.json"
SENTENCES_FILE = "annotated_sentences_final.jsonl"
RESULTS_DIR = "scaling_analysis_results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "scaling_results.json")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Model configurations for scaling analysis
MODEL_CONFIGS = {
    "small": {
        "model_name": "allenai/scibert_scivocab_uncased",
        "hidden_size": 768,
        "num_hidden_layers": 6,
        "num_attention_heads": 12,
        "intermediate_size": 2048,
        "params_estimate": "50M"
    },
    "medium": {
        "model_name": "allenai/scibert_scivocab_uncased", 
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "params_estimate": "110M"
    },
    "large": {
        "model_name": "allenai/scibert_scivocab_uncased",
        "hidden_size": 1024,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "params_estimate": "200M"
    },
    "very_large": {
        "model_name": "allenai/scibert_scivocab_uncased",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "params_estimate": "400M"
    }
}

# Data size percentages for scaling analysis
DATA_SIZES = [0.1, 0.3, 0.5, 0.8, 1.0]

# Training hyperparameters (reduced for scaling analysis)
EPOCHS = 3  # Reduced for faster scaling analysis
LEARNING_RATE = 3e-5
MAX_LEN = 128
BATCH_SIZE = 16

# ======================================================================================
# SCSA ARCHITECTURE (Same as original)
# ======================================================================================
class SpanCentricSelfAttention(BertSelfAttention):
    def __init__(self, config, max_span_width=10):
        super().__init__(config)
        self.span_bias = nn.Parameter(torch.zeros(self.num_attention_heads, max_span_width))

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None,
        encoder_hidden_states=None, encoder_attention_mask=None,
        past_key_value=None, output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / (self.attention_head_size ** 0.5)
        seq_len = hidden_states.size(1)
        pos_range = torch.arange(seq_len, device=hidden_states.device)
        relative_pos = torch.abs(pos_range.unsqueeze(0) - pos_range.unsqueeze(1))
        clipped_pos = torch.clamp(relative_pos, 0, self.span_bias.shape[1] - 1)
        bias = self.span_bias.gather(1, clipped_pos.view(1, -1).expand(self.num_attention_heads, -1)).view(self.num_attention_heads, seq_len, seq_len)
        attention_scores = attention_scores + bias
        if attention_mask is not None: attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None: attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

# ======================================================================================
# DATA LOADING AND PROCESSING (Same as original)
# ======================================================================================
def load_custom_data(kb_path, sentences_path):
    print("Loading custom data...")
    try:
        with open(kb_path, 'r') as f: 
            knowledge_base = json.load(f)
        sentences = [json.loads(line) for line in open(sentences_path, 'r')]
        concept_types = sorted(list(set(c['type'] for c in knowledge_base['concepts'])))
        print(f"Loaded {len(sentences)} sentences.")
        print(f"Found {len(concept_types)} unique entity types: {concept_types}")
        return sentences, concept_types
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None, None

def create_custom_label_maps(entity_types):
    tag_to_id = {'O': 0}
    for etype in entity_types:
        tag_to_id[f'B-{etype}'] = len(tag_to_id)
        tag_to_id[f'I-{etype}'] = len(tag_to_id)
    return tag_to_id, {v: k for k, v in tag_to_id.items()}

class CustomNERDataset(Dataset):
    def __init__(self, data, tokenizer, tag_to_id, max_len):
        self.tokenizer = tokenizer
        self.tag_to_id = tag_to_id
        self.max_len = max_len
        self.all_tokenized_inputs = []
        
        for sentence_data in tqdm(data, desc="Preprocessing data"):
            text = sentence_data['text']
            annotations = sentence_data.get('entities', [])
            tokenized = tokenizer(text, max_length=self.max_len, padding="max_length", 
                                truncation=True, return_offsets_mapping=True)
            labels = [self.tag_to_id['O']] * self.max_len
            
            for entity in annotations:
                start, end = entity['span']
                etype = entity.get('type')
                if not etype: continue
                b_tag = self.tag_to_id.get(f"B-{etype}")
                i_tag = self.tag_to_id.get(f"I-{etype}")
                if b_tag is None or i_tag is None: continue
                
                entity_tokens = [i for i, (os, oe) in enumerate(tokenized['offset_mapping']) 
                               if oe > 0 and max(os, start) < min(oe, end)]
                if entity_tokens:
                    labels[entity_tokens[0]] = b_tag
                    for idx in entity_tokens[1:]: 
                        labels[idx] = i_tag
            
            for i, input_id in enumerate(tokenized['input_ids']):
                if input_id in self.tokenizer.all_special_ids: 
                    labels[i] = -100
            
            self.all_tokenized_inputs.append({
                'input_ids': torch.tensor(tokenized['input_ids']),
                'attention_mask': torch.tensor(tokenized['attention_mask']),
                'labels': torch.tensor(labels)
            })
    
    def __len__(self): 
        return len(self.all_tokenized_inputs)
    
    def __getitem__(self, idx): 
        return self.all_tokenized_inputs[idx]

# ======================================================================================
# MODEL CREATION WITH CUSTOM CONFIGURATIONS
# ======================================================================================
def create_model_with_config(config, num_labels):
    """Create a model with custom configuration for scaling analysis"""
    from transformers import BertConfig
    
    # Create custom BERT config
    bert_config = BertConfig(
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        num_labels=num_labels,
        vocab_size=31090,  # SciBERT vocab size
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    
    # Initialize model with custom config
    model = AutoModelForTokenClassification.from_config(bert_config)
    
    # Add SCSA mechanism
    for layer in model.bert.encoder.layer:
        custom_attention = SpanCentricSelfAttention(bert_config)
        layer.attention.self = custom_attention
    
    return model

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# ======================================================================================
# TRAINING AND EVALUATION FOR SCALING ANALYSIS
# ======================================================================================
def train_model(model, train_loader, val_loader, device):
    """Train model and return training statistics"""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    training_stats = {
        'epoch_losses': [],
        'epoch_times': [],
        'val_f1_scores': []
    }
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        # Quick validation
        val_f1 = evaluate_model(model, val_loader, device)
        
        training_stats['epoch_losses'].append(avg_loss)
        training_stats['epoch_times'].append(epoch_time)
        training_stats['val_f1_scores'].append(val_f1)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val F1={val_f1:.4f}, Time={epoch_time:.2f}s")
    
    return training_stats

def evaluate_model(model, data_loader, device):
    """Evaluate model and return F1 score"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds = torch.argmax(logits, dim=-1)
            
            for i in range(labels.shape[0]):
                active_mask = labels[i] != -100
                true_labels = labels[i][active_mask]
                pred_labels = preds[i][active_mask]
                all_labels.extend(true_labels.cpu().numpy())
                all_preds.extend(pred_labels.cpu().numpy())
    
    # Calculate macro F1 score (ignoring 'O' labels for entity-level performance)
    non_o_mask = np.array(all_labels) != 0
    if np.sum(non_o_mask) > 0:
        f1 = f1_score(
            np.array(all_labels)[non_o_mask], 
            np.array(all_preds)[non_o_mask], 
            average='macro'
        )
    else:
        f1 = 0.0
    
    return f1

# ======================================================================================
# SCALING ANALYSIS EXECUTION
# ======================================================================================
def run_scaling_analysis():
    """Run the complete scaling analysis"""
    print("🚀 Starting Neural Scaling Analysis...")
    
    # Load data
    sentences, concept_types = load_custom_data(KB_FILE, SENTENCES_FILE)
    if not sentences:
        return
    
    tag_to_id, id_to_tag = create_custom_label_maps(concept_types)
    num_labels = len(tag_to_id)
    
    # Split data
    train_val_sentences, test_sentences = train_test_split(
        sentences, test_size=0.1, random_state=42
    )
    train_sentences, val_sentences = train_test_split(
        train_val_sentences, test_size=0.1, random_state=42
    )
    
    print(f"Total training sentences available: {len(train_sentences)}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    
    # Results storage
    scaling_results = []
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run scaling experiments
    total_experiments = len(MODEL_CONFIGS) * len(DATA_SIZES)
    experiment_count = 0
    
    for model_size, model_config in MODEL_CONFIGS.items():
        for data_size in DATA_SIZES:
            experiment_count += 1
            print(f"\n{'='*60}")
            print(f"Experiment {experiment_count}/{total_experiments}")
            print(f"Model Size: {model_size} ({model_config['params_estimate']})")
            print(f"Data Size: {int(data_size*100)}% ({int(len(train_sentences)*data_size)} sentences)")
            print(f"{'='*60}")
            
            # Create data subset
            subset_size = int(len(train_sentences) * data_size)
            train_subset = train_sentences[:subset_size]
            
            # Create datasets
            train_dataset = CustomNERDataset(train_subset, tokenizer, tag_to_id, MAX_LEN)
            val_dataset = CustomNERDataset(val_sentences, tokenizer, tag_to_id, MAX_LEN)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            
            # Create model
            model = create_model_with_config(model_config, num_labels)
            total_params, trainable_params = count_parameters(model)
            
            print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
            
            # Train and evaluate
            start_time = time.time()
            training_stats = train_model(model, train_loader, val_loader, device)
            total_time = time.time() - start_time
            
            # Final evaluation on test set (small subset for speed)
            test_dataset = CustomNERDataset(test_sentences[:100], tokenizer, tag_to_id, MAX_LEN)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
            final_f1 = evaluate_model(model, test_loader, device)
            
            # Store results
            result = {
                'timestamp': datetime.now().isoformat(),
                'model_size': model_size,
                'data_size_percent': int(data_size * 100),
                'data_size_samples': subset_size,
                'model_params_total': total_params,
                'model_params_trainable': trainable_params,
                'training_time_total': total_time,
                'final_val_f1': max(training_stats['val_f1_scores']),
                'final_test_f1': final_f1,
                'epoch_losses': training_stats['epoch_losses'],
                'epoch_times': training_stats['epoch_times'],
                'val_f1_scores': training_stats['val_f1_scores'],
                'model_config': model_config
            }
            
            scaling_results.append(result)
            
            # Save intermediate results
            with open(RESULTS_FILE, 'w') as f:
                json.dump(scaling_results, f, indent=2)
            
            print(f"✅ Completed: Final Test F1 = {final_f1:.4f}")
            
            # Clear memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n🎉 Scaling analysis complete! Results saved to {RESULTS_FILE}")
    return scaling_results

# ======================================================================================
# VISUALIZATION AND ANALYSIS
# ======================================================================================
def create_scaling_plots(results):
    """Create comprehensive scaling law visualizations"""
    print("📊 Creating scaling analysis plots...")
    
    # Convert results to DataFrame
    df_list = []
    for result in results:
        df_list.append({
            'model_size': result['model_size'],
            'data_size_percent': result['data_size_percent'],
            'data_size_samples': result['data_size_samples'],
            'model_params': result['model_params_total'],
            'final_test_f1': result['final_test_f1'],
            'final_val_f1': result['final_val_f1'],
            'training_time': result['training_time_total'],
            'params_estimate': result['model_config']['params_estimate']
        })
    
    df = pd.DataFrame(df_list)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Performance vs Model Size
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Neural Scaling Laws Analysis for Z-NERD Model', fontsize=16, fontweight='bold')
    
    # Plot 1: F1 Score vs Model Parameters
    for data_size in sorted(df['data_size_percent'].unique()):
        subset = df[df['data_size_percent'] == data_size]
        axes[0,0].plot(subset['model_params'], subset['final_test_f1'], 
                      marker='o', linewidth=2, markersize=8, 
                      label=f'{data_size}% Data')
    
    axes[0,0].set_xlabel('Model Parameters')
    axes[0,0].set_ylabel('Test F1 Score')
    axes[0,0].set_title('Performance vs Model Size')
    axes[0,0].set_xscale('log')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: F1 Score vs Data Size
    for model_size in ['small', 'medium', 'large', 'very_large']:
        subset = df[df['model_size'] == model_size]
        if len(subset) > 0:
            axes[0,1].plot(subset['data_size_samples'], subset['final_test_f1'],
                          marker='s', linewidth=2, markersize=8,
                          label=f'{model_size.title()} Model')
    
    axes[0,1].set_xlabel('Training Data Size (samples)')
    axes[0,1].set_ylabel('Test F1 Score')
    axes[0,1].set_title('Performance vs Data Size')
    axes[0,1].set_xscale('log')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Training Time vs Model Size
    for data_size in sorted(df['data_size_percent'].unique()):
        subset = df[df['data_size_percent'] == data_size]
        axes[1,0].plot(subset['model_params'], subset['training_time'],
                      marker='^', linewidth=2, markersize=8,
                      label=f'{data_size}% Data')
    
    axes[1,0].set_xlabel('Model Parameters')
    axes[1,0].set_ylabel('Training Time (seconds)')
    axes[1,0].set_title('Training Time vs Model Size')
    axes[1,0].set_xscale('log')
    axes[1,0].set_yscale('log')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Efficiency (F1/Time) vs Model Size
    df['efficiency'] = df['final_test_f1'] / (df['training_time'] / 3600)  # F1 per hour
    for data_size in sorted(df['data_size_percent'].unique()):
        subset = df[df['data_size_percent'] == data_size]
        axes[1,1].plot(subset['model_params'], subset['efficiency'],
                      marker='d', linewidth=2, markersize=8,
                      label=f'{data_size}% Data')
    
    axes[1,1].set_xlabel('Model Parameters')
    axes[1,1].set_ylabel('Efficiency (F1 Score / Hour)')
    axes[1,1].set_title('Training Efficiency vs Model Size')
    axes[1,1].set_xscale('log')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'scaling_laws_overview.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Heatmap of Performance
    pivot_table = df.pivot(index='model_size', columns='data_size_percent', values='final_test_f1')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis', 
                cbar_kws={'label': 'Test F1 Score'})
    plt.title('Performance Heatmap: Model Size vs Data Size', fontsize=14, fontweight='bold')
    plt.xlabel('Data Size (%)')
    plt.ylabel('Model Size')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Power Law Analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Model scaling power law
    full_data = df[df['data_size_percent'] == 100]
    if len(full_data) > 1:
        log_params = np.log10(full_data['model_params'])
        log_f1 = np.log10(full_data['final_test_f1'])
        
        # Fit power law
        coeffs = np.polyfit(log_params, log_f1, 1)
        power_law_slope = coeffs[0]
        
        axes[0].scatter(full_data['model_params'], full_data['final_test_f1'], 
                       s=100, alpha=0.7, color='red')
        axes[0].plot(full_data['model_params'], 
                    10**(coeffs[1]) * (full_data['model_params']**coeffs[0]),
                    '--', color='blue', linewidth=2, 
                    label=f'Power Law: slope={power_law_slope:.3f}')
        
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Model Parameters')
        axes[0].set_ylabel('Test F1 Score')
        axes[0].set_title(f'Model Scaling Power Law\n(100% Data)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Data scaling power law
    large_model = df[df['model_size'] == 'large']
    if len(large_model) > 1:
        log_data = np.log10(large_model['data_size_samples'])
        log_f1 = np.log10(large_model['final_test_f1'])
        
        coeffs = np.polyfit(log_data, log_f1, 1)
        data_law_slope = coeffs[0]
        
        axes[1].scatter(large_model['data_size_samples'], large_model['final_test_f1'],
                       s=100, alpha=0.7, color='green')
        axes[1].plot(large_model['data_size_samples'],
                    10**(coeffs[1]) * (large_model['data_size_samples']**coeffs[0]),
                    '--', color='orange', linewidth=2,
                    label=f'Power Law: slope={data_law_slope:.3f}')
        
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Training Data Size')
        axes[1].set_ylabel('Test F1 Score')
        axes[1].set_title(f'Data Scaling Power Law\n(Large Model)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'power_law_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Summary Statistics
    print("\n📋 SCALING ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total experiments run: {len(results)}")
    print(f"Model sizes tested: {sorted(df['model_size'].unique())}")
    print(f"Data sizes tested: {sorted(df['data_size_percent'].unique())}%")
    print(f"Best performance: {df['final_test_f1'].max():.4f} F1")
    print(f"Best efficiency: {df['efficiency'].max():.4f} F1/hour")
    
    # Best configurations
    best_perf = df.loc[df['final_test_f1'].idxmax()]
    best_eff = df.loc[df['efficiency'].idxmax()]
    
    print(f"\nBest Performance Configuration:")
    print(f"  Model: {best_perf['model_size']} ({best_perf['params_estimate']})")
    print(f"  Data: {best_perf['data_size_percent']}%")
    print(f"  F1 Score: {best_perf['final_test_f1']:.4f}")
    
    print(f"\nMost Efficient Configuration:")
    print(f"  Model: {best_eff['model_size']} ({best_eff['params_estimate']})")
    print(f"  Data: {best_eff['data_size_percent']}%")
    print(f"  Efficiency: {best_eff['efficiency']:.4f} F1/hour")
    
    return df

# ======================================================================================
# MAIN EXECUTION
# ======================================================================================
if __name__ == "__main__":
    # Run scaling analysis
    results = run_scaling_analysis()
    
    if results:
        # Create visualizations
        df = create_scaling_plots(results)
        
        # Save summary statistics
        summary_file = os.path.join(RESULTS_DIR, "scaling_summary.csv")
        df.to_csv(summary_file, index=False)
        print(f"\n📊 Summary statistics saved to {summary_file}")
        print(f"📁 All results and plots saved in {RESULTS_DIR}/")
    else:
        print("❌ Scaling analysis failed. Please check your data files.")