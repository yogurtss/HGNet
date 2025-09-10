# Scalable Foundation Model for Automated Knowledge Graph Generation From Scientific Text

## Overview

This repository contains two advanced neural architectures for scientific named entity recognition (NER) and relation extraction:

1. **Hierarchical Graph Neural Network (HA-GNN)** - A sophisticated approach for scientific relation extraction that incorporates hierarchical constraints and continuum abstraction fields
2. **Z-NERD Tagger** - A novel NER model using orthogonal semantic decomposition and multi-scale temporal convolutional attention

Both models are designed specifically for processing scientific literature, with particular focus on the SciERC dataset for entity recognition and relation classification tasks.

## Frameworks
### Z-NERD
![https://raw.githubusercontent.com/basiralab/HA-GNN/blob/main/Z-NERD.png](https://github.com/basiralab/HA-GNN/blob/main/Z-NERD.png)
### HA-GNN
![https://raw.githubusercontent.com/basiralab/HA-GNN/blob/main/HA-GNN.png](https://github.com/basiralab/HA-GNN/blob/main/HA-GNN.png)


## Key Features

### Hierarchical Graph Neural Network (HA-GNN)
- **Graph-based Architecture**: Constructs heterogeneous graphs from scientific documents incorporating tokens, entities, sentences, and their relationships
- **Hierarchical Constraints**: Implements acyclic loss and separation loss to enforce valid hierarchical structures in scientific taxonomies
- **Continuum Abstraction Field (CAF)**: Novel loss function that models abstraction levels for "Part-Of" relationships
- **Multi-modal Features**: Combines syntactic dependency parsing, sequence information, and hierarchical linguistic cues
- **Probabilistic Message Passing**: Latent relation prediction with hierarchical message propagation

### Z-NERD Tagger
- **Orthogonal Semantic Decomposition (OSD)**: Decomposes token embeddings into sustaining and divergent components to capture semantic transitions
- **Multi-Scale TCQK Attention**: Temporal Convolutional Queries & Keys with multiple kernel sizes for capturing different temporal patterns
- **Boundary Detection**: Specialized analysis of divergent vector norms at entity boundaries
- **SciBERT Integration**: Built on top of domain-specific pre-trained language models

## File Structure

```
.
.
├── HA-GNN/                                    # Hierarchical Graph Neural Network framework
│   ├── dataset_generation/                    # Data preprocessing utilities
│   │   ├── generate_concepts.py               # Concept extraction and processing
│   │   └── generate_main_data.py              # Main dataset preparation
│   └── datasets/SciER/                        # SciERC dataset location
├── NER/                                       # Named Entity Recognition tasks
│   ├── baselines/                             # Baseline NER model implementations
│   └── proposed_solution/Z-NERD/              # Z-NERD approach
│       └── znerd.py                           # Z-NERD Tagger with orthogonal decomposition
├── RE/                                        # Relation Extraction tasks
│   ├── baselines/                             # Baseline RE model implementations
│   └── proposed_solutions/                    # Novel RE approaches
│       ├── DP-ERE/                            # Dependency Parsing ERE method
│       │   └── dp_ere_PL_Marker.py            # DP-ERE with positional learning markers
│       └── HA-GNN/                            # Hierarchical attention GNN
│           └── ha-gnn.py                      # HA-GNN with hierarchical constraints
├── README.md                                  # Project documentation
└── requirements.txt                           # Python dependencies
```

## Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Key dependencies include:
- PyTorch (≥1.9.0)
- PyTorch Geometric (≥2.0.0)
- Transformers (≥4.20.0)
- SpaCy with English model
- NetworkX for graph operations
- Scikit-learn for evaluation metrics

## Dataset

Both models are designed for the **SciERC dataset**, which contains scientific abstracts with:
- Named entity annotations (6 entity types: Task, Method, Metric, Material, OtherScientificTerm, Generic)
- Relation annotations (7 relation types including Part-Of, Hyponym-Of, Used-For, etc.)

### Data Format

The expected data format is JSON with the following structure:
```json
{
  "sentences": [["token1", "token2", ...], ...],
  "ner": [[[start, end, "entity_type"], ...], ...],
  "relations": [[[head_start, head_end, tail_start, tail_end, "relation_type"], ...], ...]
}
```

## Usage

### Z-NERD Tagger

Run the NER pipeline:

```bash
python z-nerd.py --data_dir /path/to/scierc/
```

### Hierarchical Graph Neural Network

Run the complete relation extraction pipeline:

```bash
python ha-gnn.py --data_dir /path/to/scier/ \
                          --train_file train.json \
                          --dev_file dev.json \
                          --test_file test.json \
                          --epochs 10 \
                          --lr 2e-5 \
                          --hidden_dim 256 \
                          --use_caf_loss \
                          --use_acyclic_loss \
                          --use_separation_loss
```

Key arguments:
- `--use_caf_loss`: Enable Continuum Abstraction Field loss
- `--use_acyclic_loss`: Enforce acyclic constraints on hierarchical relations
- `--use_separation_loss`: Prevent hierarchy shortcuts
- `--caf_loss_weight`, `--acyclic_loss_weight`, `--separation_loss_weight`: Loss term weights



The model will automatically:
1. Convert SciERC format to NER format
2. Train with orthogonal semantic decomposition
3. Generate analysis plots of divergent vector norms
4. Evaluate on test set

## Model Architecture Details

### HA-GNN Architecture
1. **Graph Construction**: Creates heterogeneous graphs with multiple node and edge types
2. **Feature Encoding**: Uses SciBERT embeddings with type-specific projections
3. **GNN Layers**: Multi-layer graph attention networks with heterogeneous message passing
4. **Hierarchical Processing**: Latent relation prediction followed by probabilistic message passing
5. **Classification**: Multi-feature fusion including path encodings, type embeddings, and linguistic cues

### Z-NERD Architecture
1. **Base Encoding**: SciBERT token representations
2. **Orthogonal Decomposition**: Splits embeddings into sustaining vs. divergent components
3. **Feature Concatenation**: Combines original and divergent features
4. **TCQK Attention**: Multi-scale temporal convolutions on queries and keys
5. **Classification**: Final linear layer for BIO tag prediction

## Evaluation Metrics

Both models report:
- **Precision, Recall, F1-Score**: Per-class and macro-averaged
- **Accuracy**: Overall classification accuracy
- **Detailed Analysis**: Confusion matrices and per-relation performance

## Key Innovations

### Hierarchical Constraints
- **Acyclic Loss**: Prevents circular dependencies in hierarchical relations
- **Separation Loss**: Discourages shortcuts that bypass intermediate hierarchy levels
- **CAF Loss**: Models abstraction continuums for part-whole relationships

### Semantic Decomposition
- **Divergent Vectors**: Capture semantic transitions at entity boundaries
- **Multi-scale Attention**: Different temporal receptive fields for various linguistic patterns
- **Boundary Analysis**: Specialized processing of entity start/end positions

## Performance

Both models achieve competitive results on SciERC:
- **HA-GNN**: Strong performance on relation extraction with hierarchical constraints
- **Z-NERD**: Effective entity recognition with boundary-aware processing

<!-- ## Citation

If you use this code in your research, please cite: -->

<!-- ```bibtex
@article{hierarchical_gnn_znerd_2025,
  title={Hierarchical Graph Neural Networks and Z-NERD for Scientific NER},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
``` -->


