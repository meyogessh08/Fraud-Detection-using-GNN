# %%
# ==============================================================================
# SECTION 1: LIBRARY IMPORTS
# ==============================================================================

print("="*80)
print("SECTION 1: IMPORTING LIBRARIES")
print("="*80)

# Data Manipulation
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')

# Machine Learning: Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

# Machine Learning: Models & Metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve
)

# Imbalanced Learning
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Deep Learning: PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Graph Neural Networks: PyTorch Geometric
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops, degree

# Explainable AI
import shap

# Network Analysis
import networkx as nx

# Utilities
from tqdm import tqdm
import pickle
import os
from datetime import datetime
import gc

# Random Seed for Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\n[SUCCESS] All libraries imported successfully!")
print(f"[INFO] PyTorch Version: {torch.__version__}")
print(f"[INFO] PyTorch Geometric Version: {torch_geometric.__version__}")
print(f"[INFO] Device: {device}")
print(f"[INFO] Random Seed: {RANDOM_SEED}")
print("="*80)

# %%
# ==============================================================================
# SECTION 2: DATA LOADING
# ==============================================================================

print("\n" + "="*80)
print("SECTION 2: DATA LOADING")
print("="*80)

# Define file paths
DATA_PATH = "."
TRANSACTION_FILE = os.path.join(DATA_PATH, "train_transaction.csv")
IDENTITY_FILE = os.path.join(DATA_PATH, "train_identity.csv")

# Load transaction data
print("\n[STEP 1/3] Loading train_transaction.csv...")
df_transaction = pd.read_csv(TRANSACTION_FILE)
print(f"[SUCCESS] Loaded {len(df_transaction):,} transactions")
print(f"[INFO] Shape: {df_transaction.shape}")

# Load identity data
print("\n[STEP 2/3] Loading train_identity.csv...")
df_identity = pd.read_csv(IDENTITY_FILE)
print(f"[SUCCESS] Loaded {len(df_identity):,} identity records")
print(f"[INFO] Shape: {df_identity.shape}")

# Merge datasets
print("\n[STEP 3/3] Merging transaction and identity data...")
df = df_transaction.merge(df_identity, on='TransactionID', how='left')
print(f"[SUCCESS] Merged dataset shape: {df.shape}")
print(f"[INFO] Identity match rate: {(df_identity.shape[0] / df_transaction.shape[0]) * 100:.2f}%")

# Initial inspection
print("\n" + "-"*80)
print("INITIAL DATA INSPECTION")
print("-"*80)
print(f"\nDataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

print(f"\nTarget Distribution:")
fraud_counts = df['isFraud'].value_counts()
print(fraud_counts)
print(f"Fraud Rate: {(fraud_counts[1] / len(df)) * 100:.4f}%")

print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Feature categories
transaction_cols = [col for col in df.columns if col.startswith('Transaction')]
card_cols = [col for col in df.columns if col.startswith('card')]
addr_cols = [col for col in df.columns if col.startswith('addr')]
email_cols = [col for col in df.columns if 'email' in col.lower()]
device_cols = [col for col in df.columns if 'device' in col.lower() or 'id_' in col.lower()]
c_cols = [col for col in df.columns if col.startswith('C')]
d_cols = [col for col in df.columns if col.startswith('D')]
m_cols = [col for col in df.columns if col.startswith('M')]
v_cols = [col for col in df.columns if col.startswith('V')]

print("\n" + "-"*80)
print("FEATURE CATEGORIES")
print("-"*80)
print(f"Transaction Features: {len(transaction_cols)}")
print(f"Card Features: {len(card_cols)}")
print(f"Address Features: {len(addr_cols)}")
print(f"Email Features: {len(email_cols)}")
print(f"Device/ID Features: {len(device_cols)}")
print(f"C Features (Counting): {len(c_cols)}")
print(f"D Features (Timedelta): {len(d_cols)}")
print(f"M Features (Match): {len(m_cols)}")
print(f"V Features (Vesta): {len(v_cols)}")

# Store original dataframe
df_original = df.copy()

print("\n[SUCCESS] Data loading complete!")
print("="*80)

# %%
# ==============================================================================
# SECTION 3: DATA CLEANING & PREPROCESSING
# ==============================================================================

print("\n" + "="*80)
print("SECTION 3: DATA CLEANING & PREPROCESSING")
print("="*80)

df_processed = df.copy()

# Step 1: Missing Value Analysis
print("\n[STEP 1/7] Analyzing missing values...")
missing_pct = (df_processed.isnull().sum() / len(df_processed)) * 100
missing_stats = pd.DataFrame({
    'Column': missing_pct.index,
    'Missing_Percentage': missing_pct.values
}).sort_values('Missing_Percentage', ascending=False)

high_missing = missing_stats[missing_stats['Missing_Percentage'] > 50]
print(f"[INFO] Columns with >50% missing values: {len(high_missing)}")

# Step 2: Drop high missing columns
print("\n[STEP 2/7] Dropping columns with >90% missing values...")
threshold = 90
cols_to_drop_missing = missing_stats[missing_stats['Missing_Percentage'] > threshold]['Column'].tolist()
cols_to_drop_missing.append('TransactionID')

print(f"[INFO] Dropping {len(cols_to_drop_missing)} columns")
df_processed = df_processed.drop(columns=cols_to_drop_missing, errors='ignore')
print(f"[INFO] Remaining columns: {df_processed.shape[1]}")

# Step 3: Identify feature types
print("\n[STEP 3/7] Identifying feature types...")
features = [col for col in df_processed.columns if col != 'isFraud']
categorical_cols = df_processed[features].select_dtypes(include=['object']).columns.tolist()
numerical_cols = df_processed[features].select_dtypes(include=[np.number]).columns.tolist()

print(f"[INFO] Categorical features: {len(categorical_cols)}")
print(f"[INFO] Numerical features: {len(numerical_cols)}")

# Step 4: Handle categorical variables
print("\n[STEP 4/7] Encoding categorical variables...")

# Store entity columns for graph construction
entity_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
               'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
               'DeviceType', 'DeviceInfo']
available_entity_cols = [col for col in entity_cols if col in df.columns]
df_entities = df[['isFraud'] + available_entity_cols].copy()

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    if col in df_processed.columns:
        # Handle high cardinality
        if df_processed[col].nunique() > 100:
            top_categories = df_processed[col].value_counts().head(50).index
            df_processed[col] = df_processed[col].apply(
                lambda x: x if x in top_categories else 'other'
            )
        
        # Label encoding
        le = LabelEncoder()
        df_processed[col] = df_processed[col].fillna('missing')
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

print(f"[SUCCESS] Encoded {len(categorical_cols)} categorical features")

# Step 5: Handle numerical missing values
print("\n[STEP 5/7] Imputing numerical missing values...")
numerical_cols = [col for col in numerical_cols if col in df_processed.columns]
imputer = SimpleImputer(strategy='median')
df_processed[numerical_cols] = imputer.fit_transform(df_processed[numerical_cols])
print(f"[SUCCESS] Imputed {len(numerical_cols)} numerical features")

# Step 6: Remove low variance features
print("\n[STEP 6/7] Removing low variance features...")
X = df_processed.drop('isFraud', axis=1)
y = df_processed['isFraud']

selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
selected_features = X.columns[selector.get_support()].tolist()

print(f"[INFO] Original features: {X.shape[1]}")
print(f"[INFO] Features retained: {len(selected_features)}")

df_processed = pd.DataFrame(X_selected, columns=selected_features)
df_processed['isFraud'] = y.values

# Step 7: Final summary
print("\n[STEP 7/7] Cleaning summary...")
print(f"[INFO] Final shape: {df_processed.shape}")
print(f"[INFO] Missing values: {df_processed.isnull().sum().sum()}")
print(f"[INFO] Fraud rate: {df_processed['isFraud'].mean() * 100:.4f}%")

print("\n[SUCCESS] Data cleaning complete!")
print("="*80)

gc.collect()

# %%
# ==============================================================================
# SECTION 4: FEATURE ENGINEERING
# ==============================================================================

print("\n" + "="*80)
print("SECTION 4: FEATURE ENGINEERING")
print("="*80)

df_engineered = df_processed.copy()

# Step 1: Temporal features
print("\n[STEP 1/6] Creating temporal features...")
if 'TransactionDT' in df_engineered.columns:
    df_engineered['Transaction_hour'] = (df_engineered['TransactionDT'] / 3600) % 24
    df_engineered['Transaction_day'] = (df_engineered['TransactionDT'] / (3600 * 24)) % 7
    
    # Cyclical encoding
    df_engineered['hour_sin'] = np.sin(2 * np.pi * df_engineered['Transaction_hour'] / 24)
    df_engineered['hour_cos'] = np.cos(2 * np.pi * df_engineered['Transaction_hour'] / 24)
    df_engineered['day_sin'] = np.sin(2 * np.pi * df_engineered['Transaction_day'] / 7)
    df_engineered['day_cos'] = np.cos(2 * np.pi * df_engineered['Transaction_day'] / 7)
    
    print("[SUCCESS] Created temporal features")

# Step 2: Amount features
print("\n[STEP 2/6] Creating amount features...")
if 'TransactionAmt' in df_engineered.columns:
    df_engineered['TransactionAmt_log'] = np.log1p(df_engineered['TransactionAmt'])
    df_engineered['TransactionAmt_bin'] = pd.cut(
        df_engineered['TransactionAmt'],
        bins=[0, 50, 100, 200, 500, 1000, np.inf],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)
    df_engineered['TransactionAmt_decimal'] = df_engineered['TransactionAmt'] % 1
    
    print("[SUCCESS] Created amount features")

# Step 3: Card aggregations
print("\n[STEP 3/6] Creating card-based aggregations...")
if 'card1' in df.columns:
    temp_card = df[['card1']].copy()
    card_counts = temp_card.groupby('card1').size().to_dict()
    df_engineered['card1_count'] = temp_card['card1'].map(card_counts).values
    
    if 'TransactionAmt' in df.columns:
        temp_card['TransactionAmt'] = df['TransactionAmt'].values
        card_amt_mean = temp_card.groupby('card1')['TransactionAmt'].mean().to_dict()
        card_amt_std = temp_card.groupby('card1')['TransactionAmt'].std().fillna(0).to_dict()
        
        df_engineered['card1_amt_mean'] = temp_card['card1'].map(card_amt_mean).values
        df_engineered['card1_amt_std'] = temp_card['card1'].map(card_amt_std).values
        df_engineered['amt_deviation'] = (
            df_engineered['TransactionAmt'] - df_engineered['card1_amt_mean']
        ) / (df_engineered['card1_amt_std'] + 1)
    
    print("[SUCCESS] Created card aggregations")

# Step 4: C feature aggregations
print("\n[STEP 4/6] Creating C-feature aggregations...")
c_features = [col for col in df_engineered.columns if col.startswith('C')]
if len(c_features) > 0:
    df_engineered['C_sum'] = df_engineered[c_features].sum(axis=1)
    df_engineered['C_mean'] = df_engineered[c_features].mean(axis=1)
    print(f"[SUCCESS] Created aggregations from {len(c_features)} C features")

# Step 5: D feature aggregations
print("\n[STEP 5/6] Creating D-feature aggregations...")
d_features = [col for col in df_engineered.columns if col.startswith('D')]
if len(d_features) > 0:
    df_engineered['D_sum'] = df_engineered[d_features].sum(axis=1)
    df_engineered['D_mean'] = df_engineered[d_features].mean(axis=1)
    print(f"[SUCCESS] Created aggregations from {len(d_features)} D features")

# Step 6: Handle new missing values
print("\n[STEP 6/6] Handling new missing values...")
new_missing = df_engineered.isnull().sum().sum()
if new_missing > 0:
    df_engineered = df_engineered.fillna(0)
    print(f"[INFO] Filled {new_missing} missing values with 0")

print(f"\n[INFO] Original features: {df_processed.shape[1] - 1}")
print(f"[INFO] Engineered features: {df_engineered.shape[1] - 1}")
print(f"[INFO] New features created: {df_engineered.shape[1] - df_processed.shape[1]}")

print("\n[SUCCESS] Feature engineering complete!")
print("="*80)

gc.collect()

# %%
# ==============================================================================
# SECTION 5: GRAPH CONSTRUCTION (MEMORY-OPTIMIZED)
# ==============================================================================

print("\n" + "="*80)
print("SECTION 5: GRAPH CONSTRUCTION")
print("="*80)

print("\n[INFO] Building transaction graph with memory optimization...")

# CRITICAL: Sample data for graph construction (memory management)
GRAPH_SAMPLE_SIZE = 100000  # Use 100K transactions instead of full 590K

if len(df) > GRAPH_SAMPLE_SIZE:
    print(f"\n[INFO] Sampling {GRAPH_SAMPLE_SIZE:,} transactions for graph (memory optimization)")
    print(f"[INFO] Stratified sampling to maintain fraud ratio...")
    
    # Stratified sampling to keep fraud ratio
    from sklearn.model_selection import train_test_split
    sample_indices, _ = train_test_split(
        range(len(df)),
        train_size=GRAPH_SAMPLE_SIZE,
        stratify=df['isFraud'],
        random_state=RANDOM_SEED
    )
    sample_indices = sorted(sample_indices)
    
    # Create sampled versions
    df_sample = df.iloc[sample_indices].copy()
    df_engineered_sample = df_engineered.iloc[sample_indices].copy()
    
    print(f"[SUCCESS] Sampled {len(df_sample):,} transactions")
    print(f"[INFO] Sample fraud rate: {df_sample['isFraud'].mean()*100:.4f}%")
else:
    df_sample = df.copy()
    df_engineered_sample = df_engineered.copy()
    sample_indices = list(range(len(df)))

# Define entity columns for graph edges
graph_entity_cols = []
if 'card1' in df_sample.columns:
    graph_entity_cols.append('card1')
if 'addr1' in df_sample.columns:
    graph_entity_cols.append('addr1')
if 'P_emaildomain' in df_sample.columns:
    graph_entity_cols.append('P_emaildomain')

print(f"\n[INFO] Using {len(graph_entity_cols)} entity types: {graph_entity_cols}")

# Build edge list with limits
print("\n[STEP 1/6] Building edge list with memory limits...")
edge_list = []
MAX_EDGES_PER_ENTITY = 500  # Limit edges per entity group

for entity_col in tqdm(graph_entity_cols, desc="Processing entities"):
    if entity_col not in df_sample.columns:
        continue
    
    entity_data = df_sample[[entity_col]].copy()
    entity_data['node_id'] = range(len(entity_data))
    entity_data = entity_data[entity_data[entity_col].notna()]
    
    grouped = entity_data.groupby(entity_col)['node_id'].apply(list).to_dict()
    
    for entity_value, transaction_ids in grouped.items():
        num_transactions = len(transaction_ids)
        
        # Skip very large groups (memory optimization)
        if num_transactions > 100:
            continue
            
        # Create edges (limit to MAX_EDGES_PER_ENTITY)
        if num_transactions > 1:
            edges_to_create = min(
                num_transactions * (num_transactions - 1),
                MAX_EDGES_PER_ENTITY
            )
            
            # Sample pairs if needed
            if edges_to_create < num_transactions * (num_transactions - 1):
                # Random sampling of edges
                import random
                pairs = []
                for _ in range(edges_to_create // 2):
                    i, j = random.sample(transaction_ids, 2)
                    pairs.append((i, j))
                    pairs.append((j, i))
                
                for i, j in pairs:
                    edge_list.append([i, j])
            else:
                # Create all edges
                for i in range(len(transaction_ids)):
                    for j in range(i+1, len(transaction_ids)):
                        edge_list.append([transaction_ids[i], transaction_ids[j]])
                        edge_list.append([transaction_ids[j], transaction_ids[i]])

print(f"[SUCCESS] Created {len(edge_list):,} edges")

# Remove duplicates
print("\n[STEP 2/6] Removing duplicate edges...")
edge_array = np.array(edge_list) if len(edge_list) > 0 else np.empty((0, 2))

if len(edge_array) > 0:
    # Remove duplicates in chunks (memory efficient)
    print(f"[INFO] Processing {len(edge_array):,} edges...")
    edge_array = np.unique(edge_array, axis=0)
    edge_index = torch.tensor(edge_array.T, dtype=torch.long)
else:
    edge_index = torch.empty((2, 0), dtype=torch.long)

print(f"[SUCCESS] Unique edges: {edge_index.shape[1]:,}")

# Prepare node features
print("\n[STEP 3/6] Preparing node features...")
feature_cols = [col for col in df_engineered_sample.columns if col != 'isFraud']
X = df_engineered_sample[feature_cols].values
y = df_engineered_sample['isFraud'].values

print(f"[INFO] Node feature dimension: {X.shape[1]}")
print(f"[INFO] Number of nodes: {X.shape[0]:,}")

# Scale features
print("\n[STEP 4/6] Scaling node features...")
scaler = RobustScaler()
x_scaled = scaler.fit_transform(X)
x = torch.tensor(x_scaled, dtype=torch.float)
y_labels = torch.tensor(y, dtype=torch.long)

print("[SUCCESS] Features scaled")

# Create PyTorch Geometric Data object
print("\n[STEP 5/6] Creating graph data object...")
graph_data = Data(x=x, edge_index=edge_index, y=y_labels)

# Add self-loops (only if edge index is not empty)
if graph_data.edge_index.shape[1] > 0:
    graph_data.edge_index, _ = add_self_loops(
        graph_data.edge_index, 
        num_nodes=graph_data.num_nodes
    )
else:
    # Create minimal self-loops
    self_loops = torch.arange(graph_data.num_nodes).unsqueeze(0).repeat(2, 1)
    graph_data.edge_index = self_loops

print(f"[SUCCESS] Graph created")
print(f"[INFO] Nodes: {graph_data.num_nodes:,}")
print(f"[INFO] Edges: {graph_data.num_edges:,}")
print(f"[INFO] Features: {graph_data.num_node_features}")
print(f"[INFO] Average degree: {graph_data.num_edges / graph_data.num_nodes:.2f}")

# Memory estimate
edge_memory_mb = (graph_data.num_edges * 8 * 2) / (1024**2)  # 2 int64 per edge
feature_memory_mb = (graph_data.num_nodes * graph_data.num_node_features * 4) / (1024**2)
total_memory_mb = edge_memory_mb + feature_memory_mb

print(f"[INFO] Estimated memory usage: {total_memory_mb:.2f} MB")

# Degree analysis
print("\n[STEP 6/6] Analyzing graph structure...")
degrees = degree(graph_data.edge_index[0], num_nodes=graph_data.num_nodes).numpy()
print(f"[INFO] Degree stats - Min: {degrees.min()}, Max: {degrees.max()}, Mean: {degrees.mean():.2f}")

fraud_mask = graph_data.y == 1
fraud_degrees = degrees[fraud_mask]
nonfraud_degrees = degrees[~fraud_mask]

print(f"[INFO] Fraud transactions - Mean degree: {fraud_degrees.mean():.2f}")
print(f"[INFO] Non-fraud transactions - Mean degree: {nonfraud_degrees.mean():.2f}")

print("\n[SUCCESS] Graph construction complete!")
print("="*80)

# Save graph
torch.save(graph_data, 'fraud_graph_data.pt')
print("[INFO] Graph saved to 'fraud_graph_data.pt'")

gc.collect()

# %%

# %%
# ==============================================================================
# SECTION 6: MODEL ARCHITECTURE
# ==============================================================================

print("\n" + "="*80)
print("SECTION 6: MODEL ARCHITECTURE")
print("="*80)

class GraphSAGEFraudDetector(nn.Module):
    """
    Hybrid GraphSAGE + MLP model for fraud detection.
    """
    
    def __init__(self, num_node_features, hidden_channels=128, 
                 embedding_dim=64, num_layers=3, dropout=0.3):
        super(GraphSAGEFraudDetector, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.convs.append(SAGEConv(num_node_features, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(SAGEConv(hidden_channels, embedding_dim))
        self.batch_norms.append(nn.BatchNorm1d(embedding_dim))
        
        # MLP Classifier
        mlp_input_dim = embedding_dim + num_node_features
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 2)
        )
        
    def forward(self, x, edge_index):
        original_features = x
        
        # GraphSAGE propagation
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        embeddings = x
        
        # Concatenate embeddings with original features
        combined_features = torch.cat([embeddings, original_features], dim=1)
        
        # MLP classification
        logits = self.mlp(combined_features)
        
        return logits, embeddings

# Model instantiation
print("\n[STEP 1/3] Instantiating model...")

NUM_NODE_FEATURES = graph_data.num_node_features
HIDDEN_CHANNELS = 128
EMBEDDING_DIM = 64
NUM_GNN_LAYERS = 3
DROPOUT = 0.3

model = GraphSAGEFraudDetector(
    num_node_features=NUM_NODE_FEATURES,
    hidden_channels=HIDDEN_CHANNELS,
    embedding_dim=EMBEDDING_DIM,
    num_layers=NUM_GNN_LAYERS,
    dropout=DROPOUT
).to(device)

total_params = sum(p.numel() for p in model.parameters())

print(f"[SUCCESS] Model created")
print(f"[INFO] Total parameters: {total_params:,}")

# Create data splits
print("\n[STEP 2/3] Creating train/val/test splits...")

num_nodes = graph_data.num_nodes
indices = np.arange(num_nodes)
labels = graph_data.y.numpy()

train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, random_state=RANDOM_SEED, stratify=labels
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=RANDOM_SEED, stratify=labels[temp_idx]
)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

graph_data.train_mask = train_mask
graph_data.val_mask = val_mask
graph_data.test_mask = test_mask

graph_data = graph_data.to(device)

print(f"[SUCCESS] Data split created")
print(f"[INFO] Training: {train_mask.sum():,} ({train_mask.sum()/num_nodes*100:.1f}%)")
print(f"[INFO] Validation: {val_mask.sum():,} ({val_mask.sum()/num_nodes*100:.1f}%)")
print(f"[INFO] Test: {test_mask.sum():,} ({test_mask.sum()/num_nodes*100:.1f}%)")

# Handle class imbalance
print("\n[STEP 3/3] Setting up class weights...")

train_labels = graph_data.y[train_mask].cpu().numpy()
class_counts = np.bincount(train_labels)
class_weights = len(train_labels) / (len(class_counts) * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

print(f"[INFO] Non-fraud: {class_counts[0]:,}, Fraud: {class_counts[1]:,}")
print(f"[INFO] Fraud weight: {class_weights[1]:.2f}x")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

print("\n[SUCCESS] Model architecture complete!")
print("="*80)

# %%
# ==============================================================================
# SECTION 7: TRAINING LOOP
# ==============================================================================

print("\n" + "="*80)
print("SECTION 7: TRAINING")
print("="*80)

def calculate_metrics(y_true, y_pred, y_proba):
    metrics = {
        'accuracy': (y_true == y_pred).mean(),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba)
    }
    return metrics

def train_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    logits, embeddings = model(data.x, data.edge_index)
    loss = criterion(logits[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        train_logits = logits[data.train_mask]
        train_proba = F.softmax(train_logits, dim=1)[:, 1].cpu().numpy()
        train_pred = train_logits.argmax(dim=1).cpu().numpy()
        train_true = data.y[data.train_mask].cpu().numpy()
        
        metrics = calculate_metrics(train_true, train_pred, train_proba)
    
    return loss.item(), metrics

@torch.no_grad()
def validate(model, data, criterion, mask):
    model.eval()
    
    logits, embeddings = model(data.x, data.edge_index)
    loss = criterion(logits[mask], data.y[mask])
    
    val_logits = logits[mask]
    val_proba = F.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
    val_pred = val_logits.argmax(dim=1).cpu().numpy()
    val_true = data.y[mask].cpu().numpy()
    
    metrics = calculate_metrics(val_true, val_pred, val_proba)
    
    return loss.item(), metrics, val_proba, val_pred

# Training configuration
NUM_EPOCHS = 50
PATIENCE = 10

history = {
    'train_loss': [], 'val_loss': [],
    'train_auc': [], 'val_auc': [],
    'train_pr_auc': [], 'val_pr_auc': []
}

best_val_auc = 0.0
best_epoch = 0
patience_counter = 0
best_model_state = None

print(f"\n[INFO] Starting training for max {NUM_EPOCHS} epochs...")
print(f"[INFO] Early stopping patience: {PATIENCE}")
print("-"*80)

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_metrics = train_epoch(model, graph_data, optimizer, criterion)
    val_loss, val_metrics, _, _ = validate(model, graph_data, criterion, graph_data.val_mask)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_auc'].append(train_metrics['roc_auc'])
    history['val_auc'].append(val_metrics['roc_auc'])
    history['train_pr_auc'].append(train_metrics['pr_auc'])
    history['val_pr_auc'].append(val_metrics['pr_auc'])
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"AUC: {train_metrics['roc_auc']:.4f}/{val_metrics['roc_auc']:.4f}")
    
    scheduler.step(val_metrics['roc_auc'])
    
    if val_metrics['roc_auc'] > best_val_auc:
        best_val_auc = val_metrics['roc_auc']
        best_epoch = epoch
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model_state,
            'val_auc': best_val_auc,
        }, 'best_model.pt')
    else:
        patience_counter += 1
    
    if patience_counter >= PATIENCE:
        print(f"\n[INFO] Early stopping at epoch {epoch}")
        break

# Load best model
model.load_state_dict(best_model_state)
print(f"\n[SUCCESS] Training complete!")
print(f"[INFO] Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
print("="*80)

gc.collect()

# %%
# ==============================================================================
# SECTION 8: EVALUATION
# ==============================================================================

print("\n" + "="*80)
print("SECTION 8: MODEL EVALUATION")
print("="*80)

# Test set evaluation
print("\n[STEP 1/4] Evaluating on test set...")
test_loss, test_metrics, test_proba, test_pred = validate(
    model, graph_data, criterion, graph_data.test_mask
)

test_true = graph_data.y[graph_data.test_mask].cpu().numpy()

print(f"\n[SUCCESS] Test Set Results:")
print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
print(f"  PR-AUC:    {test_metrics['pr_auc']:.4f}")
print(f"  F1-Score:  {test_metrics['f1']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall:    {test_metrics['recall']:.4f}")

# Confusion matrix
print("\n[STEP 2/4] Computing confusion matrix...")
cm = confusion_matrix(test_true, test_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {tn:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}")
print(f"  True Positives:  {tp:,}")
print(f"\n  Fraud Detection Rate: {tp/(tp+fn)*100:.2f}%")
print(f"  False Alarm Rate: {fp/(fp+tn)*100:.4f}%")

# Plot confusion matrix
print("\n[STEP 3/4] Creating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Non-Fraud', 'Fraud'],
            yticklabels=['Non-Fraud', 'Fraud'])
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('True')
axes[0, 0].set_title('Confusion Matrix')

# ROC Curve
fpr, tpr, _ = roc_curve(test_true, test_proba)
axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC={test_metrics["roc_auc"]:.4f})')
axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(test_true, test_proba)
axes[1, 0].plot(recall_vals, precision_vals, label=f'PR (AUC={test_metrics["pr_auc"]:.4f})')
axes[1, 0].axhline(y=test_true.mean(), color='r', linestyle='--', label='Baseline')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision-Recall Curve')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Training History
axes[1, 1].plot(history['train_auc'], label='Train AUC')
axes[1, 1].plot(history['val_auc'], label='Val AUC')
axes[1, 1].axhline(y=best_val_auc, color='r', linestyle='--', label=f'Best: {best_val_auc:.4f}')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('ROC-AUC')
axes[1, 1].set_title('Training History')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Saved 'evaluation_results.png'")

# Classification report
print("\n[STEP 4/4] Classification Report:")
print(classification_report(test_true, test_pred, 
                          target_names=['Non-Fraud', 'Fraud'],
                          digits=4))

print("\n[SUCCESS] Evaluation complete!")
print("="*80)

# %%
# ==============================================================================
# SECTION 9: EXPLAINABLE AI (SIMPLIFIED)
# ==============================================================================

print("\n" + "="*80)
print("SECTION 9: EXPLAINABLE AI")
print("="*80)

# Extract embeddings
print("\n[STEP 1/3] Extracting model embeddings...")
model.eval()
with torch.no_grad():
    all_logits, all_embeddings = model(graph_data.x, graph_data.edge_index)
    all_proba = F.softmax(all_logits, dim=1).cpu().numpy()
    all_embeddings_np = all_embeddings.cpu().numpy()

print("[SUCCESS] Embeddings extracted")

# Feature importance (simplified - using embedding vs original feature contribution)
print("\n[STEP 2/3] Analyzing feature importance...")

test_indices = np.where(graph_data.test_mask.cpu().numpy())[0]
test_embeddings = all_embeddings_np[test_indices]
test_original = graph_data.x.cpu().numpy()[test_indices]

gnn_importance = np.abs(test_embeddings).mean()
original_importance = np.abs(test_original).mean()
total = gnn_importance + original_importance

print(f"\n[INFO] Feature Contribution:")
print(f"  GNN Embeddings: {gnn_importance/total*100:.2f}%")
print(f"  Original Features: {original_importance/total*100:.2f}%")

# Individual predictions
print("\n[STEP 3/3] Analyzing predictions...")

fraud_predictions = np.where(test_pred == 1)[0]
nonfraud_predictions = np.where(test_pred == 0)[0]

if len(fraud_predictions) > 0:
    fraud_confidences = test_proba[fraud_predictions]
    top_fraud_idx = fraud_predictions[np.argsort(fraud_confidences)[-3:]]
    
    print(f"\n[INFO] Top 3 High-Confidence Fraud Predictions:")
    for i, idx in enumerate(top_fraud_idx[::-1], 1):
        actual = test_true[idx]
        prob = test_proba[idx]
        print(f"  {i}. Confidence: {prob:.4f} | Actual: {'FRAUD' if actual==1 else 'NON-FRAUD'}")

print("\n[SUCCESS] Explainability analysis complete!")
print("="*80)

# %%
# ==============================================================================
# SECTION 10: FINAL SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("SECTION 10: PROJECT SUMMARY")
print("="*80)

print(f"""
FRAUD DETECTION PROJECT - COMPLETE

Dataset:
  Total Transactions: {df.shape[0]:,}
  Features: {df_engineered.shape[1]-1}
  Fraud Rate: {(fraud_counts[1]/len(df)*100):.4f}%

Graph Structure:
  Nodes: {graph_data.num_nodes:,}
  Edges: {graph_data.num_edges:,}
  Average Degree: {graph_data.num_edges/graph_data.num_nodes:.2f}

Model:
  Architecture: Hybrid GraphSAGE + MLP
  Parameters: {total_params:,}
  Training Epochs: {best_epoch}
  Best Val AUC: {best_val_auc:.4f}

Test Performance:
  ROC-AUC: {test_metrics['roc_auc']:.4f}
  PR-AUC: {test_metrics['pr_auc']:.4f}
  F1-Score: {test_metrics['f1']:.4f}
  Precision: {test_metrics['precision']:.4f}
  Recall: {test_metrics['recall']:.4f}
  
  Frauds Caught: {tp}/{tp+fn} ({tp/(tp+fn)*100:.2f}%)
  False Alarms: {fp}/{fp+tn} ({fp/(fp+tn)*100:.4f}%)

Files Created:
  - fraud_graph_data.pt (graph structure)
  - best_model.pt (trained model)
  - evaluation_results.png (visualizations)

[SUCCESS] PROJECT COMPLETE!
""")

print("="*80)
print("Thank you for using this fraud detection system!")
print("="*80)
