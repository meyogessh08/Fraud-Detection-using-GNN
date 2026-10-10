import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from model import EnhancedFraudDetectionModel

# ═══════════════════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════════

class FraudDataset(Dataset):
    """Enhanced Fraud Detection Dataset with Risk Analysis"""
    
    def __init__(self, features, labels, risk_scores, risk_levels, actions):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.risk_scores = torch.FloatTensor(risk_scores)
        self.risk_levels = torch.LongTensor(risk_levels)
        self.actions = torch.LongTensor(actions)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx],
            'risk_score': self.risk_scores[idx],
            'risk_level': self.risk_levels[idx],
            'action': self.actions[idx]
        }

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def train_enhanced_model(model, train_loader, val_loader, epochs=50, device='cpu'):
    """Train the enhanced multi-task fraud detection model"""
    
    # Loss functions
    fraud_criterion = nn.CrossEntropyLoss()
    risk_score_criterion = nn.MSELoss()
    risk_level_criterion = nn.CrossEntropyLoss()
    action_criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_fraud_acc': [],
        'val_fraud_acc': [],
        'train_action_acc': [],
        'val_action_acc': []
    }
    
    best_val_loss = float('inf')
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    for epoch in range(epochs):
        # ─── Training Phase ─────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_fraud_correct = 0
        train_action_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(device)
            fraud_labels = batch['label'].to(device)
            risk_scores = batch['risk_score'].to(device).unsqueeze(1)
            risk_levels = batch['risk_level'].to(device)
            actions = batch['action'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            
            # Compute losses
            fraud_loss = fraud_criterion(outputs['fraud_logits'], fraud_labels)
            risk_score_loss = risk_score_criterion(outputs['risk_score'], risk_scores)
            risk_level_loss = risk_level_criterion(outputs['risk_level_logits'], risk_levels)
            action_loss = action_criterion(outputs['action_logits'], actions)
            
            # Combined loss with weights
            total_loss = (
                2.0 * fraud_loss +
                1.0 * risk_score_loss +
                1.5 * risk_level_loss +
                1.5 * action_loss
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            # Calculate accuracies
            fraud_preds = torch.argmax(outputs['fraud_logits'], dim=1)
            train_fraud_correct += (fraud_preds == fraud_labels).sum().item()
            
            action_preds = torch.argmax(outputs['action_logits'], dim=1)
            train_action_correct += (action_preds == actions).sum().item()
            
            train_total += fraud_labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_fraud_acc = train_fraud_correct / train_total
        train_action_acc = train_action_correct / train_total
        
        # ─── Validation Phase ───────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_fraud_correct = 0
        val_action_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                fraud_labels = batch['label'].to(device)
                risk_scores = batch['risk_score'].to(device).unsqueeze(1)
                risk_levels = batch['risk_level'].to(device)
                actions = batch['action'].to(device)
                
                # Forward pass
                outputs = model(features)
                
                # Compute losses
                fraud_loss = fraud_criterion(outputs['fraud_logits'], fraud_labels)
                risk_score_loss = risk_score_criterion(outputs['risk_score'], risk_scores)
                risk_level_loss = risk_level_criterion(outputs['risk_level_logits'], risk_levels)
                action_loss = action_criterion(outputs['action_logits'], actions)
                
                total_loss = (
                    2.0 * fraud_loss +
                    1.0 * risk_score_loss +
                    1.5 * risk_level_loss +
                    1.5 * action_loss
                )
                
                val_loss += total_loss.item()
                
                # Calculate accuracies
                fraud_preds = torch.argmax(outputs['fraud_logits'], dim=1)
                val_fraud_correct += (fraud_preds == fraud_labels).sum().item()
                
                action_preds = torch.argmax(outputs['action_logits'], dim=1)
                val_action_correct += (action_preds == actions).sum().item()
                
                val_total += fraud_labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_fraud_acc = val_fraud_correct / val_total
        val_action_acc = val_action_correct / val_total
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_fraud_acc'].append(train_fraud_acc)
        history['val_fraud_acc'].append(val_fraud_acc)
        history['train_action_acc'].append(train_action_acc)
        history['val_action_acc'].append(val_action_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train Fraud Acc: {train_fraud_acc:.4f} | Train Action Acc: {train_action_acc:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f} | Val Fraud Acc: {val_fraud_acc:.4f} | Val Action Acc: {val_action_acc:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_fraud_model.pth')
            if (epoch + 1) % 5 == 0:
                print(f"  ✓ Best model saved (Val Loss: {best_val_loss:.4f})")
    
    return model, history

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("="*80)
    print("ENHANCED FRAUD DETECTION MODEL TRAINING")
    print("="*80)
    
    # ─── Load Data ──────────────────────────────────────────────────────
    print("\n📂 Loading dataset...")
    try:
        df = pd.read_csv('fraud_training_enhanced.csv')
        print(f"✓ Loaded {len(df)} samples")
        print(f"✓ Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    except FileNotFoundError:
        print("❌ Error: fraud_training_enhanced.csv not found!")
        print("   Please run 'python generate_dataset.py' first.")
        return
    
    # ─── Prepare Features ───────────────────────────────────────────────
    print("\n🔧 Preparing features...")
    
    feature_columns = [
        'amount', 'prev_transactions_24h', 'velocity_24h', 'failed_attempts',
        'distance_from_home_km', 'unusual_time', 'new_device', 'new_location',
        'high_risk_merchant', 'suspicious_email', 'international_txn',
        'card_age_days', 'hour', 'day_of_week'
    ]
    
    X = df[feature_columns].values
    y_fraud = df['is_fraud'].values
    y_risk_score = df['risk_score'].values
    
    # Encode categorical targets
    risk_level_encoder = LabelEncoder()
    y_risk_level = risk_level_encoder.fit_transform(df['risk_level'])
    
    action_encoder = LabelEncoder()
    y_action = action_encoder.fit_transform(df['recommended_action'])
    
    print(f"✓ Feature shape: {X.shape}")
    print(f"✓ Risk levels: {list(risk_level_encoder.classes_)}")
    print(f"✓ Actions: {list(action_encoder.classes_)}")
    
    # ─── Train-Test Split ───────────────────────────────────────────────
    print("\n✂️ Splitting data (80% train, 20% test)...")
    X_train, X_test, y_fraud_train, y_fraud_test, y_risk_train, y_risk_test, \
    y_level_train, y_level_test, y_action_train, y_action_test = train_test_split(
        X, y_fraud, y_risk_score, y_risk_level, y_action,
        test_size=0.2, random_state=42, stratify=y_fraud
    )
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Testing samples: {len(X_test)}")
    
    # ─── Normalize Features ─────────────────────────────────────────────
    print("\n📏 Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features normalized")
    
    # ─── Create DataLoaders ─────────────────────────────────────────────
    print("\n🔄 Creating data loaders...")
    train_dataset = FraudDataset(X_train_scaled, y_fraud_train, y_risk_train, y_level_train, y_action_train)
    test_dataset = FraudDataset(X_test_scaled, y_fraud_test, y_risk_test, y_level_test, y_action_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # ─── Initialize Model ───────────────────────────────────────────────
    print("\n🏗️ Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    model = EnhancedFraudDetectionModel(input_dim=X_train.shape[1])
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total trainable parameters: {total_params:,}")
    
    # ─── Train Model ────────────────────────────────────────────────────
    print("\n🚀 Starting training (50 epochs)...")
    
    model, history = train_enhanced_model(
        model, train_loader, test_loader,
        epochs=50, device=device
    )
    
    print("\n✅ Training completed!")
    
    # ─── Evaluate on Test Set ───────────────────────────────────────────
    print("\n📊 Evaluating on test set...")
    model.eval()
    
    all_fraud_preds = []
    all_fraud_labels = []
    all_action_preds = []
    all_action_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            fraud_labels = batch['label'].to(device)
            actions = batch['action'].to(device)
            
            outputs = model(features)
            
            fraud_preds = torch.argmax(outputs['fraud_logits'], dim=1)
            action_preds = torch.argmax(outputs['action_logits'], dim=1)
            
            all_fraud_preds.extend(fraud_preds.cpu().numpy())
            all_fraud_labels.extend(fraud_labels.cpu().numpy())
            all_action_preds.extend(action_preds.cpu().numpy())
            all_action_labels.extend(actions.cpu().numpy())
    
    # Fraud detection metrics
    fraud_acc = accuracy_score(all_fraud_labels, all_fraud_preds)
    print(f"\n🎯 Test Set Performance:")
    print(f"  Fraud Detection Accuracy: {fraud_acc:.4f}")
    print(f"  Action Recommendation Accuracy: {accuracy_score(all_action_labels, all_action_preds):.4f}")
    
    print("\n📋 Fraud Detection Classification Report:")
    print(classification_report(all_fraud_labels, all_fraud_preds, 
                               target_names=['Legitimate', 'Fraud']))
    
    # ─── Save Everything ────────────────────────────────────────────────
    print("\n💾 Saving model and components...")
    
    # Save complete checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'risk_level_encoder': risk_level_encoder,
        'action_encoder': action_encoder,
        'feature_columns': feature_columns,
        'history': history
    }, 'fraud_detection_complete.pth')
    print("✓ Model saved as 'fraud_detection_complete.pth'")
    
    # Save encoders separately for easy access
    with open('encoders.pkl', 'wb') as f:
        pickle.dump({
            'risk_level_encoder': risk_level_encoder,
            'action_encoder': action_encoder
        }, f)
    print("✓ Encoders saved as 'encoders.pkl'")
    
    # ─── Plot Training History ──────────────────────────────────────────
    print("\n📊 Plotting training history...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Fraud accuracy plot
    axes[0, 1].plot(history['train_fraud_acc'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history['val_fraud_acc'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Fraud Detection Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Action accuracy plot
    axes[1, 0].plot(history['train_action_acc'], label='Train Accuracy', linewidth=2)
    axes[1, 0].plot(history['val_action_acc'], label='Val Accuracy', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Action Recommendation Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confusion matrix
    cm = confusion_matrix(all_fraud_labels, all_fraud_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].set_title('Confusion Matrix (Fraud Detection)')
    axes[1, 1].set_xticklabels(['Legitimate', 'Fraud'])
    axes[1, 1].set_yticklabels(['Legitimate', 'Fraud'])
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history saved as 'training_history.png'")
    
    print("\n" + "="*80)
    print("✅ ALL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. fraud_detection_complete.pth  (complete model)")
    print("  2. best_fraud_model.pth          (best weights)")
    print("  3. encoders.pkl                  (label encoders)")
    print("  4. training_history.png          (training plots)")

if __name__ == "__main__":
    main()