import torch
import torch.nn as nn

class EnhancedFraudDetectionModel(nn.Module):
    """
    Multi-Task Learning Model for:
    1. Fraud Classification (binary)
    2. Risk Score Prediction (regression)
    3. Risk Level Classification (4 classes)
    4. Action Recommendation (4 classes)
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(EnhancedFraudDetectionModel, self).__init__()
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.Dropout(0.2)
        )
        
        # Task-specific heads
        
        # 1. Fraud Classification Head (binary)
        self.fraud_classifier = nn.Sequential(
            nn.Linear(hidden_dims[2], 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # 2. Risk Score Prediction Head (regression)
        self.risk_regressor = nn.Sequential(
            nn.Linear(hidden_dims[2], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 3. Risk Level Classification Head (4 classes)
        self.risk_level_classifier = nn.Sequential(
            nn.Linear(hidden_dims[2], 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        
        # 4. Action Recommendation Head (4 classes)
        self.action_classifier = nn.Sequential(
            nn.Linear(hidden_dims[2], 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
    
    def forward(self, x):
        # Shared feature extraction
        shared_features = self.shared_layers(x)
        
        # Task-specific predictions
        fraud_logits = self.fraud_classifier(shared_features)
        risk_score = self.risk_regressor(shared_features)
        risk_level_logits = self.risk_level_classifier(shared_features)
        action_logits = self.action_classifier(shared_features)
        
        return {
            'fraud_logits': fraud_logits,
            'risk_score': risk_score,
            'risk_level_logits': risk_level_logits,
            'action_logits': action_logits
        }