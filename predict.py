import torch
import pandas as pd
import numpy as np
import pickle
from model import EnhancedFraudDetectionModel

def load_trained_model(model_path='fraud_detection_complete.pth'):
    """Load the trained model and associated components"""
    print("📦 Loading trained model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize model
    input_dim = len(checkpoint['feature_columns'])
    model = EnhancedFraudDetectionModel(input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ Model loaded successfully")
    
    return {
        'model': model,
        'scaler': checkpoint['scaler'],
        'risk_level_encoder': checkpoint['risk_level_encoder'],
        'action_encoder': checkpoint['action_encoder'],
        'feature_columns': checkpoint['feature_columns']
    }

def generate_explanation(features, feature_names, fraud_pred, risk_score, risk_level):
    """Generate human-readable explanation"""
    
    # Map feature indices
    feature_dict = dict(zip(feature_names, features))
    
    risk_factors = []
    protective_factors = []
    
    # Analyze features
    if feature_dict.get('amount', 0) > 200:
        risk_factors.append(f"High transaction amount (${feature_dict['amount']:.2f})")
    elif feature_dict.get('amount', 0) < 50:
        protective_factors.append("Normal transaction amount")
    
    if feature_dict.get('velocity_24h', 0) > 15:
        risk_factors.append(f"High transaction velocity ({int(feature_dict['velocity_24h'])} txns/24h)")
    elif feature_dict.get('velocity_24h', 0) < 5:
        protective_factors.append("Normal transaction velocity")
    
    if feature_dict.get('failed_attempts', 0) > 0:
        risk_factors.append(f"Failed authentication attempts ({int(feature_dict['failed_attempts'])})")
    else:
        protective_factors.append("No failed attempts")
    
    if feature_dict.get('distance_from_home_km', 0) > 200:
        risk_factors.append(f"Unusual distance from home ({feature_dict['distance_from_home_km']:.0f} km)")
    elif feature_dict.get('distance_from_home_km', 0) < 50:
        protective_factors.append("Transaction near home location")
    
    if feature_dict.get('new_device', 0) == 1:
        risk_factors.append("New device detected")
    
    if feature_dict.get('unusual_time', 0) == 1:
        risk_factors.append("Unusual transaction time")
    else:
        protective_factors.append("Normal transaction hours")
    
    if feature_dict.get('high_risk_merchant', 0) == 1:
        risk_factors.append("High-risk merchant category")
    
    if feature_dict.get('new_location', 0) == 1:
        risk_factors.append("New location detected")
    
    if feature_dict.get('suspicious_email', 0) == 1:
        risk_factors.append("Suspicious email domain")
    
    if feature_dict.get('international_txn', 0) == 1:
        risk_factors.append("International transaction")
    
    if feature_dict.get('card_age_days', 0) < 30:
        risk_factors.append(f"New card ({int(feature_dict['card_age_days'])} days old)")
    elif feature_dict.get('card_age_days', 0) > 365:
        protective_factors.append("Established card")
    
    # Build explanation
    if fraud_pred == 1:
        explanation = f"⚠️ FRAUD DETECTED ({risk_level} RISK)\n\n"
        explanation += f"Risk Score: {risk_score:.2f}/100\n\n"
        explanation += f"Risk Factors Identified ({len(risk_factors)}):\n"
        for factor in risk_factors:
            explanation += f"  • {factor}\n"
    else:
        explanation = f"✓ LEGITIMATE TRANSACTION ({risk_level} RISK)\n\n"
        explanation += f"Risk Score: {risk_score:.2f}/100\n\n"
        if protective_factors:
            explanation += f"Protective Factors ({len(protective_factors)}):\n"
            for factor in protective_factors:
                explanation += f"  • {factor}\n"
    
    return explanation

def predict_transaction(transaction_data, model_components):
    """
    Make prediction with detailed explanation
    
    Args:
        transaction_data: dict with transaction features
        model_components: dict from load_trained_model()
    
    Returns:
        dict with prediction results
    """
    model = model_components['model']
    scaler = model_components['scaler']
    risk_level_encoder = model_components['risk_level_encoder']
    action_encoder = model_components['action_encoder']
    feature_columns = model_components['feature_columns']
    
    # Extract features in correct order
    features = [transaction_data[col] for col in feature_columns]
    
    # Prepare input
    features_scaled = scaler.transform([features])
    features_tensor = torch.FloatTensor(features_scaled)
    
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
    
    # Get predictions
    fraud_probs = torch.softmax(outputs['fraud_logits'], dim=1)[0]
    fraud_pred = torch.argmax(fraud_probs).item()
    fraud_confidence = fraud_probs[fraud_pred].item()
    
    risk_score = outputs['risk_score'][0].item()
    
    risk_level_probs = torch.softmax(outputs['risk_level_logits'], dim=1)[0]
    risk_level_pred = torch.argmax(risk_level_probs).item()
    risk_level_name = risk_level_encoder.inverse_transform([risk_level_pred])[0]
    
    action_probs = torch.softmax(outputs['action_logits'], dim=1)[0]
    action_pred = torch.argmax(action_probs).item()
    action_name = action_encoder.inverse_transform([action_pred])[0]
    
    # Create explanation
    explanation = generate_explanation(features, feature_columns, fraud_pred, 
                                      risk_score, risk_level_name)
    
    result = {
        'fraud_prediction': 'FRAUD' if fraud_pred == 1 else 'LEGITIMATE',
        'fraud_probability': fraud_probs[1].item(),
        'confidence': fraud_confidence,
        'risk_score': risk_score,
        'risk_level': risk_level_name,
        'recommended_action': action_name,
        'explanation': explanation
    }
    
    return result

def main():
    """Example usage"""
    print("="*80)
    print("FRAUD DETECTION MODEL - PREDICTION EXAMPLE")
    print("="*80)
    
    # Load model
    model_components = load_trained_model()
    
    # Example transaction 1: High-risk fraud scenario
    print("\n" + "─"*80)
    print("EXAMPLE 1: High-Risk Transaction")
    print("─"*80)
    
    transaction1 = {
        'amount': 350.00,
        'prev_transactions_24h': 45,
        'velocity_24h': 25,
        'failed_attempts': 3,
        'distance_from_home_km': 500.0,
        'unusual_time': 1,
        'new_device': 1,
        'new_location': 1,
        'high_risk_merchant': 1,
        'suspicious_email': 1,
        'international_txn': 1,
        'card_age_days': 5,
        'hour': 3,
        'day_of_week': 2
    }
    
    result1 = predict_transaction(transaction1, model_components)
    
    print(f"\n🔍 Prediction: {result1['fraud_prediction']}")
    print(f"📊 Fraud Probability: {result1['fraud_probability']:.2%}")
    print(f"✅ Confidence: {result1['confidence']:.2%}")
    print(f"⚠️ Risk Level: {result1['risk_level']}")
    print(f"🎯 Recommended Action: {result1['recommended_action']}")
    print(f"\n{result1['explanation']}")
    
    # Example transaction 2: Low-risk legitimate scenario
    print("\n" + "─"*80)
    print("EXAMPLE 2: Low-Risk Transaction")
    print("─"*80)
    
    transaction2 = {
        'amount': 35.50,
        'prev_transactions_24h': 12,
        'velocity_24h': 3,
        'failed_attempts': 0,
        'distance_from_home_km': 5.0,
        'unusual_time': 0,
        'new_device': 0,
        'new_location': 0,
        'high_risk_merchant': 0,
        'suspicious_email': 0,
        'international_txn': 0,
        'card_age_days': 450,
        'hour': 14,
        'day_of_week': 3
    }
    
    result2 = predict_transaction(transaction2, model_components)
    
    print(f"\n🔍 Prediction: {result2['fraud_prediction']}")
    print(f"📊 Fraud Probability: {result2['fraud_probability']:.2%}")
    print(f"✅ Confidence: {result2['confidence']:.2%}")
    print(f"⚠️ Risk Level: {result2['risk_level']}")
    print(f"🎯 Recommended Action: {result2['recommended_action']}")
    print(f"\n{result2['explanation']}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()