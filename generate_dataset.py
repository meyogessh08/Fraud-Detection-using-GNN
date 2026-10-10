import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 80)
print("GENERATING ENHANCED FRAUD DETECTION DATASET")
print("=" * 80)

# Generate Enhanced Fraud Detection Dataset
n_samples = 10000
fraud_ratio = 0.15  # 15% fraud for better learning

# Initialize lists
data = []

# Risk categories and actions mapping
risk_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
actions_map = {
    'LOW': 'APPROVE',
    'MEDIUM': 'APPROVE_WITH_MONITORING',
    'HIGH': 'CHALLENGE_WITH_2FA',
    'CRITICAL': 'BLOCK_AND_ALERT'
}

print(f"\n📊 Generating {n_samples} samples with {fraud_ratio*100:.0f}% fraud rate...")

for i in range(n_samples):
    if (i + 1) % 1000 == 0:
        print(f"  Progress: {i+1}/{n_samples} samples generated...")
    
    is_fraud = 1 if random.random() < fraud_ratio else 0
    
    # Transaction Features
    if is_fraud:
        amount = np.random.exponential(scale=200) + 50
        prev_txns = np.random.randint(30, 100)
        velocity_24h = np.random.randint(15, 50)
        failed_attempts = np.random.randint(1, 5)
        distance_from_home = np.random.uniform(200, 2000)
        unusual_time = 1 if random.random() < 0.7 else 0
        new_device = 1 if random.random() < 0.8 else 0
        new_location = 1 if random.random() < 0.75 else 0
        high_risk_merchant = 1 if random.random() < 0.6 else 0
        suspicious_email = 1 if random.random() < 0.5 else 0
        international = 1 if random.random() < 0.6 else 0
        card_age_days = np.random.randint(1, 30)
        
        # Risk scoring (fraud cases have higher risk)
        risk_score = np.random.uniform(60, 100)
        if risk_score >= 85:
            risk_level = 'CRITICAL'
        elif risk_score >= 70:
            risk_level = 'HIGH'
        elif risk_score >= 55:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
            
    else:
        amount = np.random.exponential(scale=50) + 10
        prev_txns = np.random.randint(5, 30)
        velocity_24h = np.random.randint(1, 10)
        failed_attempts = 0 if random.random() < 0.9 else 1
        distance_from_home = np.random.uniform(0, 100)
        unusual_time = 0 if random.random() < 0.8 else 1
        new_device = 0 if random.random() < 0.9 else 1
        new_location = 0 if random.random() < 0.85 else 1
        high_risk_merchant = 0 if random.random() < 0.95 else 1
        suspicious_email = 0 if random.random() < 0.95 else 1
        international = 0 if random.random() < 0.9 else 1
        card_age_days = np.random.randint(30, 1000)
        
        # Risk scoring (legitimate cases have lower risk)
        risk_score = np.random.uniform(0, 50)
        if risk_score >= 35:
            risk_level = 'MEDIUM'
        elif risk_score >= 20:
            risk_level = 'LOW'
        else:
            risk_level = 'LOW'
    
    # Recommended action based on risk level
    recommended_action = actions_map[risk_level]
    
    # Risk factors explanation (comma-separated)
    risk_factors = []
    if amount > 200:
        risk_factors.append(f"High_Amount_${amount:.2f}")
    if velocity_24h > 15:
        risk_factors.append(f"High_Velocity_{velocity_24h}_txns")
    if failed_attempts > 0:
        risk_factors.append(f"Failed_Attempts_{failed_attempts}")
    if distance_from_home > 200:
        risk_factors.append(f"Distance_{distance_from_home:.0f}km")
    if unusual_time:
        risk_factors.append("Unusual_Time")
    if new_device:
        risk_factors.append("New_Device")
    if new_location:
        risk_factors.append("New_Location")
    if high_risk_merchant:
        risk_factors.append("High_Risk_Merchant")
    if suspicious_email:
        risk_factors.append("Suspicious_Email")
    if international:
        risk_factors.append("International_Transaction")
    if card_age_days < 30:
        risk_factors.append(f"New_Card_{card_age_days}_days")
    
    risk_factors_str = "|".join(risk_factors) if risk_factors else "None"
    
    # Protective factors
    protective_factors = []
    if amount < 50:
        protective_factors.append("Normal_Amount")
    if velocity_24h < 5:
        protective_factors.append("Normal_Velocity")
    if failed_attempts == 0:
        protective_factors.append("No_Failed_Attempts")
    if distance_from_home < 50:
        protective_factors.append("Near_Home")
    if not unusual_time:
        protective_factors.append("Normal_Hours")
    if card_age_days > 365:
        protective_factors.append("Established_Card")
    
    protective_factors_str = "|".join(protective_factors) if protective_factors else "None"
    
    # Confidence score (model's certainty)
    if is_fraud:
        confidence = np.random.uniform(0.7, 0.99)
    else:
        confidence = np.random.uniform(0.6, 0.95)
    
    # Explanation text
    if is_fraud and risk_level == 'CRITICAL':
        explanation = f"CRITICAL RISK: Transaction flagged due to {len(risk_factors)} risk factors including high amount (${amount:.2f}), velocity ({velocity_24h} txns/24h), and suspicious patterns. Immediate blocking recommended."
    elif is_fraud and risk_level == 'HIGH':
        explanation = f"HIGH RISK: Multiple suspicious indicators detected. Amount ${amount:.2f} with {velocity_24h} recent transactions. Additional verification required."
    elif risk_level == 'MEDIUM':
        explanation = f"MEDIUM RISK: Some unusual patterns detected but within acceptable range. Monitor transaction closely."
    else:
        explanation = f"LOW RISK: Transaction appears legitimate with {len(protective_factors)} protective factors. Normal approval recommended."
    
    # Fraud probability
    fraud_probability = risk_score / 100.0 if is_fraud else (1 - risk_score / 100.0)
    
    # Create row
    row = {
        'transaction_id': f'TXN_{i:06d}',
        'amount': round(amount, 2),
        'card_id': f'CARD_{random.randint(1000, 9999)}',
        'prev_transactions_24h': prev_txns,
        'velocity_24h': velocity_24h,
        'failed_attempts': failed_attempts,
        'distance_from_home_km': round(distance_from_home, 2),
        'unusual_time': unusual_time,
        'new_device': new_device,
        'new_location': new_location,
        'high_risk_merchant': high_risk_merchant,
        'suspicious_email': suspicious_email,
        'international_txn': international,
        'card_age_days': card_age_days,
        'hour': random.randint(0, 23),
        'day_of_week': random.randint(0, 6),
        'risk_score': round(risk_score, 2),
        'risk_level': risk_level,
        'fraud_probability': round(fraud_probability, 4),
        'confidence_score': round(confidence, 4),
        'recommended_action': recommended_action,
        'risk_factors': risk_factors_str,
        'protective_factors': protective_factors_str,
        'explanation': explanation,
        'is_fraud': is_fraud
    }
    
    data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('fraud_training_enhanced.csv', index=False)

print("\n✅ Dataset created successfully!")
print(f"📊 Total samples: {len(df)}")
print(f"🔴 Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
print(f"🟢 Legitimate cases: {(~df['is_fraud'].astype(bool)).sum()} ({(1-df['is_fraud'].mean())*100:.2f}%)")
print(f"\n📈 Risk Level Distribution:")
print(df['risk_level'].value_counts())
print(f"\n🎯 Recommended Action Distribution:")
print(df['recommended_action'].value_counts())
print(f"\n💾 Dataset saved as: fraud_training_enhanced.csv")
print("=" * 80)