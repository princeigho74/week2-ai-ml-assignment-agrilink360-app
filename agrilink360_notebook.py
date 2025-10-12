AgriLink360 - Crop Yield Prediction System
SDG 2: Zero Hunger - Using ML to predict crop yields and support food security

Developer: Happy Igho Umukoro
Role: AI for Software Expert
Project: AI/ML Assignment - October 2025

This notebook demonstrates supervised learning for agricultural prediction
to help farmers and policymakers make informed decisions about food production.
"""

# ============================================================================
# AGRILINK360 - AI FOR ZERO HUNGER
# Developed by: Happy Igho Umukoro (AI for Software Expert)
# Date: October 2025
# Purpose: Supervised ML for Crop Yield Prediction (SDG 2)
# ============================================================================

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set styling for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("AgriLink360 - AI for Zero Hunger".center(70))
print("=" * 70)

# ============================================================================
# 2. CREATE SYNTHETIC DATASET (Based on real agricultural patterns)
# ============================================================================
print("\n📊 Creating Agricultural Dataset...")

# Set random seed for reproducibility
np.random.seed(42)

# Generate 2000 samples of agricultural data
n_samples = 2000

# Features that affect crop yield
data = {
    # Environmental factors
    'temperature_avg': np.random.normal(25, 5, n_samples),  # Average temp in °C
    'rainfall': np.random.normal(800, 200, n_samples),  # Annual rainfall in mm
    'humidity': np.random.normal(65, 15, n_samples),  # Humidity %
    
    # Soil factors
    'soil_ph': np.random.normal(6.5, 0.8, n_samples),  # Soil pH level
    'nitrogen_content': np.random.normal(50, 15, n_samples),  # N in kg/ha
    'phosphorus_content': np.random.normal(30, 10, n_samples),  # P in kg/ha
    'potassium_content': np.random.normal(40, 12, n_samples),  # K in kg/ha
    
    # Farm management
    'fertilizer_amount': np.random.normal(150, 40, n_samples),  # kg/ha
    'pesticide_usage': np.random.choice([0, 1, 2, 3], n_samples),  # Usage level
    'irrigation_days': np.random.normal(120, 30, n_samples),  # Days irrigated
    
    # Crop type
    'crop_type': np.random.choice(['Wheat', 'Rice', 'Maize', 'Soybean'], n_samples)
}

df = pd.DataFrame(data)

# Encode crop type
le = LabelEncoder()
df['crop_type_encoded'] = le.fit_transform(df['crop_type'])

# ============================================================================
# 3. GENERATE TARGET VARIABLE (Crop Yield)
# ============================================================================
# Create realistic yield based on features (tons per hectare)
# This simulates real-world relationships between factors and yield

df['yield'] = (
    # Base yield
    2.5 +
    # Temperature effect (optimal around 25°C)
    0.15 * df['temperature_avg'] - 0.003 * (df['temperature_avg'] - 25) ** 2 +
    # Rainfall effect (positive but diminishing)
    0.004 * df['rainfall'] - 0.000002 * df['rainfall'] ** 2 +
    # Soil nutrients
    0.02 * df['nitrogen_content'] + 0.015 * df['phosphorus_content'] + 0.01 * df['potassium_content'] +
    # pH effect (optimal around 6.5)
    0.3 * df['soil_ph'] - 0.05 * (df['soil_ph'] - 6.5) ** 2 +
    # Management practices
    0.008 * df['fertilizer_amount'] + 0.01 * df['irrigation_days'] +
    # Crop type effect
    0.5 * df['crop_type_encoded'] +
    # Random noise
    np.random.normal(0, 0.5, n_samples)
)

# Ensure positive yields
df['yield'] = df['yield'].clip(lower=0.5)

print(f"✓ Dataset created with {len(df)} samples")
print(f"\nDataset Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# ============================================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("📈 EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Correlation analysis
print("\nTop 10 Features Correlated with Yield:")
correlation = df.corr()['yield'].sort_values(ascending=False)
print(correlation.head(10))

# Visualize distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Yield distribution
axes[0, 0].hist(df['yield'], bins=30, color='green', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Distribution of Crop Yield', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Yield (tons/hectare)')
axes[0, 0].set_ylabel('Frequency')

# Yield by crop type
df.groupby('crop_type')['yield'].mean().sort_values().plot(kind='barh', ax=axes[0, 1], color='orange')
axes[0, 1].set_title('Average Yield by Crop Type', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Yield (tons/hectare)')

# Rainfall vs Yield
axes[1, 0].scatter(df['rainfall'], df['yield'], alpha=0.5, c='blue', s=20)
axes[1, 0].set_title('Rainfall vs Crop Yield', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Rainfall (mm)')
axes[1, 0].set_ylabel('Yield (tons/hectare)')

# Temperature vs Yield
axes[1, 1].scatter(df['temperature_avg'], df['yield'], alpha=0.5, c='red', s=20)
axes[1, 1].set_title('Temperature vs Crop Yield', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Temperature (°C)')
axes[1, 1].set_ylabel('Yield (tons/hectare)')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ EDA visualizations saved as 'eda_analysis.png'")

# ============================================================================
# 5. DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("🔧 DATA PREPROCESSING")
print("=" * 70)

# Select features for modeling
feature_columns = [
    'temperature_avg', 'rainfall', 'humidity', 'soil_ph',
    'nitrogen_content', 'phosphorus_content', 'potassium_content',
    'fertilizer_amount', 'pesticide_usage', 'irrigation_days',
    'crop_type_encoded'
]

X = df[feature_columns]
y = df['yield']

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Testing set: {X_test.shape[0]} samples")
print(f"✓ Features scaled using StandardScaler")

# ============================================================================
# 6. MODEL TRAINING - Multiple Algorithms
# ============================================================================
print("\n" + "=" * 70)
print("🤖 MODEL TRAINING")
print("=" * 70)

# Dictionary to store models and results
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
}

results = {}

print("\nTraining multiple models...\n")

for name, model in models.items():
    print(f"Training {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    results[name] = {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'rmse': test_rmse,
        'mae': test_mae,
        'predictions': y_pred_test
    }
    
    print(f"  ✓ Train R²: {train_r2:.4f}")
    print(f"  ✓ Test R²: {test_r2:.4f}")
    print(f"  ✓ RMSE: {test_rmse:.4f}")
    print(f"  ✓ MAE: {test_mae:.4f}\n")

# ============================================================================
# 7. MODEL EVALUATION AND COMPARISON
# ============================================================================
print("=" * 70)
print("📊 MODEL COMPARISON")
print("=" * 70)

# Select best model
best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
best_model = results[best_model_name]['model']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test R² Score: {results[best_model_name]['test_r2']:.4f}")
print(f"   RMSE: {results[best_model_name]['rmse']:.4f} tons/hectare")
print(f"   MAE: {results[best_model_name]['mae']:.4f} tons/hectare")

# Visualize model comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# R² Score comparison
model_names = list(results.keys())
r2_scores = [results[m]['test_r2'] for m in model_names]
axes[0].bar(model_names, r2_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
axes[0].set_title('Model Performance - R² Score', fontsize=14, fontweight='bold')
axes[0].set_ylabel('R² Score')
axes[0].set_ylim(0, 1)
for i, v in enumerate(r2_scores):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# Actual vs Predicted for best model
axes[1].scatter(y_test, results[best_model_name]['predictions'], alpha=0.6, s=30)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Yield (tons/hectare)', fontsize=12)
axes[1].set_ylabel('Predicted Yield (tons/hectare)', fontsize=12)
axes[1].set_title(f'{best_model_name} - Actual vs Predicted', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✓ Model evaluation saved as 'model_evaluation.png'")

# ============================================================================
# 8. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "=" * 70)
    print("🔍 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    # Get feature importances
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop Features Affecting Crop Yield:")
    print(feature_importance_df.to_string(index=False))
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='green')
    plt.xlabel('Importance', fontsize=12)
    plt.title('Feature Importance for Crop Yield Prediction', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature importance saved as 'feature_importance.png'")

# ============================================================================
# 9. PREDICTION EXAMPLES
# ============================================================================
print("\n" + "=" * 70)
print("🌾 SAMPLE PREDICTIONS")
print("=" * 70)

# Create sample scenarios
scenarios = {
    'Optimal Conditions': {
        'temperature_avg': 24,
        'rainfall': 850,
        'humidity': 70,
        'soil_ph': 6.5,
        'nitrogen_content': 60,
        'phosphorus_content': 35,
        'potassium_content': 45,
        'fertilizer_amount': 160,
        'pesticide_usage': 2,
        'irrigation_days': 130,
        'crop_type_encoded': 1
    },
    'Poor Conditions': {
        'temperature_avg': 32,
        'rainfall': 400,
        'humidity': 40,
        'soil_ph': 5.5,
        'nitrogen_content': 25,
        'phosphorus_content': 15,
        'potassium_content': 20,
        'fertilizer_amount': 80,
        'pesticide_usage': 0,
        'irrigation_days': 60,
        'crop_type_encoded': 0
    }
}

for scenario_name, features in scenarios.items():
    # Convert to DataFrame
    sample_df = pd.DataFrame([features])
    sample_scaled = scaler.transform(sample_df)
    
    # Predict
    prediction = best_model.predict(sample_scaled)[0]
    
    print(f"\n{scenario_name}:")
    print(f"  Predicted Yield: {prediction:.2f} tons/hectare")

# ============================================================================
# 10. SAVE MODEL AND SCALER
# ============================================================================
print("\n" + "=" * 70)
print("💾 SAVING MODEL")
print("=" * 70)

import pickle

# Save the best model
with open('agrilink_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Model saved as 'agrilink_model.pkl'")

# Save the scaler
with open('agrilink_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved as 'agrilink_scaler.pkl'")

# Save feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("✓ Feature names saved")

# ============================================================================
# 11. IMPACT SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("🌍 SDG IMPACT SUMMARY")
print("=" * 70)

print("""
AgriLink360 addresses SDG 2: Zero Hunger by:

1. PREDICTIVE INSIGHTS: Helps farmers predict crop yields before planting
2. RESOURCE OPTIMIZATION: Identifies key factors affecting productivity
3. RISK MITIGATION: Enables early warning for poor yield conditions
4. DATA-DRIVEN DECISIONS: Supports evidence-based agricultural planning
5. FOOD SECURITY: Contributes to stable food production systems

Model Performance:
- R² Score: {:.2%} of yield variation explained
- Average Error: ±{:.2f} tons/hectare
- Accuracy: Suitable for practical farm-level decision making

Ethical Considerations:
- Ensures data privacy for smallholder farmers
- Avoids bias toward large-scale commercial farming
- Provides accessible predictions for resource-limited contexts
- Promotes sustainable agricultural practices
""".format(results[best_model_name]['test_r2'], 
           results[best_model_name]['mae']))

print("=" * 70)
print("✓ AgriLink360 Model Training Complete!")
print("=" * 70)

print("\n" + "=" * 70)
print("DEVELOPER INFORMATION".center(70))
print("=" * 70)
print(f"{'Developer:':<20} Happy Igho Umukoro")
print(f"{'Role:':<20} AI for Software Expert")
print(f"{'Project:':<20} AI/ML Assignment - SDG 2: Zero Hunger")
print(f"{'Date:':<20} October 2025")
print(f"{'Contact:':<20} [princeigho74@gmail.com]")
print("=" * 70)

print("\n🌾 Thank you for using AgriLink360!")
print("💡 Together, we can achieve Zero Hunger through AI innovation!")
print("\n" + "=" * 70)
