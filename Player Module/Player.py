# Setup
!pip install pandas numpy seaborn matplotlib scikit-learn tensorflow joblib

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# Load Data
from google.colab import files
uploaded = files.upload()
cols = ['pace','shooting','passing','dribbling','defending','physic','overall','club_name']
df_male = pd.read_csv('Male Players.csv', usecols=cols, low_memory=False)
df_female = pd.read_csv('Female Players.csv', usecols=cols, low_memory=False)

# Add Gender & Merge
df_male['gender'] = 0
df_female['gender'] = 1
df = pd.concat([df_male, df_female], ignore_index=True)

# Cleaning
df.rename(columns={'physic': 'physical'}, inplace=True)
df = df.dropna()
df.reset_index(drop=True, inplace=True)
print(df.shape)

# EDA
# Numeric
numeric_df = df.select_dtypes(include=['number'])

# Distribution
plt.figure()
sns.histplot(df['overall'], bins=30)
plt.title("Overall Rating Distribution")
plt.show()

# Correlation
plt.figure()
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Feature vs Target
features = ['pace','shooting','passing','dribbling','defending','physical']
for col in features:
    plt.figure()
    sns.regplot(x=df[col], y=df['overall'])
    plt.title(f"{col} vs Overall")
    plt.show()

# Train-Test Split
X = df[['pace','shooting','passing','dribbling','defending','physical','gender']]
y = df['overall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
model = Sequential([
    Input(shape=(7,)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Training
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    verbose=1
)

# Evaluation
pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print("RMSE:", rmse)
print("R2:", r2)

# Save Models
model.save("player_model.keras")
joblib.dump(scaler, "scaler.pkl")
df.to_csv("clean_players.csv", index=False)