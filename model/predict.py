import numpy as np
import joblib
import time
from tensorflow.keras.models import load_model

# ------------------------------
# Load Model and Scaler
# ------------------------------
model = load_model("bilstm_dengue_model.keras")
scaler = joblib.load("scaler.pkl")

print("Model and Scaler loaded successfully.\n")

# ------------------------------
# Take User Input
# Order MUST match training features
# ------------------------------

print("Enter CBC Values:")

Gender = int(input("Gender (Male=1, Female=0): "))
Age = float(input("Age: "))
Haemoglobin = float(input("Haemoglobin: "))
ESR = float(input("ESR: "))
WBC = float(input("WBC: "))
Neutrophil = float(input("Neutrophil: "))
Lymphocyte = float(input("Lymphocyte: "))
Monocyte = float(input("Monocyte: "))
Eosinophil = float(input("Eosinophil: "))
Basophil = float(input("Basophil: "))
RBC = float(input("RBC: "))
Platelets = float(input("Platelets: "))

# Create array
input_data = np.array([[Gender, Age, Haemoglobin, ESR, WBC,
                        Neutrophil, Lymphocyte, Monocyte,
                        Eosinophil, Basophil, RBC, Platelets]])

# ------------------------------
# Start Timing
# ------------------------------
start_time = time.time()

# Scale
input_scaled = scaler.transform(input_data)

# Reshape for RNN
input_rnn = input_scaled.reshape((1, input_scaled.shape[1], 1))

# Predict
probability = model.predict(input_rnn)[0][0]

# Stop Timing
end_time = time.time()
decision_time = end_time - start_time

# ------------------------------
# Interpret Result
# ------------------------------

predicted_class = 1 if probability > 0.5 else 0
class_label = "Positive (Dengue)" if predicted_class == 1 else "Negative (No Dengue)"

confidence = probability if predicted_class == 1 else 1 - probability

# ------------------------------
# Output
# ------------------------------

print("\n----- Prediction Result -----")
print(f"Predicted Class: {class_label}")
print(f"Probability: {probability:.4f}")
print(f"Confidence: {confidence*100:.2f}%")
print(f"Decision Time: {decision_time:.6f} seconds")