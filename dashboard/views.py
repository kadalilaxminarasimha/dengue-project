from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Prediction
from django.db.models import Count

import numpy as np
import joblib
import time
from tensorflow.keras.models import load_model

model = load_model("model/bilstm_dengue_model.keras")
scaler = joblib.load("model/scaler.pkl")

@login_required
def dashboard_view(request):
    return render(request, 'dashboard/dashboard.html')

@login_required
def predict_view(request):
    result = None

    if request.method == "POST":
        try:
            # ------------------------------
            # Get input from form
            # ------------------------------
            Gender = int(request.POST.get("gender"))
            Age = float(request.POST.get("age"))
            Haemoglobin = float(request.POST.get("hb"))
            ESR = float(request.POST.get("esr"))
            WBC = float(request.POST.get("wbc"))
            Neutrophil = float(request.POST.get("neutrophil"))
            Lymphocyte = float(request.POST.get("lymphocyte"))
            Monocyte = float(request.POST.get("monocyte"))
            Eosinophil = float(request.POST.get("eosinophil"))
            Basophil = float(request.POST.get("basophil"))
            RBC = float(request.POST.get("rbc"))
            Platelets = float(request.POST.get("platelets"))

            # ------------------------------
            # Prepare data
            # ------------------------------
            input_data = np.array([[Gender, Age, Haemoglobin, ESR, WBC,
                                    Neutrophil, Lymphocyte, Monocyte,
                                    Eosinophil, Basophil, RBC, Platelets]])

            start_time = time.time()

            input_scaled = scaler.transform(input_data)
            input_rnn = input_scaled.reshape((1, input_scaled.shape[1], 1))

            probability = model.predict(input_rnn)[0][0]

            end_time = time.time()
            decision_time = end_time - start_time

            # ------------------------------
            # Interpret result
            # ------------------------------
            predicted_class = 1 if probability > 0.5 else 0
            class_label = "Positive (Dengue)" if predicted_class == 1 else "Negative (No Dengue)"
            confidence = probability if predicted_class == 1 else 1 - probability

            # ------------------------------
            # Save to DB
            # ------------------------------
            Prediction.objects.create(
                user=request.user,
                input_data={
                    "Gender": Gender,
                    "Age": Age,
                    "Haemoglobin": Haemoglobin,
                    "ESR": ESR,
                    "WBC": WBC,
                    "Neutrophil": Neutrophil,
                    "Lymphocyte": Lymphocyte,
                    "Monocyte": Monocyte,
                    "Eosinophil": Eosinophil,
                    "Basophil": Basophil,
                    "RBC": RBC,
                    "Platelets": Platelets,
                },
                predicted_class=class_label,
                confidence=confidence * 100
            )

            result = f"{class_label} | Confidence: {confidence*100:.2f}% | Time: {decision_time:.4f}s"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render(request, 'dashboard/predict.html', {"result": result})

@login_required
def history_view(request):
    qs = Prediction.objects.filter(user=request.user)

    # Count per class
    class_counts = qs.values('predicted_class').annotate(count=Count('id'))

    labels = [item['predicted_class'] for item in class_counts]
    data = [item['count'] for item in class_counts]

    return render(request, 'dashboard/history.html', {'labels': labels,'data': data,})

@login_required
def profile_page(request):
    profile = request.user.profile
    return render(request, 'dashboard/profile.html', {'profile': profile})

@login_required
def my_predictions(request):
    predictions = Prediction.objects.filter(user=request.user)
    return render(request, 'dashboard/my_predictions.html', {'predictions': predictions})
