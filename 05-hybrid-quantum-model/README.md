# 🫀 Hybrid Quantum-Classical Heart Disease Classifier (HQNN)

Un'architettura di Machine Learning Quantistico (QML) ibrida per la classificazione binaria del rischio di malattie cardiache. Il modello combina la capacità di estrazione delle feature delle reti neurali classiche (PyTorch Lightning) con le proprietà di entanglement e sovrapposizione di un circuito quantistico variazionale (TorchQuantum).

---

## 📐 Architettura del Modello

Il flusso di lavoro integra elementi classici e quantistici in una pipeline end-to-end:

1. **Preprocessing & Encoding**:
   - Pulizia delle feature cliniche e codifica One-Hot delle variabili categoriche (15 feature finali).
   - Normalizzazione standard via `StandardScaler`.
   - Bilanciamento dinamico della loss tramite pesi di classe ponderati (`compute_class_weight`).

2. **Feature Reduction Classica**:
   - Layer Lineare Denso (15 -> 16 neuroni) con attivazione ReLU e Dropout (0.2).
   - Compressione dimensionale (16 -> 4 valori) proiettata nell'intervallo [-π, π] per mappare gli angoli di rotazione dei qubit.

3. **Quantum Layer (Variational Quantum Circuit)**:
   - **Qubit Count**: 4 qubit (1 per canale ridotto).
   - **State Preparation**: Rotazioni Ry(θ) basate sui parametri provenienti dallo strato classico.
   - **Variational Circuit**: Layer casuale parametrizzato (`tq.RandomLayer`) con porte quantistiche entangled.
   - **Measurement**: Misurazione Pauli-Z su tutti i qubit per estrarre uno stato aspettato reale a 4 dimensioni.

4. **Concatenazione & Classificazione**:
   - Fusione dei vettori: 16 feature classiche + 4 misurazioni quantistiche = 20 dimensioni aggregate.
   - Layer finale lineare (20 -> 2 classi: *Basso Rischio* vs *Alto Rischio*).

---

## 🛠️ Tech Stack & Dipendenze

- **Python 3.12**
- **PyTorch & PyTorch Lightning**: Gestione dell'architettura e del ciclo di training.
- **TorchQuantum & Qiskit**: Simulazione del dispositivo quantistico e dei circuiti variazionali.
- **Scikit-Learn & Pandas**: Preprocessing, scaling e metriche.

---

## 🚀 Setup e Installazione

Per garantire la compatibilità binaria tra i moduli C di PyTorch/Qiskit e NumPy, utilizzare le seguenti versioni:

```bash
pip install "numpy<2.0.0" "pandas<2.2.0" pytorch-lightning "qiskit==0.45.3" "qiskit-aer==0.13.3" "qiskit-ibm-runtime==0.19.1" torchquantum
```

---

## 💻 Esempio di Utilizzo e Inferenza

Dopo l'addestramento, il modello e lo scaler consentono la diagnosi preventiva su nuovi profili paziente:

```python
import torch
import joblib
from model import QMLModel  # Import della classe del modello

# 1. Profilo paziente da analizzare
sample_patient = {
    "Age": 55, "RestingBP": 140, "Cholesterol": 289, "FastingBS": 1,
    "MaxHR": 122, "Oldpeak": 1.5, "Sex_M": 1, "ChestPainType_ATA": 0,
    "ChestPainType_NAP": 0, "ChestPainType_TA": 0, "RestingECG_Normal": 0,
    "RestingECG_ST": 1, "ExerciseAngina_Y": 1, "ST_Slope_Flat": 1, "ST_Slope_Up": 0
}

# 2. Caricamento dello scaler e preprocessing
scaler = joblib.load("scaler.joblib")
patient_scaled = scaler.transform([list(sample_patient.values())])
patient_tensor = torch.tensor(patient_scaled, dtype=torch.float32)

# 3. Inferenza del modello ibrido
model.eval()
with torch.no_grad():
    logits = model(patient_tensor)
    probs = torch.softmax(logits, dim=1)

risk_percent = round(probs[0][1].item() * 100, 2)
label = "Alto Rischio" if probs.argmax(dim=1) == 1 else "Basso Rischio"

print(f"Diagnosi: {label} ({risk_percent}% rischio)")
```

---

## 📊 Risultati

- **Validation Accuracy**: ~91.3%
- **Loss di Addestramento**: ~0.31
- **Epochs**: 15 (Ottimizzatore AdamW + LR Scheduler *ReduceLROnPlateau*)