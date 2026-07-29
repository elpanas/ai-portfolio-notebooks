# 🫀 Classificazione di Segnali ECG tramite GRU & SMOTE

Questo repository contiene una pipeline di Deep Learning basata su un'architettura **Gated Recurrent Unit (GRU)** per classificare 5 diverse tipologie di battiti cardiaci da dati elettrocardiografici (ECG) (dataset MIT-BIH / PTB-XL).

Il progetto affronta il forte sbilanciamento delle classi applicando **SMOTE** rigorosamente dopo lo split dei dati, garantendo un processo di validazione pulito, affidabile e del tutto privo di *data leakage*.

---

## 📌 Punti Chiave del Progetto

* **Strategia di Validazione Pulita:** `train_test_split` (80/20) stratificato applicato prima di qualsiasi tecnica di data augmentation.
* **Addestramento Bilanciato:** `SMOTE` applicato **esclusivamente** sul set di training, generando istanze sintetiche per le classi minoritarie e mantenendo i set di validazione e test 100% reali e autentici.
* **Rappresentazione Sequenziale:** I segnali 1D sono stati trasformati in tensori 3D `(Campioni, Istanti Temporali, Feature)` per alimentare la rete **GRU**, ottimizzata per catturare le dipendenze temporali dei tracciati ECG.
* **Early Stopping Callback:** Configurato con `restore_best_weights=True` per prevenire l'overfitting e ripristinare i pesi ottimali del modello al termine dell'addestramento.

---

## 📊 Prestazioni e Risultati

Il modello GRU ha dimostrato un'elevata capacità di generalizzazione, eccellendo in particolare nel riconoscimento di patologie cardiache rare e clinicamente critiche (come i battiti sopraventricolari e quelli di fusione).

### 📈 Classification Report (Validation Set)

```text
              precision    recall  f1-score   support

           N       0.99      0.97      0.98     18118
           S       0.58      0.81      0.67       556
           V       0.93      0.94      0.94      1448
           F       0.45      0.91      0.60       162
           Q       0.98      0.98      0.98      1608

    accuracy                           0.96     21892
   macro avg       0.79      0.92      0.83     21892
weighted avg       0.97      0.96      0.97     21892
```

### 🔍 Insight sulle Metriche
* **Accuracy Globale:** **96%** su dati di validazione reali mai visti durante il training.
* **Altissima Sensibilità (Recall) sulle Classi Rare:** 
  * **Classe F (Battito di Fusione):** **91% Recall**
  * **Classe S (Battito Sopraventricolare Prematuro):** **81% Recall**
* **Stabilità su Classi V e Q:** F1-score eccezionali pari a **0.94** e **0.98**.

---

## 🛠️ Architettura della Pipeline

1. **Split dei dati**
```python
# 20% in validation
X_train_splitted, X_val_splitted, y_train_splitted, y_val_splitted = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)
```

2. **SMOTE Resampling (Solo su Training Set)**
```python
# Genera i nuovi campioni sintetici e relative etichette
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_splitted, y_train_splitted
)
```

3. **Shuffle Post-SMOTE e Reshape 3D**
```python
# Mescola i dati
X_train_resampled, y_train_resampled = shuffle(
    X_train_resampled, y_train_resampled, random_state=42
)
# Reshape 3D per il modello GRU
X_train_resampled_3D = np.expand_dims(X_train_resampled, axis=-1)
X_val_splitted_3D = np.expand_dims(X_val_splitted, axis=-1)
```

4. **Addestramento del Modello**
* **Ottimizzatore:** Adam
* **Funzione di Loss:** `sparse_categorical_crossentropy`
* **Callback:** `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`

---

## 🚀 Come Eseguire il Codice

1. Clona il repository:
```bash
git clone [https://github.com/el_panas/ai-portfolio-notebooks.git](https://github.com/el_panas/ai-portfolio-notebooks.git)
cd ai-portfolio-notebooks
```

2. Installa le dipendenze:
```bash
pip install tensorflow numpy pandas scikit-learn imbalanced-learn matplotlib seaborn plotly kagglehub
```

3. Esegui il notebook:
Apri `KerasGRUClassification.ipynb` su Google Colab o Jupyter Lab ed esegui tutte le celle.