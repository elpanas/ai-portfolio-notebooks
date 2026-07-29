# Classificazione di dati tabellari con XGBoost

Questo progetto dimostra l'applicazione di algoritmi di Machine Learning su **dati tabellari non sequenziali**, concentrandosi sulla predizione dell'abbandono dei clienti (**Churn**) per un'azienda di Telecomunicazioni (Telco).

Mentre il Deep Learning (es. CNN, RNN) eccelle su immagini e sequenze, per i dati tabellari strutturati i modelli ad albero come **XGBoost** offrono accuratezza superiore, velocità di addestramento istantanea e completa interpretabilità delle decisioni aziendali.

---

## 📌 Obiettivi del Progetto

1. **Data Preprocessing Completo**: Pulizia, gestione dei valori mancanti, mapping e One-Hot Encoding efficiente.
2. **Preventing Multicollinearity**: Utilizzo di `drop_first=True` per eliminare colonne ridondanti senza perdere informazione matematica.
3. **Training di XGBoost**: Configurazione e addestramento del modello `XGBClassifier`.
4. **Valutazione con Probabilità Continuative**: Differenza tra classificazione secca (Accuracy) e Ranking di rischio aziendale (ROC-AUC).
5. **Feature Importance**: Identificazione delle leve commerciali principali che causano la disdetta dei clienti.

---

## 🛠️ Data Preprocessing & Encoding

Il dataset richiede una fase accurata di preparazione prima di essere passato al modello:

* **Rimozione ID**: Eliminazione di colonne non informative come `customerID`.
* **Binary Mapping**: Conversione diretta di colonne Target come `Churn` in valori interi (`No` $\rightarrow$ `0`, `Yes` $\rightarrow$ `1`).
* **Coercizione Tipi e Imputazione**: Conversione dei campi di testo con spazi vuoti in valori numerici (`pd.to_numeric(..., errors='coerce')`) e imputazione dei `NaN` con zeri (`.fillna(0)`).
* **One-Hot Encoding Ottimizzato**: Conversione delle variabili categoriali multivalore tramite `pd.get_dummies()`.

```python
# Applicazione del One-Hot Encoding sulle variabili categoriche
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

df_encoded = pd.get_dummies(
    df_clean,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)
```
### 💡 Perché si usa `drop_first=True`?
Quando una variabile categoriale ha $N$ categorie (es. *InternetService* con "DSL", "Fiber optic", "No"), la trasformazione standard creerebbe $N$ nuove colonne binarie. 

Impostando `drop_first=True`, la prima colonna in ordine alfabetico viene eliminata, ottenendo **$N - 1$ colonne**. 

**Non si perde alcuna informazione**: se per un cliente le $N - 1$ colonne rimanenti valgono tutte `0`, il modello deduce con certezza matematica che il cliente appartiene alla categoria eliminata. Questa tecnica evita la **multicollinearità perfetta** (Trappola della Dummy Variable) e riduce la dimensione del dataset.

---

## 🚀 Model Training (XGBoost)

Il dataset viene suddiviso in set di addestramento e di test tramite `train_test_split` con stratificazione per preservare la distribuzione delle classi.

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 1. Separazione Feature (X) e Target (y)
X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

# 2. Split Stratificato
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Configurazione e Fit del Modello
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)
```

## 📊 Valutazione & Probabilità Continuativa

I modelli di classificazione restituiscono sia predizioni secche che punteggi di probabilità:

```python
from sklearn.metrics import accuracy_score, roc_auc_score

# Predizioni secche (0 o 1)
y_pred_xgb = xgb_model.predict(X_test)

# Probabilità continuativa (da 0.0 a 1.0) per la classe 1
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Calcolo Metriche
acc_xgb = accuracy_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_proba_xgb)
```

### 📈 Accuracy vs. ROC-AUC (Probabilità Continuativa)

* **Accuracy**: Calcola la percentuale di risposte esatte applicando una soglia rigida del 50% (se probabilità $< 0.50 \rightarrow 0$, se $\ge 0.50 \rightarrow 1$). Perde il livello di confidenza reale.
* **ROC-AUC (Probabilità Continuativa)**: Misura la capacità del modello di ordinare i clienti in base al rischio reale, valutando la bontà della stima da 0% a 100%.

> **Valore di Business**:
> La **probabilità continuativa** trasforma un problema di classificazione secca (Sì/No) in una **classifica ordinata di rischio (Ranking)**. Permette all'azienda di intervenire in modo proattivo, allocando il budget di marketing prioritariamente sui clienti a più alto rischio di disdetta (es. con probabilità $> 80\%$).

---

## 🔍 Feature Importance

XGBoost permette di identificare quali variabili influiscono maggiormente sulla decisione di disdetta del cliente:

```python
import pandas as pd

# Estrazione delle 10 feature più influenti
feature_importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
top_10_features = feature_importances.nlargest(10)

print(top_10_features)
```

### 📌 Conclusione
L'analisi delle feature guida le decisioni strategiche (es. rimodulazione dei contratti *month-to-month* o revisione dei prezzi della Fibra Ottica) per massimizzare la retention dei clienti.

---

