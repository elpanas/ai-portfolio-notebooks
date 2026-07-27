# ❤️ ECG Heartbeat Classification & Time Series Analysis with Keras (LSTM)

Questo progetto implementa una rete neurale ricorsiva (**LSTM - Long Short-Term Memory**) per l'analisi di segnali temporali ECG e la classificazione di battiti cardiaci in **5 categorie** (Normal, Supraventricular Premature, Premature Ventricular Contraction, Fusion of Ventricular, Unclassifiable Beat).

Il focus principale del lavoro è mostrare come gestire correttamente la dimensionalità dei dati temporali 1D (trasformazione in tensori 3D), come evitare il data leakage con un partizionamento e mescolamento idoneo, e come gestire lo sbilanciamento delle classi (classi rare/aritmie).

---

## 📊 Dataset

- **Fonte:** [MIT-BIH Arrhythmia Dataset (Kaggle)](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- **Struttura:** File CSV con campionamenti temporali continui
- **Dimensioni Campione:** Ogni riga rappresenta 1 singolo battito cardiaco composto da **187 timesteps** (campionati a circa 250 Hz) + **1 colonna label/target** (la 188ª colonna)
- **Canali / Features:** 1 singolo canale (ECG a sensore singolo)
- **Classi (5 Categorie):**
  - **0 (N):** Non-ectopic beat (normal beat)
  - **1 (S):** Supraventricular ectopic beat
  - **2 (V):** Ventricular ectopic beat
  - **3 (F):** Fusion beat
  - **4 (Q):** Unknown / Unclassifiable beat

---

## 🛠️ Pre-elaborazione Dati & Gestione dei Tensori

I segnali temporali richiedono un trattamento specifico rispetto alle immagini o ai dati tabellari standard.

### 1. Separazione Features / Labels
Dal file CSV di test/train viene estratta l'ultima colonna (`iloc[:, -1]`) contenente le risposte corrette ($y$) per lasciarla da parte nella fase di valutazione, mentre le prime 187 colonne (`iloc[:, :-1]`) formano la matrice delle feature ($X$).

```python
X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values

X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values
```

### 2. Espansione a Tensore 3D `(Batch, Timesteps, Features)`
I layer ricorsivi Keras (LSTM/GRU) richiedono in ingresso un tensore tridimensionale della forma `(batch_size, timesteps, features)`.
Essendo il segnale a canale singolo, occorre aggiungere la 3ª dimensione (Features = 1) tramite `np.expand_dims`:

```python
# Da (N_campioni, 187) a (N_campioni, 187, 1)
X_train_3D = np.expand_dims(X_train, axis=2)
X_test_3D  = np.expand_dims(X_test, axis=2)
```

### 3. Shuffle Precauzionale
Poiché molti dataset clinici presentano i battiti ordinati per classe, l'estrazione casuale del validation set tramite `validation_split` senza uno shuffle preliminare rischierebbe di escludere intere classi dalla fase di validazione. Viene applicato uno **shuffle casuale** tramite Pandas/NumPy prima dell'addestramento per bilanciare correttamente il train/validation set.

---

## 🏗️ Architettura del Modello (LSTM Classifier)

Il modello è sviluppato con la sintassi **Keras 3** seguendo una struttura sequenziale pulita:

```python
model = keras.Sequential([
    # 1. Input Layer Esplicito per Sequenze Temporali 1D
    # (187 campioni temporali per 1 canale/sensore)
    layers.Input(shape=(187, 1)),

    # 2. Strato Ricorsivo LSTM per estrazione dipendenze temporali
    # 64 unità di memoria temporale
    layers.LSTM(64, return_sequences=False),

    # 3. Classificatore Finale (5 Classi Softmax)
    layers.Dense(5, activation='softmax')
])
```

### Dettagli del Layer `LSTM(64, return_sequences=False)`
- **Unità (64):** Rappresenta la dimensione dello spazio di memoria interna della cella ricorsiva.
- **`return_sequences=False`:** Fa sì che il layer restituisca solo l'ultimo stato hidden vettoriale `(batch_size, 64)` ricavato dopo aver letto tutti e 187 i punti del battito. Questo stato viene direttamente passato al layer `Dense` finale per la classificazione.
- *(Nota: Se fosse seguito da un ulteriore strato LSTM, si sarebbe dovuto impostare `return_sequences=True` per trasmettere la sequenza completa `(batch_size, 187, 64)`)*.

---

## 📈 Gestione del Class Imbalance e Valutazione

Nel dataset MIT-BIH, la stragrande maggioranza dei battiti è di tipo **N (Normale)** (~85%), mentre le aritmie (**S, V, F, Q**) sono molto rare.

### Selezione della Classe Predetta (`np.argmax`)
Il modello genera in output un vettore di 5 probabilità per ogni battito `(N_test, 5)`. Per identificare la classe predetta per ogni riga, si applica `np.argmax` lungo l'asse orizzontale (`axis=1`):

```python
# Predizioni probabilistiche della rete
predictions = model.predict(X_test_3D)

# Estrazione dell'indice di colonna con la probabilità massima per ciascun battito
predicted_classes = np.argmax(predictions, axis=1)
```

### Report di Classificazione
Poiché la semplice Accuracy globale può trarre in inganno su dataset sbilanciati, l'analisi delle performance viene integrata tramite **Precision**, **Recall** e **F1-Score** specifici per classe:

```python
from sklearn.metrics import classification_report

nomi_classi = ['N', 'S', 'V', 'F', 'Q']
print(classification_report(y_test, predicted_classes, target_names=nomi_classi))
```

---

## Come Eseguire i Notebook

1. **Notebook Inclusi:**
   - `KerasLSTMPredictor.ipynb` / `KerasLSTMClassification.ipynb`
2. **Requisiti:**
   ```bash
   pip install keras tensorflow pandas numpy scikit-learn matplotlib
   ```
3. **Esecuzione:** Esegui le celle in sequenza. Assicurarsi che i file del dataset (`mitbih_train.csv` e `mitbih_test.csv`) siano posizionati nella cartella di lavoro o scaricati via Kaggle API.