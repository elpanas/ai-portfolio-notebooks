# 🖼️ Intel Image Classification with Keras (CNN)

Questo progetto implementa una rete neurale convoluzionale (**CNN**) per la classificazione di immagini di paesaggi naturali in **6 categorie** (buildings, forest, glacier, mountain, sea, street).

Il focus principale del notebook è dimostrare l'efficacia delle architetture convoluzionali e analizzare l'impatto dell'uso di **`GlobalAveragePooling2D`** rispetto al classico **`Flatten`** in termini di parametri e rischio di overfitting.

---

## 📊 Dataset

- **Fonte:** [Intel Image Classification (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Download tramite:** `kagglehub.dataset_download("puneet6060/intel-image-classification")`
- **Dimensione Immagini:** `150x150` pixel (RGB)
- **Classi:** 6 paesaggi (Buildings, Forest, Glacier, Mountain, Sea, Street)

---

## 🚀 Data Augmentation
In caso di dataset piccoli o semplicemente per generalizzare di più, è possibile usare una tecnica come la **Data Augmentation** per aumentare le variazioni dei campioni e migliorare l'efficacia del modello.

```python
# DATA AUGMENTATION
data_augmentation = keras.Sequential(
    [
        # Capovolge l'immagine in orizzontale casualmente (es. un palazzo a destra diventa a sinistra)
        layers.RandomFlip("horizontal"),
        
        # Ruota casualmente l'immagine fino a un massimo del 10% (circa 36 gradi)
        layers.RandomRotation(0.1),
        
        # Effettua uno zoom casuale (in avanti o indietro) fino al 10%
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation"
)
```
poi nel modello aggiungo i livelli `data_augmentation` dopo il livello di input.

```python
model = keras.Sequential([                          
    # 1. Input Layer Esplicito (sintassi pulita Keras 3)
    layers.Input(shape=(150, 150, 3)),

    # 1.1. LAYER DI AUGMENTATION (Attivo solo durante model.fit)
    data_augmentation,

    # ... gli altri livelli come prima
```
Aumentando le variazioni dei campioni, è necessario incrementare anche le epoche per dare al modello la possibilità di vedere la maggior parte di queste variazioni.

## 🏗️ Architettura del Modello

Il modello è sviluppato con la sintassi **Keras 3** seguendo una struttura sequenziale pulita:

```python
model = keras.Sequential([
    # 1. Input Layer
    layers.Input(shape=(150, 150, 3)),

    # 2. Normalizzazione Pixel (da 0..255 a 0..1)
    layers.Rescaling(1./255),

    # 3. Blocco Conv 1: Dettagli di base (32 filtri)
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 4. Blocco Conv 2: Forme e texture (64 filtri)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 5. Blocco Conv 3: Oggetti e paesaggi complessi (128 filtri)
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 6. Classificatore Finale
    layers.GlobalAveragePooling2D(), # Riduzione a un unico vettore 2D -> 1D
    layers.Dense(128, activation='relu'), # Crea combinazioni complesse prima di classificare

    # 6 Classificazione con 1 neurone per ogni classe
    layers.Dense(num_classes, activation='softmax')
])
```
### La strategia a blocchi
Le immagini hanno una dimensione medio-grande, quindi si è optato per 3 blocchi composti dai seguenti layer con numero di filtri incrementale:
```python
layers.Conv2D(<32-64-128>, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
```

### Il doppio livello Dense() nel finale
Il penultimo livello Dense() è una buona pratica per aumentare le combinazioni e rendere quindi il modello più generalizzato. Rallenta, quindi è preferibile con dataset piccoli/leggeri. Può introdurre overfitting; nel caso si aggiunge un livello Dropout.




## 🔬 Confronto Tecnico: GlobalAveragePooling2D vs Flatten

Nel notebook viene effettuato un confronto diretto sostituendo il livello di pooling globale con un livello di `Flatten()`.

| Metrica / Feature | Con `GlobalAveragePooling2D()` | Con `Flatten()` |
| :--- | :--- | :--- |
| **Output Shape (Pre-Dense)** | `(None, 128)` | `(None, 36992)` |
| **Parametri Totali** | **110,534** (~431 KB) | **4,829,126** (~18.42 MB) |
| **Parametri Dense Layer** | 16,512 | 4,735,104 |
| **Rischio Overfitting** | **Basso** (Forte regolarizzazione) | **Alto** (Troppi pesi da addestrare) |
| **Velocità Addestramento** | **Molto alta** | Più lenta |

### 💡 Considerazioni Tecniche
- **GlobalAveragePooling2D:** Riduce le dimensioni spaziali (17x17x128) calcolando la media di ciascun feature map. Questo abbatte drasticamente i parametri del livello `Dense` da **4.7M a sole 16k**, fungendo da eccellente regolarizzatore strutturale.
- **Flatten:** Appiattisce tutti i valori (17x17x128 = 36.992), portando a un'esplosione dei parametri collegati al livello denso e rendendo la rete più incline all'overfitting sui dati di training se non bilanciata con Dropout/L2.

---

## Ottimizzazioni

1. ### Data Augmentation
Con pochi dati (es.: 500 immagini invece di 3000) è necessario creare variazioni artificiali in modo da fare in modo che il modello sia capace di generalizzare nonostante le poche informazioni in input.

Keras fornisce già layer che effettuano queste variazioni automaticamente

```python
# DATA AUGMENTATION
data_augmentation = keras.Sequential(
    [
        # Capovolge l'immagine in orizzontale casualmente (es. un palazzo a destra diventa a sinistra)
        layers.RandomFlip("horizontal"),
        
        # Ruota casualmente l'immagine fino a un massimo del 10% (circa 36 gradi)
        layers.RandomRotation(0.1),
        
        # Effettua uno zoom casuale (in avanti o indietro) fino al 10%
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation"
)
```
E poi vanno inseriti nel modello dopo il livello di input e verranno automaticamente saltati da Keras in fase di training

```python
model = keras.Sequential([                          
    # 1. Input Layer Esplicito (sintassi pulita Keras 3)
    layers.Input(shape=(150, 150, 3)),

    # 1.1. LAYER DI AUGMENTATION <----
    data_augmentation,

    # ...il resto dei layer uguali a prima 
```
Così facendo ad ogni epoca, il modello analizzerà 3 versioni diverse della stessa immagine. Le variazioni sono casuali, quindi non vede TUTTE le variazioni ma solo quelle scelte a caso dal livello. 
Dato che aumentano le variazioni, serviranno più epoche per sfruttare bene la augmentation, ma alla fine si avrà un modello più generalizzato e robusto.

2. ### Transfer Learning
L’altra tecnica in caso di dataset piccoli o semplicemente per generalizzare di più è utilizzare reti neurali già pre-addestrate sullo stesso tipo di dati eliminando testa e coda. La testa sono i dati in input (uso i miei), la coda è il layer Dense: ovviamente userò il mio con il mio numero di classi.

```python
# 1. Carichiamo la base di MobileNetV2 di Google
base_model = keras.applications.MobileNetV2(
    input_shape=(150, 150, 3),
    include_top=False,     # Escludiamo la testa da 1000 classi
    weights='imagenet'     # Usiamo i pesi pre-addestrati da Google
)

# 2. CONGELIAMO la base (non deve aggiornare i suoi pesi)
base_model.trainable = False
```

Tolgo i livelli di elaborazione del mio modello originale e li sostituisco con quelli del modello pre-addestrato, lasciando appunto testa e coda del mio invariat.

```python
model = keras.Sequential([
    layers.Input(shape=(150, 150, 3)),
    
    # Manteniamo il Data Augmentation
    data_augmentation,
    
    # NOTA: MobileNetV2 si aspetta i pixel tra -1 e 1 anziché tra 0 e 1.
    # Keras ha una funzione apposita per la pre-elaborazione di MobileNetV2!
    layers.Lambda(keras.applications.mobilenet_v2.preprocess_input),
    
    # Inseriamo la base pre-addestrata
    base_model,
    
    # La nuova testa personalizzata
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2), # Opzionale: Dropout per evitare Overfitting
    layers.Dense(num_classes, activation='softmax')
])
```

3. ### Fine Tuning
Anziché congelare tutti i livelli del modello, posso dire a keras di sbloccarne 20 o 30 che si adatteranno alle mie 6 classi. In questo modo il modello non sarà più generale su ogni tipo di immagini, ma un po’ specifico sul tipo di immagini che voglio classificare.

1. Sblocco il modello pre-trained
```python
base_model.trainable = True
```

2. Congelo tutti i livelli eccetto gli ultimi 24
```python
# (MobileNetV2 ha 154 layer in totale)
fine_tune_at = 130
# mi fermo a 130
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
```

3. Ricompilo ma riducendo il learning rate perché…

```python
model_pre_trained.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), # 0.00001
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']

```
4. …perché fa il training ma a partire dall’epoca successiva quindi siamo già vicini all’errore minimo e un LR troppo alto farebbe saltare il bersaglio

```python
history_fine = model_pre_trained.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15, # gli dico quante epoche in totale
    initial_epoch=8 # Riparte da dove era rimasto
)
```

## 🚀 Come Eseguire il Notebook

1. **Apri il notebook:** [`KerasCNN.ipynb`](./KerasCNN.ipynb)
2. **Requisiti:**
   ```bash
   pip install keras tensorflow kagglehub matplotlib numpy
   ```
3. **Esecuzione:** Esegui le celle in sequenza. Il dataset verrà scaricato automaticamente tramite `kagglehub`