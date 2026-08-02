# Unsupervised Machine Learning: PCA & K-Means Clustering

Questo modulo contiene l'implementazione pratica di tecniche di **Unsupervised Learning** per la riduzione della dimensionalità e la clusterizzazione, applicate al dataset chimico *Wine*.

## 🎯 Obiettivo
L'obiettivo di questi notebook è dimostrare l'intera pipeline di apprendimento non supervisionato: dalla standardizzazione delle feature, alla riduzione del rumore, fino all'individuazione dei gruppi naturali presenti nei dati.

## 📂 Contenuto della Cartella
* **`ML_PCA.ipynb`**: Applicazione della Principal Component Analysis (PCA). Esplora il trade-off tra l'interpretabilità visiva (2D, ~55% della varianza) e la conservazione dell'informazione reale (riduzione multidimensionale al 90% della varianza spiegata).
* **`ML_Kmeans.ipynb`**: Implementazione dell'algoritmo K-Means. Ricerca del numero ottimale di cluster tramite l'**Elbow Method** e validazione della geometria dei gruppi tramite il **Silhouette Score**.

## 🛠️ Tecnologie Utilizzate
* **Librerie ML/Data Science:** `scikit-learn`, `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`

## 📊 Highlight Tecnici
* **Preprocessing Rigoroso:** Utilizzo di `StandardScaler` per normalizzare le distribuzioni ed evitare distorsioni nel calcolo delle distanze euclidee.
* **Gestione della Dimensionalità:** Dimostrazione pratica di come il Silhouette Score vari in base alle dimensioni dello spazio latente, evidenziando il concetto di rumore vs. segnale utile.

## 🚀 Come eseguire i notebook
1. Installare le dipendenze necessarie tramite:
   `pip install scikit-learn pandas matplotlib seaborn`
2. Apri i notebook nel tuo ambiente preferito (VS Code, Jupyter Notebook, Google Colab):
   `jupyter notebook`