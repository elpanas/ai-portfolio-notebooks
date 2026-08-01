# 🤖 AI & Deep Learning Project Collection

Raccolta di notebook, esperimenti e modelli di Machine Learning e Deep Learning implementati a scopo didattico e dimostrativo.

---

## 📂 Struttura del Repository

La repository è organizzata in moduli distinti per tipologia di architettura e ambito di applicazione:

```text
.
├── 01-computer-vision/
│   └── KerasCNN.ipynb          # Classificazione immagini con CNN
│   └── KerasCNN Enhanced.ipynb          # Classificazione immagini con CNN e tecniche avanzate
├── 02-time-series/
│   └── LSTM/KerasLSTMPredictor.ipynb      # Previsioni meteo su sequenze con LSTM
│   └── LSTM/KerasLSTMClassification.ipynb  # Classificazione ECG con LSTM
│   └── GRU/KerasGRUClassification.ipynb  # Classificazione ECG con GRU
├── 03-machine-learning/
│   └── sueprvised_tabular/XGBoost.ipynb              # Predizione abbandono clienti con XGBoost
│   └── unsupervised/ML_Kmeans.ipynb          # Clusterizzazione clienti con K-Means
│   └── unsupervised/ML_PCA.ipynb              # Dimensionalità ridotta con PCA
├── 04-nlp-llm-finetuning/
│   └── NLP_LLM_LoRA.ipynb          # LLM fine-tuning con LoRA
└── README.md
```

---

## 📊 Indice dei Modelli ed Esperimenti

| Categoria | Architettura / Algoritmo | Notebook | Dataset | Descrizione & Risultati |
| :--- | :--- | :--- | :--- | :--- |
| **Computer Vision** | CNN (Keras / TensorFlow) | [`KerasCNN.ipynb`](./01-computer-vision/KerasCNN.ipynb) | Custom / Natural Images | Classificazione multilabelle. Raggiunta un'accuratezza del **~81%** sul test set. |
| **Sequenze Temporali** | LSTM | [`KerasLSTMPredictor.ipynb`](./02-time-series-sequences/LSTM/KerasLSTMPredictor.ipynb) | Daily Delhi Climate | Previsioni meteo e analisi di serie temporali. |
| **Sequenze Temporali** | LSTM | [`KerasLSTMClassification.ipynb`](./02-time-series-sequences/LSTM/KerasLSTMClassification.ipynb) | ECG Signals | Classificazione di segnali elettrocardiografici (ECG). |
| **Sequenze Temporali** | GRU | [`KerasGRUClassification.ipynb`](./02-time-series-sequences/GRU/KerasGRUClassification.ipynb) | ECG Signals | Classificazione di segnali ECG ottimizzata con unità GRU. |
| **Machine Learning** | XGBoost (Gradient Boosting) | [`XGBoost.ipynb`](./03-machine-learning/supervised_tabular/XGBoost.ipynb) | Telco Churn | Predizione dell'abbandono clienti basata sul ranking delle probabilità continuative (ROC-AUC). |
| **Machine Learning** | K-Means | [`ML_K-Means.ipynb`](./03-machine-learning/unsupervised/ML_K-Means.ipynb) | Mall Customer Segmentation Data | Segmentazione dei clienti con K-Means. |
| **Machine Learning** | PCA | [`ML_PCA.ipynb`](./03-machine-learning/unsupervised/ML_PCA.ipynb) | Wine dataset | Dimensionalità ridotta con PCA. |
| **Natural Language Processing** | NLP / LLM (LoRA) | [`NLP_LLM_QLoRA.ipynb`](./04-nlp-llm-finetuning/NLP_LLM_QLoRA.ipynb) | Custom / Ticket JSON | Fine-Tuning efficiente a 4-bit per l'estrazione dati da ticket in formato JSON strutturato. |

---

## 🛠️ Tecnologie e Framework Utilizzati

- **Linguaggio:** Python 3.x
- **Deep Learning Frameworks:** Keras, TensorFlow, PyTorch
- **Data Manipulation & Viz:** Pandas, NumPy, Matplotlib
- **Ambiente di Sviluppo:** VS Code, Google Colab / Jupyter Notebooks

---

## 🚀 Come Eseguire i Notebook in Locale

1. **Clona la repository:**
   ```bash
   git clone [https://github.com/el_panas/ai-portfolio-notebooks.git](https://github.com/el_panas/ai-portfolio-notebooks.git)
   cd ai-portfolio-notebooks
   ```

2. **Crea ed attiva un ambiente virtuale (consigliato):**
   ```bash
   python -m venv venv
    # Su Linux/macOS:
    source venv/bin/activate
    # Su Windows:
    .\venv\Scripts\activate
   ```

3. **Installa le dipendenze principali:**
   ```bash
   pip install tensorflow keras pandas numpy matplotlib
   ```

4. **Apri VSCode:**
   ```bash
   code .
   ```
   
5. **Esegui i Notebook:**
   ```bash
   jupyter notebook
   ```

## 📧 Contatti e Link

- **GitHub Repository:** [https://github.com/el_panas/ai-portfolio-notebooks](https://github.com/el_panas/ai-portfolio-notebooks)
- **LinkedIn:** [https://www.linkedin.com/in/luca-panariello-panas/](https://www.linkedin.com/in/luca-panariello-panas/)
