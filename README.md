# 🤖 AI & Deep Learning Project Collection

Raccolta di notebook, esperimenti e modelli di Machine Learning e Deep Learning implementati a scopo didattico e dimostrativo.

---

## 📂 Struttura del Repository

La repository è organizzata in moduli distinti per tipologia di architettura e ambito di applicazione:

```text
.
├── 01-computer-vision/
│   └── KerasCNN.ipynb          # Classificazione immagini con CNN
├── 02-time-series/
│   └── TODO.ipynb      # Generazione/Analisi sequenze con LSTM
├── 03-classical-ml/
│   └── TODO  # Algoritmi di regressione e analisi
└── README.md
```

---

## 📊 Indice dei Modelli ed Esperimenti

| Categoria            | Architettura / Algoritmo | Notebook                                                             | Dataset                 | Descrizione & Risultati                                                           |
| :------------------- | :----------------------- | :------------------------------------------------------------------- | :---------------------- | :-------------------------------------------------------------------------------- |
| **Computer Vision**  | CNN (Keras / TensorFlow) | [`KerasCNN.ipynb`](./01-computer-vision/KerasCNN.ipynb)              | Custom / Natural Images | Classificazione multilabelle. Raggiunta un'accuratezza del **~81%** sul test set. |
| **Sequenze / NLP**   | LSTM / Recurrent NN      | [`TODO.ipynb`](./02-time-series/TODO.ipynb)         | Time Series / Text      | Modellazione di sequenze temporali e gestione della memoria a lungo termine.      |
| **Machine Learning** | Regressione Lineare      | [`TODO.ipynb`](./03-classical-ml/LinearRegression.ipynb) | Synthetic / Tabular     | Implementazione e analisi dei residui, ottimizzazione tramite gradient descent.   |

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
