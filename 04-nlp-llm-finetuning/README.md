# 🚀 Fine-Tuning LLM con LoRA / QLoRA per Estrazione Ticket JSON

Questo progetto mostra come effettuare il **Fine-Tuning di un LLM (Qwen2.5-0.5B-Instruct)** in modo efficiente per la RAM/VRAM sfruttando le tecniche di **Quantizzazione a 4-bit (QLoRA)** e **LoRA (Low-Rank Adaptation)**.

L'obiettivo è addestrare il modello a prendere in ingresso **ticket di assistenza clienti** non strutturati e convertirli rigorosamente in un **formato JSON strutturato**, prevenendo risposte errate e garantendo l'integrazione con database e sistemi di backend.

---

## 🧠 Architettura e Logica

1. **Quantizzazione 4-bit (NF4):** Riduce l'impronta di memoria del modello da 16/32-bit a soli 4-bit, congelando i pesi originali per non gravare sulla VRAM della GPU.
2. **LoRA (Low-Rank Adaptation):** Aggancia piccoli moduli di adattamento ("post-it") solo su specifici layer di attenzione (`q_proj`, `v_proj`). Solo lo ~0.1% dei pesi totali viene addestrato.
3. **Template & Processing:** Converte la chat a ruoli (`user`/`assistant`) nel formato nativo del modello e gestisce la generazione dei vettori numerici (`input_ids`) e della maschera d'attenzione (`attention_mask`).
4. **KV Cache Control:** Disattivata in fase di addestramento per risparmiare memoria e riattivata per un'inferenza fluida e veloce.

---

## 🛠️ Requisiti e Dipendenze

È consigliato l'uso di **Google Colab (GPU T4)** o un ambiente locale con supporto CUDA.

```bash
pip install torch transformers datasets peft bitsandbytes accelerator
```

---

## 💻 Struttura del Codice

### 1. Quantizzazione e Caricamento Modello

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
```

---

### 2. Dataset & Generazione Matrici (`input_ids` + `attention_mask`)

```python
from datasets import Dataset

data = [
    {"t": "Salve, vi scrivo perché il modem spedito lunedì non si accende proprio. Ho provato a cambiare presa ma niente. Voglio il rimborso o disdico tutto subito!", "r": '{"categoria": "Hardware / Guasto", "sentimento": "molto negativo", "priorita": "alta", "azione_richiesta": "Rimborso o sostituzione apparato"}'},
    {"t": "Buongiorno, la velocità della fibra è scesa moltissimo da ieri sera. Riscontro lentezza nello streaming.", "r": '{"categoria": "Connessione / Lentezza", "sentimento": "negativo", "priorita": "media", "azione_richiesta": "Verifica linea e reset segnale"}'},
    {"t": "Vorrei informazioni sul costo del passaggio alla tariffa 2.5 Gbps per i già clienti. Grazie.", "r": '{"categoria": "Commerciale / Info", "sentimento": "neutro", "priorita": "bassa", "azione_richiesta": "Invio prospetto offerte commerciale"}'},
    {"t": "Servizio pessimo!! Siete dei truffatori, mi avete addebitato 50 euro in più in fattura! Pretendo subito una spiegazione o vi denuncio!", "r": '{"categoria": "Fatturazione / Contestazione", "sentimento": "molto negativo", "priorita": "alta", "azione_richiesta": "Verifica amministrativa fattura e storno"}'},
    {"t": "Ho ricevuto il nuovo router stamattina, volevo solo ringraziare l'assistenza per la velocità. Installato e perfettamente funzionante!", "r": '{"categoria": "Feedback / Elogio", "sentimento": "positivo", "priorita": "bassa", "azione_richiesta": "Nessuna azione richiesta"}'},
    {"t": "Non riesco ad accedere all'area clienti dal sito web, mi dà errore 500 dopo il login.", "r": '{"categoria": "Piattaforma / Bug", "sentimento": "negativo", "priorita": "media", "azione_richiesta": "Segnalazione al team IT web"}'},
    {"t": "Mi è arrivata una SIM con numero errato, non corrisponde a quello che avevo richiesto in fase di contratto.", "r": '{"categoria": "Spedizioni / Errore", "sentimento": "negativo", "priorita": "alta", "azione_richiesta": "Invio nuova SIM corretta"}'},
    {"t": "È da due giorni che il telefono fisso non ha segnale, mentre internet funziona benissimo. Potete verificare?", "r": '{"categoria": "Voce / Guasto", "sentimento": "negativo", "priorita": "media", "azione_richiesta": "Controllo centrale telefonica"}'},
    {"t": "Vorrei disdire la mia linea a fine mese per cambio domicilio.", "r": '{"categoria": "Disdetta / Amministrativo", "sentimento": "neutro", "priorita": "media", "azione_richiesta": "Avvio pratica disattivazione"}'},
    {"t": "Complimenti per la gentilezza dell'operatore Marco che mi ha aiutato ieri al telefono!", "r": '{"categoria": "Feedback / Elogio", "sentimento": "molto positivo", "priorita": "bassa", "azione_richiesta": "Nessuna azione richiesta"}'}
]

texts = []
for item in data:
    prompt = f"""Sei un assistente per l'analisi dei ticket. Estrai le informazioni ed esegui l'output SOLTANTO in formato JSON con le chiavi: categoria, sentimento, priorita, azione_richiesta.

Ticket: "{item['t']}"
"""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": item['r']}
    ]
    texts.append(tokenizer.apply_chat_template(messages, tokenize=False))

tokenized_inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

dataset = Dataset.from_dict({
    "input_ids": tokenized_inputs["input_ids"],
    "attention_mask": tokenized_inputs["attention_mask"]
})
```

---

### 3. Setup Modello e Moduli LoRA

```python
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# Prepara il modello a 4-bit ed imposta use_cache = False per il training
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
```

---

### 4. Training Loop

```python
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./qwen-ticket-json",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=8,
    learning_rate=3e-4,
    fp16=False,
    bf16=False,
    logging_steps=1,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()
```

---

### 5. Inferenza ed Estrazione della Risposta

```python
# Riabilitazione della Cache e modalità di valutazione
model.config.use_cache = True
model.eval()

prompt_test = "Estrai informazioni dal ticket: 'Salve, da due giorni il telefono fisso non ha segnale.'"
messages = [{"role": "user", "content": prompt_test}]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

question_length = inputs.input_ids.shape[1]

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.01)

# Slicing per rimuovere la domanda ed isolare soltanto la risposta generata
response = tokenizer.decode(outputs[0][question_length:], skip_special_tokens=True)
print(response)
```

---

## ⚡ Risultati Attesi

Dopo l'addestramento, il modello genererà direttamente l'output formattato in **JSON pulito**:

```json
{
  "categoria": "Voce / Guasto",
  "sentimento": "negativo",
  "priorita": "media",
  "azione_richiesta": "Controllo centrale telefonica"
}
```