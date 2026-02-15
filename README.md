# Quantum Entanglement & Bell Violation Classifier

Progetto di Machine Learning applicato alla Fisica Quantistica per la classificazione di stati bipartiti (due qubit). Il sistema utilizza reti neurali profonde (**PyTorch**) e support vector machines per mappare la frontiera tra stati separabili, entangled e non-locali.

---

##  Descrizione del Progetto

Il progetto affronta il problema della caratterizzazione dell'entanglement in sistemi quantistici descritti da matrici di densità $\rho$ di dimensione $4 \times 4$. L'obiettivo è addestrare modelli in grado di distinguere:
1.  **Criterio di Peres-Horodecki (PPT):** Identificazione degli stati *Separabili* vs *Entangled*.
2.  **Violazione delle Disuguaglianze di Bell:** Identificazione degli stati che manifestano correlazioni non-locali.

Il workflow include il preprocessing fisico delle matrici, la riduzione della dimensionalità tramite **PCA** (Principal Component Analysis) e l'ottimizzazione di un'architettura **Multi-Layer Perceptron (MLP)**.

---

## Caratteristiche Tecniche

* **Dataset:** Matrici di densità quantistiche ricostruite (32 parametri reali per stato).
* **Riduzione Dimensionale:** PCA per il mantenimento del 95% della varianza.
* **Deep Learning:** Implementazione di `QuantumMLP` in PyTorch.
* **Ottimizzazione:** Grid Search sistematica per funzioni di attivazione (SiLU, ELU, LeakyReLU) e ottimizzatori (AdamW, RMSprop).
* **Gestione Sbilanciamento:** Bilanciamento delle classi tramite `pos_weight` per la gestione della rarità degli stati che violano Bell.

---

## Struttura del Repository

* `Esame.ipynb`: Notebook principale contenente l'intero workflow di analisi e addestramento.
* `funzioni.py`: Modulo Python con le utility per la ricostruzione delle matrici e il plotting professionale.
* `data_gen.cpp`: Main per la generazione del Dataset 
* `funzioni.h`: Funzioni per la generazione del Dataset
* `Makefile`
