# Automatic-Threat-Detection: an Ensemble Model for Honeypots
# Descrizione del Progetto
Questo progetto di tesi magistrale in Cybersecurity si concentra sull'implementazione di un sistema di rilevamento automatico delle minacce utilizzando un modello di ensemble. Il sistema è progettato per analizzare gli attacchi raccolti da un honeypot (Cowrie) e classificarli utilizzando un approccio basato su un ensemble di modelli di machine learning: CNN, Bi-Directional LSTM (Bi-LSTM) e GRU. Successivamente, i risultati della classificazione vengono inviati a Splunk per l'analisi e la visualizzazione, con un'integrazione aggiuntiva di AbuseIPDB per la verifica delle IP malevole.

# Struttura del Progetto
- Modelli Utilizzati: CNN, BDLSTM, GRU
- Honeypot: Cowrie (per la configurazione di Cowrie, fare riferimento alla pagina ufficiale)
- Analisi dei Dati: Splunk, con integrazione di AbuseIPDB
  
# Requisiti
- Conda (distribuzione Anaconda o Miniconda)
- Librerie Python necessarie (indicate nel file requirements.txt)
- Cowrie honeypot configurato e funzionante
- Splunk configurato e funzionante
- Account su AbuseIPDB (per ottenere l'API key necessaria)

# Configurazione
1. Installazione delle dipendenze:
```bash
pip install -r requirements.txt
```

2. Configurazione di Cowrie:
- Seguire le istruzioni fornite nella pagina ufficiale di Cowrie per configurare e avviare Cowrie.
Configurazione di Splunk:

3. Configurare Splunk:
- Per ricevere i dati in tempo reale. Assicurarsi che Splunk sia in esecuzione e configurato per accettare input dai file di log generati da Cowrie.

4. Integrazione con AbuseIPDB:
- Inserire la vostra API key di AbuseIPDB nel file di configurazione del progetto.

# Esecuzione del Progetto
1. Avviare Cowrie:
```bash
./cowrie start
```
2. Avviare Splunk:
Assicurarsi che Splunk sia in esecuzione e pronto per ricevere i dati.

3. Avviare lo script di preprocessing e predizione:
```bash
python preprocessing_predict.py
```
Questo script si occuperà di:
- Preprocessare i dati raccolti da Cowrie
- Classificare gli attacchi utilizzando il modello di ensemble
- Inviare i risultati della classificazione a Splunk per l'analisi

# Output
I risultati della classificazione degli attacchi saranno visualizzati in Splunk, dove potranno essere analizzati ulteriormente. I dettagli sugli indirizzi IP sospetti saranno arricchiti con i dati di AbuseIPDB per fornire ulteriori informazioni sulle potenziali minacce.

# Contributi e Manutenzione
Per contribuire a questo progetto, si prega di creare un fork del repository e proporre una pull request con le modifiche.
