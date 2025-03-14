# Think Tank: KI-Agenten-Orchestrierungsplattform – Bedienungsanleitung

## Inhaltsverzeichnis

1.  [Einleitung](#einleitung)
2.  [Überblick über die Architektur](#architektur)
    *   [Kernkomponenten](#kernkomponenten)
    *   [Agenten](#agenten)
    *   [Orchestrator](#orchestrator)
    *   [Tools und Funktionen](#tools)
    *   [Sicherheitsmechanismen](#sicherheit)
3.  [Installation und Einrichtung](#installation)
    *   [Voraussetzungen](#voraussetzungen)
    *   [Installation](#installation-schritte)
    *   [Konfiguration](#konfiguration)
4.  [Verwendung](#verwendung)
    *   [Starten der Anwendung](#starten)
    *   [Interaktion mit der API](#api-interaktion)
    *   [Agenten verwalten](#agentenverwaltung)
    *   [Anfragen stellen](#anfragen)
    *   [Dateien hochladen und verarbeiten](#datei-upload)
    *   [Ergebnisse interpretieren](#ergebnisse)
5.  [Erweiterte Funktionen](#erweitert)
    *   [Caching](#caching)
    *   [Vektordatenbank](#vektordatenbank)
    *   [Blockchain (optional)](#blockchain)
    *   [Textanalyse](#textanalyse)
    *   [Eigene Tools hinzufügen](#eigene-tools)
6.  [Sicherheitshinweise](#sicherheitshinweise)
7.  [Fehlerbehebung](#fehlerbehebung)
8.  [Beispiele](#beispiele)
    *   [Einfache Agenten-Anfrage](#beispiel-agent)
    *   [Think-Tank-Diskussion](#beispiel-thinktank)
    *   [Datei-Upload und -Verarbeitung](#beispiel-datei)
9. [Testing](#testing)
10. [Beitragen](#beitragen)
11. [Lizenz](#lizenz)

## 1. Einleitung <a name="einleitung"></a>

"Think Tank" ist eine fortschrittliche Plattform zur Orchestrierung von KI-Agenten. Sie ermöglicht es, spezialisierte Agenten zu erstellen, zu konfigurieren und miteinander interagieren zu lassen, um komplexe Aufgaben zu lösen, Wissen zu generieren und Diskussionen zu führen. Die Plattform nutzt modernste Technologien wie Large Language Models (LLMs), Vektordatenbanken, eine sichere Ausführungsumgebung und optionale Blockchain-Integration.

## 2. Überblick über die Architektur <a name="architektur"></a>

### 2.1 Kernkomponenten <a name="kernkomponenten"></a>

*   **FastAPI:** Ein modernes, schnelles (High-Performance) Web-Framework zum Erstellen von APIs mit Python.
*   **Pydantic:** Eine Bibliothek zur Datenvalidierung und -einstellung, die Typannotationen verwendet.
*   **Redis:** Ein In-Memory-Datenspeicher, der als Cache verwendet wird, um die Antwortzeiten zu verkürzen.
*   **SQLite (Vektordatenbank-Ersatz):**  Anstelle einer dedizierten Vektordatenbank werden Embeddings direkt über Gemini bezogen und nur bei Bedarf berechnet. Eine SQLite-DB wird hier *nicht* verwendet.
*   **Google GenAI:** Die Google Generative AI API, die den Zugriff auf Googles LLMs ermöglicht (z. B. Gemini).
*   **Newspaper3k:** Eine Bibliothek zum Extrahieren von Text und Metadaten aus Webartikeln.
*   **NLTK, scikit-learn:** Bibliotheken für natürliche Sprachverarbeitung (NLP) und maschinelles Lernen, die für die Textanalyse verwendet werden.
*   **Cryptography:** Eine Bibliothek für Verschlüsselung, die zur Sicherung sensibler Daten verwendet wird.

### 2.2 Agenten <a name="agenten"></a>

*   **`Agent` Klasse:**  Die Basisklasse für alle Agenten.  Definiert die Kernattribute und -methoden, einschließlich:
    *   `agent_id`: Eindeutige ID.
    *   `name`:  Anzeigename des Agenten.
    *   `description`:  Kurze Beschreibung der Fähigkeiten.
    *   `system_prompt`:  Anweisung an das LLM, die das Verhalten des Agenten steuert.
    *   `role`:  Vordefinierte Rolle (z.B. Analyst, Stratege, Kritiker, ...).
    *   `temperature`:  Steuert die Zufälligkeit der LLM-Antworten.
    *   `model_name`:  Das verwendete LLM (z.B. "gemini-2.0-flash").
    *   `tools`:  Liste der verfügbaren Tools (Funktionen).
    *   `caching`:  Aktiviert/deaktiviert das Caching für den Agenten.
    *   `generate_response()`:  Methode zur Generierung von Antworten auf Anfragen.

*   **Agenten-Rollen (`AgentRole` Enum):**  Vordefinierte Rollen, die den Agenten zugewiesen werden können, um ihre Spezialisierung und Verantwortlichkeiten zu definieren. Beispiele:
    *   `ANALYST`: Führt Datenanalysen durch.
    *   `STRATEGIST`: Entwickelt Strategien.
    *   `CRITIC`:  Hinterfragt und bewertet Annahmen.
    *   `SUMMARIZER`:  Fasst Informationen zusammen.
    *   `VERIFIER`:  Überprüft Fakten.
    *   ... (und viele mehr)

### 2.3 Orchestrator <a name="orchestrator"></a>

*   **`Orchestrator` Klasse:**  Verwaltet die Agenten, leitet Anfragen weiter und koordiniert die Interaktionen.
    *   `load_agents_from_config()`:  Lädt Agentenkonfigurationen aus einer JSON-Datei.
    *   `add_agent()`:  Fügt einen Agenten hinzu.
    *   `process_request()`:  Leitet eine Anfrage an einen Agenten weiter.
    *   `remove_agent()`:  Entfernt einen Agenten.
    *   `get_all_agents()`:  Gibt eine Liste aller Agenten zurück.

### 2.4 Tools und Funktionen <a name="tools"></a>

*   **`google_search()`:**  Führt eine Websuche mit der Google GenAI API durch.
*   **`crawl_website()`:**  Extrahiert Textinhalte von einer Webseite.
*   **`analyze_text_complexity()`:** Analysiert die Komplexität eines Textes (Satzlänge, Wortanzahl, TF-IDF).
*   **`process_file()`:** Verarbeitet hochgeladene Dateien (liest den Inhalt aus).

### 2.5 Sicherheitsmechanismen <a name="sicherheit"></a>

*   **Sichere Sandbox-Umgebung (`safe_execution_environment`):**
    *   Verwendet `subprocess` und `signal`, um Code in einem isolierten Prozess mit Timeout auszuführen.  *Hinweis:* Im aktuellen Code ist die Ausführung von Python-Code aus Sicherheitsgründen deaktiviert.
    *   Schützt die Hauptanwendung vor potenziell schädlichem Code.

*   **Rate Limiter (`RateLimiter`):**
    *   Begrenzt die Anzahl der API-Aufrufe pro Zeiteinheit.
    *   Verhindert Überlastung der API und mögliche Kostenüberschreitungen.

*   **Validierung von Benutzereingaben:**
    *   `UploadFileModel`: Pydantic-Modell zur Validierung von Dateinamen und -inhalten beim Hochladen. Verhindert Path-Traversal-Angriffe und beschränkt die Dateigröße.
    *   `is_valid_url()`:  Überprüft, ob eine URL gültig ist.

*   **Verschlüsselung (`Fernet`):**
    *   `ENCRYPTION_KEY`:  Ein geheimer Schlüssel, der zur Ver- und Entschlüsselung sensibler Daten verwendet werden kann (optional).

*   **API-Schlüssel-Validierung (`validate_api_key()`):**
    *   Überprüft die Gültigkeit von API-Schlüsseln (in der aktuellen Implementierung ein Platzhalter).

*   **Blockchain (optional, `USE_BLOCKCHAIN`):**
    *   `generate_block_hash()`, `add_block_to_chain()`, `validate_chain()`: Funktionen zur Erstellung und Validierung einer einfachen Blockchain.  Kann verwendet werden, um die Integrität von Daten und Interaktionen zu gewährleisten (optional).

*   **Entfernung der `execute_python_code` Funktion:**
    * Aus Sicherheitsgründen wurde die Funktion zur Ausführung von Python-Code entfernt, um die Ausführung von nicht vertrauenswürdigem Code zu verhindern.

## 3. Installation und Einrichtung <a name="installation"></a>

### 3.1 Voraussetzungen <a name="voraussetzungen"></a>

*   Python 3.9 oder höher
*   `pip` (Python-Paketmanager)
*   Zugriff auf die Google GenAI API (API-Schlüssel erforderlich)
*   (Optional) Redis-Server für Caching
* NLTK-Daten (werden bei Bedarf heruntergeladen)
* (Optional) sklearn (scikit-learn)

### 3.2 Installation <a name="installation-schritte"></a>

1.  **Repository klonen (oder Code herunterladen):**

    ```bash
    git clone <repository_url>  # Falls von einem Git-Repository
    cd think_tank
    ```

2.  **Abhängigkeiten installieren:**

    ```bash
    pip install -r requirements.txt
    ```
    Erstellen Sie gegebenenfalls eine `requirements.txt` Datei mit folgendem Inhalt:

    ```
    fastapi
    uvicorn
    pydantic
    pydantic-settings
    python-multipart
    google-generativeai
    newspaper3k
    nltk
    scikit-learn
    cryptography
    redis
    aiohttp
    requests
    ```

3. **NLTK-Daten herunterladen (einmalig):**
   Das Programm lädt die benötigten Daten (stopwords, punkt) automatisch herunter, falls sie fehlen. Wenn Sie die Daten manuell herunterladen möchten, können Sie dies tun:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```

### 3.3 Konfiguration <a name="konfiguration"></a>

*   **`agent_config.json`:**
    *   Diese Datei enthält die Konfigurationen für die Agenten.  Sie wird beim Start des Orchestrators geladen.
    *   Beispiel:

        ```json
        [
          {
            "name": "Experte für alternative Fakten",
            "description": "Kennt sich sehr gut mit alternativen Fakten aus",
            "system_prompt": "Du bist ein Experte für Verschwörungstheorien.",
            "role": "ANALYST",
            "temperature": 0.7,
            "model_name": "gemini-2.0-flash",
            "expertise_fields": ["Verschwörungen", "alternative Fakten"],
            "caching": true
          },
          {
            "name": "Faktenprüfer",
            "description": "Spezialist für Faktenprüfung und Aufklärung",
            "system_prompt": "Du bist ein Faktenprüfer und Experte für Desinformation.",
            "role": "VERIFIER",
            "temperature": 0.2,
            "model_name": "gemini-2.0-flash",
            "expertise_fields": ["Faktenprüfung", "Verifizierung"],
            "caching": true
          }
        ]
        ```
    *   **Attribute:**
        *   `name`: (erforderlich) Der Name des Agenten.
        *   `description`: (erforderlich) Eine Beschreibung des Agenten.
        *   `system_prompt`: (erforderlich) Die Systemanweisung für den Agenten.
        *   `role`: (erforderlich) Die Rolle des Agenten (muss ein Wert aus der `AgentRole` Enum sein).
        *   `temperature`: (optional) Die Temperatur für das LLM (Standard: `DEFAULT_TEMPERATURE` aus `Settings`).
        *   `model_name`: (optional) Der Name des zu verwendenden Modells (Standard: "gemini-2.0-flash").
        *   `expertise_fields`: (optional) Eine Liste von Fachgebieten des Agenten.
        *   `caching`: (optional) Gibt an, ob das Caching für den Agenten aktiviert ist (Standard: `True`).

*   **Umgebungsvariablen (oder `.env`-Datei):**
    *   Die `Settings`-Klasse verwendet `pydantic-settings`, um Konfigurationsparameter zu laden. Sie können diese Parameter als Umgebungsvariablen setzen oder eine `.env`-Datei im Projektverzeichnis erstellen.
    *   Beispiel `.env`-Datei:

        ```
        GEMINI_API_KEY=dein_google_genai_api_key
        ENCRYPTION_KEY=dein_geheimer_verschluesselungsschluessel
        REDIS_HOST=localhost
        REDIS_PORT=6379
        # ... weitere Einstellungen
        ```
    *   **Wichtige Einstellungen:**
        *   `GEMINI_API_KEY`:  Ihr Google GenAI API-Schlüssel (erforderlich).
        *   `ENCRYPTION_KEY`: Ein sicherer Schlüssel für die Verschlüsselung (optional, wird automatisch generiert, wenn nicht angegeben).  **Bewahren Sie diesen Schlüssel sicher auf!**
        *   `REDIS_HOST`, `REDIS_PORT`:  Adresse und Port des Redis-Servers (optional).
        *   `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P`, `DEFAULT_TOP_K`, `DEFAULT_MAX_OUTPUT_TOKENS`: Standardwerte für die LLM-Parameter.
        *   `API_CALL_INTERVAL`:  Intervall zwischen API-Aufrufen (in Sekunden).
        *   `MAX_CONCURRENT_REQUESTS`:  Maximale Anzahl gleichzeitiger Anfragen.
        *   `SANDBOX_TIMEOUT`:  Timeout für die Codeausführung in der Sandbox (in Sekunden).
        *  `WEB_CRAWLING_TIMEOUT`: Timeout für das Crawlen von Webseiten.
        *   `MAX_URLS_TO_CRAWL`:  Maximale Anzahl zu crawlender URLs.
        *  `MAX_FILE_SIZE_KB`: Maximale Größe für hochgeladene Dateien.
        *  `FILE_UPLOAD_DIR`: Verzeichnis für hochgeladene Dateien.
        *   `CACHE_DIR`:  Cache-Verzeichnis (wird nicht mehr direkt verwendet, da Redis genutzt wird).
        *   `VECTORDB_PATH`: Pfad zur SQLite-Vektordatenbank (wird nicht verwendet, da Embeddings direkt über Gemini geholt werden).
        *   `EMBEDDING_MODEL`: Name des Embedding-Modells.
        *   `USE_BLOCKCHAIN`:  Aktiviert/deaktiviert die Blockchain-Funktionalität (Standard: `True`).
        * `ALLOW_ORIGINS`: Liste der erlaubten Ursprünge für CORS (Cross-Origin Resource Sharing) – in der Produktion anpassen!
        * `JWT_SECRET_KEY`: Geheimer Schlüssel für JSON Web Tokens (wird automatisch generiert, kann aber auch manuell gesetzt werden).
        * `MAX_DISCUSSION_ROUNDS`: Maximale Anzahl von Diskussionsrunden im Think Tank.

## 4. Verwendung <a name="verwendung"></a>

### 4.1 Starten der Anwendung <a name="starten"></a>

Führen Sie die `main.py`-Datei aus:

```bash
python main.py
```
Dadurch wird die FastAPI-Anwendung mit Uvicorn gestartet.  Die API ist standardmäßig unter `http://localhost:8000` verfügbar.

### 4.2 Interaktion mit der API <a name="api-interaktion"></a>

Die API bietet verschiedene Endpunkte:

*   **`/agents/` (GET):**  Ruft eine Liste aller registrierten Agenten ab.
    *   **Antwort:**  Eine Liste von JSON-Objekten, die die Agenten repräsentieren (ID und Name).
    *   **Beispiel:** `curl http://localhost:8000/agents/`

*   **`/ask_agent/` (POST):**  Sendet eine Anfrage an einen einzelnen Agenten.
    *   **Anfrage (JSON):**
        ```json
        {
          "agent_id": "die_id_des_agenten",
          "query": "Deine Frage an den Agenten"
        }
        ```
    *   **Antwort (JSON):**
        ```json
        {
          "response": "Die Antwort des Agenten"
        }
        ```

*   **`/ask_think_tank/` (POST):**  Startet eine Diskussion zwischen mehreren Agenten.
    *   **Anfrage (JSON):**
        ```json
        {
          "agent_ids": ["id_agent_1", "id_agent_2", ...],
          "query": "Die Frage, über die diskutiert werden soll",
          "rounds": 3  // Anzahl der Diskussionsrunden (optional, Standard: 3)
        }
        ```
    *   **Antwort (JSON):**
        ```json
        {
            "discussion_history": [
                {"agent_id": "id_agent_1", "response": "Antwort von Agent 1"},
                {"agent_id": "id_agent_2", "response": "Antwort von Agent 2"},
                ...
            ],
            "final_response": "Die letzte Antwort im Diskussionsverlauf"
        }
        ```

*   **`/upload_file/` (POST):**  Lädt eine Datei hoch.
    *   **Anfrage (JSON):**
        ```json
        {
          "filename": "dateiname.txt",
          "content": "Base64-codierter Inhalt der Datei"
        }
        ```
    *   **Antwort (JSON):**
        ```json
        {
          "message": "Datei 'dateiname.txt' erfolgreich hochgeladen."
        }
        ```
        Oder eine Fehlermeldung, falls der Upload fehlschlägt.
        * Stelle sicher, dass der Inhalt der Datei Base64-codiert ist, bevor du die Anfrage sendest.

*  **`/testendpoint/` (GET):** Ein einfacher Testendpunkt, der immer `{"answer": 1}` zurückgibt.  Dient zur Überprüfung, ob der Server läuft.
    * **Beispiel:** `curl http://localhost:8000/testendpoint/`

### 4.3 Agenten verwalten <a name="agentenverwaltung"></a>

*   **Hinzufügen von Agenten:**
    *   Agenten können über die `agent_config.json`-Datei hinzugefügt werden.  Der Orchestrator lädt diese Datei beim Start.
    *   Sie können auch Agenten dynamisch hinzufügen, indem Sie `orchestrator.add_agent()` in Ihrem Code verwenden.
*   **Entfernen von Agenten:**
    *   Verwenden Sie `orchestrator.remove_agent(agent_id)`, um einen Agenten zu entfernen.
* **Auflisten von Agenten:**
    * Rufen Sie eine Liste aller Agenten mit `orchestrator.get_all_agents()` ab, oder verwenden Sie den `/agents/` API-Endpunkt.

### 4.4 Anfragen stellen <a name="anfragen"></a>
*  **An einen einzelnen Agenten (`/ask_agent/`):**
    *  Sende eine POST-Anfrage mit der `agent_id` und der `query` im JSON-Format.
*   **An den Think Tank (`/ask_think_tank/`):**
    *  Sende eine POST-Anfrage mit einer Liste von `agent_ids`, der `query` und optional der Anzahl der `rounds`.

### 4.5 Dateien hochladen und verarbeiten <a name="datei-upload"></a>

1.  **Datei hochladen (`/upload_file/`):**
    *   Senden Sie eine POST-Anfrage mit dem `filename` und dem Base64-codierten `content` der Datei.
2.  **Datei verarbeiten:**
    *   Verwenden Sie die `process_file(filename, instructions)`-Funktion, um die hochgeladene Datei zu verarbeiten.  Die `instructions` geben an, wie die Datei verarbeitet werden soll (z. B. "extrahiere Schlüsselwörter", "fasse den Inhalt zusammen"). Derzeit gibt die Funktion nur den Inhalt der Datei zurück.

### 4.6 Ergebnisse interpretieren <a name="ergebnisse"></a>

*   **Einzelne Agenten-Antwort:**  Die Antwort ist ein String, der die Antwort des Agenten auf die Anfrage enthält.
*   **Think-Tank-Diskussion:**
    *   `discussion_history`:  Eine Liste von Dictionaries, die den Verlauf der Diskussion protokollieren.  Jeder Eintrag enthält die `agent_id` und die `response` des Agenten.
    *   `final_response`:  Die letzte Antwort im Diskussionsverlauf. Dies ist die abschließende Antwort nach allen Diskussionsrunden.

## 5. Erweiterte Funktionen <a name="erweitert"></a>

### 5.1 Caching <a name="caching"></a>

*   **Redis-Cache:**  Das System verwendet Redis, um Antworten von Agenten zu cachen.
    *   `RedisCache`-Klasse:  Kapselt die Interaktion mit Redis.
    *   `get_cached_response()`, `cache_response()`: Methoden der `Agent`-Klasse zum Abrufen und Speichern von Antworten im Cache.
    *   Cache-Schlüssel werden mit `generate_cache_key()` generiert.  Sie basieren auf dem Agentennamen, dem Wissen, dem Verlauf und der Anfrage.
*   **Caching aktivieren/deaktivieren:**  Setzen Sie das `caching`-Attribut des Agenten in der Konfiguration auf `True` oder `False`.

### 5.2 Vektordatenbank <a name="vektordatenbank"></a>

*   **`VectorDatabase` Klasse:**  In dieser Implementierung wird *keine* lokale SQLite-Datenbank verwendet. Stattdessen werden die Vektor-Embeddings direkt von Google Gemini bezogen, wann immer sie benötigt werden.
    *   `get_gemini_embedding()`:  Ruft das Embedding für einen Text von Google Gemini ab.
    *   `search()`:  Führt eine Ähnlichkeitssuche durch. Gibt in dieser Implementierung direkt das Embedding zurück.

### 5.3 Blockchain (optional) <a name="blockchain"></a>

*   **`USE_BLOCKCHAIN` Einstellung:**  Aktiviert/deaktiviert die Blockchain-Funktionalität.
*   **Funktionen:**
    *   `generate_block_hash()`:  Generiert einen Hash für einen Block.
    *   `add_block_to_chain()`:  Fügt einen neuen Block zur Blockchain hinzu.
    *   `validate_chain()`:  Überprüft die Integrität der Blockchain.
*   **Verwendung:**  Die Blockchain kann verwendet werden, um Interaktionen zwischen Agenten oder andere Daten unveränderlich zu protokollieren.  Die aktuelle Implementierung ist eine einfache Demonstration.

### 5.4 Textanalyse <a name="textanalyse"></a>

*   **`analyze_text_complexity()` Funktion:**
    *   Analysiert die Komplexität eines Textes.
    *   Verwendet NLTK und scikit-learn.
    *   Berechnet:
        *   Anzahl der Sätze
        *   Anzahl der Wörter
        *   Durchschnittliche Wörter pro Satz
        *   TF-IDF-Dichte (optional)

### 5.5 Eigene Tools hinzufügen <a name="eigene-tools"></a>

1.  **Tool-Funktion erstellen:**
    *   Erstellen Sie eine asynchrone Funktion, die die Logik des Tools implementiert.  Beispiel:

        ```python
        async def my_new_tool(param1: str, param2: int) -> str:
            """Beschreibung des Tools."""
            # Tool-Logik hier
            return "Ergebnis des Tools"
        ```

2.  **Tool-Definition erstellen:**
    *   Erstellen Sie ein Dictionary, das das Tool für die Google GenAI API beschreibt.  Siehe `create_tool_definitions()` für Beispiele.

3.  **Tool zum Agenten hinzufügen:**
    *   Fügen Sie die Tool-Definition zur `tools`-Liste des Agenten in der `agent_config.json`-Datei hinzu.

## 6. Sicherheitshinweise <a name="sicherheitshinweise"></a>

*   **API-Schlüssel:**  Bewahren Sie Ihren Google GenAI API-Schlüssel sicher auf.  Verwenden Sie Umgebungsvariablen oder eine `.env`-Datei.  Geben Sie den Schlüssel *nicht* im Code an und checken Sie ihn *nicht* in Versionskontrollsysteme ein.
*   **Verschlüsselungsschlüssel:** Wenn Sie die Verschlüsselung verwenden, bewahren Sie den `ENCRYPTION_KEY` sicher auf.  Verlust des Schlüssels führt zum Verlust des Zugriffs auf verschlüsselte Daten.
*   **Benutzereingaben:**  Validieren Sie alle Benutzereingaben sorgfältig, um Sicherheitslücken wie Path Traversal und Code Injection zu vermeiden.
*   **Codeausführung:**  Die Ausführung von beliebigem Python-Code ist aus Sicherheitsgründen deaktiviert.  Aktivieren Sie die `execute_python_code`-Funktionalität *nicht* in einer Produktionsumgebung.
*   **Abhängigkeiten:**  Halten Sie Ihre Abhängigkeiten auf dem neuesten Stand, um Sicherheitslücken zu vermeiden.
*   **CORS:**  In einer Produktionsumgebung sollten Sie `ALLOW_ORIGINS` auf die spezifischen Ursprünge beschränken, von denen aus auf Ihre API zugegriffen werden darf. Verwenden Sie *nicht* `["*"]` in der Produktion.
* **Rate Limiting:** Implementieren Sie ein striktes Rate Limiting, um Missbrauch und Denial-of-Service-Angriffe zu verhindern.
* **Logging:** Überwachen Sie die Logs regelmäßig auf verdächtige Aktivitäten.
* **Redis Sicherheit:** Wenn Sie Redis verwenden, sichern Sie Ihren Redis-Server ab. Setzen Sie ein starkes Passwort und beschränken Sie den Zugriff auf vertrauenswürdige Netzwerke.

## 7. Fehlerbehebung <a name="fehlerbehebung"></a>

*   **Logging:**  Das System verwendet das `logging`-Modul, um Informationen, Warnungen und Fehler zu protokollieren.  Die Logdatei (`think_tank.log`) enthält detaillierte Informationen zur Fehlerbehebung.
*   **Exceptions:**  Das System verwendet benutzerdefinierte Exceptions (`ThinkTankError`, `APIConnectionError`), um Fehler zu behandeln.
*   **FastAPI Exception Handling:**  FastAPI bietet integrierte Mechanismen zur Fehlerbehandlung.  Der `think_tank_exception_handler` fängt `ThinkTankError`-Exceptions ab und gibt eine JSON-Fehlermeldung zurück.
* **TimeoutException:** Die `TimeoutException` wird ausgelöst, wenn die Codeausführung in der Sandbox das Zeitlimit überschreitet.
* **HTTPException:** FastAPI's `HTTPException` wird verwendet, um HTTP-Fehlercodes und -meldungen zurückzugeben (z.B. bei ungültigen URLs, fehlenden Dateien, etc.).
* **Tracebacks:** Bei unerwarteten Fehlern werden vollständige Tracebacks in die Logdatei geschrieben, um die Diagnose zu erleichtern.

## 8. Beispiele <a name="beispiele"></a>

### 8.1 Einfache Agenten-Anfrage <a name="beispiel-agent"></a>
HTML/JS (index.html):
```html
<!DOCTYPE html>
<html>
<head>
    <title>Agenten-Anfrage</title>
</head>
<body>
    <h1>Agenten-Anfrage</h1>
    <select id="agentSelect"></select>
    <input type="text" id="frage" placeholder="Deine Frage">
    <button onclick="sendFrage()">Senden</button>
    <div id="antwort"></div>

    <script>
    // Agenten laden (vereinfacht)
    async function loadAgenten() {
        const response = await fetch('/agents/');
        const agenten = await response.json();
        const select = document.getElementById('agentSelect');
        agenten.forEach(agent => {
            const option = document.createElement('option');
            option.value = agent.agent_id;
            option.textContent = agent.name;
            select.appendChild(option);
        });
    }
    loadAgenten();


    async function sendFrage() {
        const agentId = document.getElementById('agentSelect').value;
        const frage = document.getElementById('frage').value;
        const response = await fetch('/ask_agent/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ agent_id: agentId, query: frage })
        });
        const data = await response.json();
        document.getElementById('antwort').textContent = data.response;
    }
    </script>
</body>
</html>
```

### 8.2 Think-Tank-Diskussion <a name="beispiel-thinktank"></a>
HTML/JS (index.html, gekürzt - vollständige Version siehe im ursprünglichen Code):

```html
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <title>KI-Agenten Anfrage</title>
    <!-- ... (CSS und Font-Links) ... -->
</head>
<body>
    <div class="container">
        <h1>KI-Agenten Anfrage</h1>
        <!-- ... (Formular-Elemente) ... -->
        <button onclick="frageSenden('thinkTank')">Frage an Think Tank senden</button>
        <h2>Antwort des Agenten:</h2>
        <div id="antwort"></div>
    </div>
    <script>
    // ... (Funktionen zum Laden der Agenten und Senden der Anfrage) ...
    // window.onload = ... (siehe vollständigen Code)
    async function frageSenden(type) {
    // ... (siehe vollständigen Code für Details, Umgang mit Diskussion, etc.) ...
    }
    </script>
</body>
</html>
```
Python (Ausschnitt aus `main.py`):

```python
# ... (vorheriger Code) ...

@app.post("/ask_think_tank/")
async def ask_think_tank(request: ThinkTankRequest):
    # ... (Logik für die Think-Tank-Diskussion, siehe vollständiger Code) ...
```

### 8.3 Datei-Upload und -Verarbeitung <a name="beispiel-datei"></a>
HTML:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Datei-Upload</title>
</head>
<body>
    <h1>Datei hochladen</h1>
    <input type="file" id="dateiInput">
    <button onclick="dateiHochladen()">Hochladen</button>
    <div id="antwort"></div>
<script>
    async function dateiHochladen() {
        const datei = document.getElementById('dateiInput').files[0];
        if (!datei) {
            alert('Bitte wähle eine Datei aus!');
            return;
        }
        const reader = new FileReader();
        reader.onload = async function(event) {
            const base64Content = event.target.result.split(',')[1]; // Nur den Base64-Teil extrahieren
            const response = await fetch('/upload_file/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: datei.name, content: base64Content })
            });
            const data = await response.json();
            document.getElementById('antwort').textContent = data.message;
        };
        reader.readAsDataURL(datei); // Datei als Base64 lesen
    }
</script>
</body>
</html>
```
Python (Ausschnitt aus `main.py`):

```python
# ... (vorheriger Code)

@app.post("/upload_file/")
async def upload_file_endpoint(file: UploadFileModel):  # Validierung durch Pydantic
	# ... (Logik für Datei-Upload, siehe vollständiger Code) ...
```

## 9. Testing <a name="testing"></a>
Das Projekt enthält Unittests, die mit der `unittest` Bibliothek geschrieben wurden. Die Tests decken die folgenden Bereiche ab:
* **RateLimiter:** Überprüft, ob der Rate Limiter korrekt funktioniert und Anfragen verzögert, wenn das Limit erreicht ist.
* **Agent:** Testet die `generate_response` Methode des Agenten, um sicherzustellen, dass eine Antwort generiert wird.
* **Orchestrator:** Testet die `process_request` Methode des Orchestrators, um sicherzustellen, dass Anfragen korrekt an Agenten weitergeleitet werden.

Die Tests können mit dem folgenden Befehl ausgeführt werden:
```bash
python main.py
```
Die Testergebnisse werden in der Konsole ausgegeben.

## 10. Beitragen <a name="beitragen"></a>

Beiträge zum Projekt sind willkommen!  Wenn Sie Fehler finden, Verbesserungsvorschläge haben oder neue Funktionen hinzufügen möchten, erstellen Sie bitte einen Pull Request.

## 11. Lizenz <a name="lizenz"></a>

Dieses Projekt ist [MIT-lizenziert – Details siehe LICENSE-Datei](https://opensource.org/license/mit/).  (Fügen Sie eine LICENSE-Datei hinzu, wenn Sie das Projekt veröffentlichen.)

*

