# PatchScout: OSRS Patch Note Search & Classification Engine

**Course:** CS410 Text Information Systems (Fall 2025)  
**Author:** Brendan Pan (bcpan2)  

## Project Overview
**PatchScout** is a specialized text information system designed for *Old School RuneScape* (OSRS). It solves the problem of navigating over a decade of unstructured game updates by aggregating them into a searchable, classified database.

Unlike standard "Ctrl+F" searches, PatchScout allows players to:
1.  **Search Semantically:** Find updates about "training methods" even if the text only mentions "XP rates."
2.  **Filter by Intent:** Use AI classification to isolate specific types of changes (e.g., "Show me Combat Nerfs, hide Bug Fixes").
3.  **Classify Live Text:** Input theoretical patch notes to see how the machine learning model categorizes them.

---

## How to Use the Software

### 1. Installation
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/YOUR_USERNAME/cs410-patch-scout.git](https://github.com/YOUR_USERNAME/cs410-patch-scout.git)
cd cs410-patch-scout
pip install pandas numpy scikit-learn rank_bm25 sentence-transformers faiss-cpu beautifulsoup4 requests
```

### 2. Data Generation (Optional)
The project comes with a pre-scraped dataset (`osrs_master_dataset.csv`). If you wish to regenerate the data from scratch (fetching the latest updates from 2013–2025), run the following command in your terminal:

```bash
`python scrape_official_full.py`
```

*Note: This script includes intentional delays to respect Jagex's rate limits. A full scrape takes approximately 30 minutes.*

### 3. Running the PatchScout Tool
To launch the interactive search and classification engine, run the command:

```bash
python patch_scout_app.py
```

**Main Menu Options:**
* **[1] Search Patch Notes:**
    * Enter a query (e.g., "nerfs to magic" or "Scythe of Vitur").
    * Optionally apply a **Category Filter** (e.g., only show `Combat Balance`).
    * The system returns the top results ranked by relevance, with their AI-predicted tags.
* **[2] Classify New Text:**
    * Enter any text string simulating a patch note.
    * The system uses the trained SVM model to predict its category (e.g., `Bug Fix`, `Mobile/UI`).

### 4. Running the Evaluation
To reproduce the Map and NDCG scores used in the report, run the command:

```bash
python evaluate_search.py
```

This script runs a test set of 10 "Gold Standard" queries against both the Keyword and Semantic engines.

---

## Technical Implementation

### 1. Data Acquisition (`scrape_official_full.py`)
Data is collected from the **Official OSRS News Archive** (`secure.runescape.com`).
* **Method:** The script iterates through the archive by Year (2013–2025) and Month, extracting articles tagged as "Game Updates."
* **Parsing:** It utilizes `BeautifulSoup` to parse HTML and isolate the content div.
* **Smart Labeling:** As data is scraped, a rule-based labeling system tags entries based on keywords (e.g., "render" -> `Mobile/UI`, "fixed" -> `Bug Fix`). This creates the "Silver Standard" dataset used for training the classifier.
* **Anti-Botting:** Random sleep intervals (1.0–3.0s) are implemented to prevent IP blocking.

### 2. Search Engine Architecture (`patch_scout_app.py`)
The tool implements a **Hybrid Search** strategy:

* **Lexical Layer (BM25):**
    * Uses `rank_bm25` to index the corpus.
    * Best for: Exact item names (e.g., "Fang", "Blowpipe").
    * *Implementation:* Tokenizes text into lower-case words to compute Term Frequency-Inverse Document Frequency (TF-IDF) scores.

* **Semantic Layer (FAISS + S-BERT):**
    * Uses `sentence-transformers` (Model: `all-MiniLM-L6-v2`) to convert patch notes into 384-dimensional dense vectors.
    * Uses `faiss` (Facebook AI Similarity Search) for efficient L2 distance retrieval.
    * Best for: Concepts and intents (e.g., "ways to train faster").

### 3. Automated Classification (`patch_scout_app.py`)
A supervised machine learning model categorizes every patch note into 5 classes: `Bug Fix`, `XP/Progression`, `Combat Balance`, `Mobile/UI`, and `Quest/Lore`.

* **Feature Extraction:** Text is vectorized using `TfidfVectorizer` with Bigrams. This allows the model to capture phrases like "mobile client" or "xp rate" rather than just single words.
* **Model:** A **Linear Support Vector Machine (LinearSVC)** is used for classification.
* **Class Imbalance Handling:** The model is initialized with balanced class weights. This automatically adjusts weights inversely proportional to class frequencies, ensuring rare categories (like `Mobile/UI`) are not ignored in favor of common ones (`Bug Fix`).

### 4. File Structure
* `patch_scout_app.py`: The main CLI application containing the `PatchScoutTool` class.
* `evaluate_search.py`: Automated grading script calculating Mean Average Precision (MAP) and NDCG@10.
* `scrape_official_full.py`: The web scraper for the official Jagex archive.
* `osrs_master_dataset.csv`: The processed dataset containing ~1,800 labeled patch notes.

## Evaluation Results
(Generated via `evaluate_search.py`)

| Metric | Keyword Search (BM25) | Semantic Search (FAISS) |
| :--- | :--- | :--- |
| **Mean Average Precision (MAP)** | 1.0000 | 0.7683 |
| **Average NDCG@10** | 1.0000 | 0.8503 |

*Note on Metrics:* BM25 achieved perfect scores on the test set because the ground truth was established using keyword pattern matching. Semantic search demonstrated strong performance (0.85 NDCG), successfully retrieving relevant documents even when exact keywords were missing.