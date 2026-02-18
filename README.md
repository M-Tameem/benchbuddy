# BenchBuddy

**AI-powered team composition tool** - matches employees to projects using NLP and ML on Slack data.

Built as a prototype for BenchSci. Research from Google, Apple, and Microsoft shows personality chemistry and work ethic compatibility can increase team productivity by ~70%. BenchBuddy operationalizes that insight for project managers.

---

## How It Works

```
Slack API / Sample Data
        ↓
Employee messages + introductions
        ↓
BERT embeddings → cosine similarity vs. project description → candidate shortlist
        ↓
DistilBERT sentiment analysis → trait scores (participation, work done, compatibility, adaptability)
        ↓
KMeans clustering → select team closest to global trait average
        ↓
Recommended team
```

1. **Candidate filtering** - BERT embeds each employee's intro message and computes cosine similarity against the project description. Filters anyone below 0.3 threshold.
2. **Trait scoring** - DistilBERT analyzes chat history and maps sentiment to four traits on a 1–9 scale.
3. **Team selection** - KMeans clusters candidates by trait profile; selects the cluster(s) closest to the global average, then picks by Euclidean distance to centroid if trimming is needed.

---

## Stack

| Layer | Tools |
|---|---|
| NLP | `sentence-transformers` (BERT), `transformers` (DistilBERT), `textblob` |
| ML | `scikit-learn` (KMeans, MinMaxScaler), `scipy` (Euclidean distance) |
| Data | `pandas`, Slack API (`slack-sdk`) |
| Frontend | `streamlit` |

---

## Running Locally

**With pip:**
```bash
pip install -r requirements.txt
streamlit run benchbuddy.py
```

**With Docker:**
```bash
docker compose up
```
Then open [http://localhost:8501](http://localhost:8501).

> **No Slack token?** Use **Refresh Data → Load Fake Data** in the sidebar to populate with sample employee data and skip directly to the analysis steps.

---

## Project Structure

```
benchbuddy/
├── benchbuddy.py           # Streamlit app + team recommendation engine
├── project_intro_sim.py    # BERT/DistilBERT NLP pipeline
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── sample_introductions.csv   # Sample employee intro messages
│   ├── sample_messages.csv        # Sample Slack chat history
│   ├── possible_teammates.csv     # Example: candidates after BERT filtering
│   └── output_trait_scores.csv    # Example: DistilBERT trait scores
└── assets/
    └── BenchSci_Slide_Deck.pdf    # Original pitch deck
```

---

## Slide Deck

The original pitch deck is in [`assets/BenchSci_Slide_Deck.pdf`](assets/BenchSci_Slide_Deck.pdf).
