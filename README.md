# Bank Statement Analyzer

A modular, production-ready application for parsing, standardizing, categorizing, and visualizing bank statements from CSV, XLS and XLSX files. Supports Armenian and English bank exports with hybrid rule-based + ML categorization.

## Project Structure

```
.
  bank_analyzer/
│  ├── config/
│  │   ├── bank_configs.json       # Column mapping per bank
│  │   └── category_rules.json     # Keyword → category rules
│  ├── data/
│  │   ├── seed_transactions.csv   # ML training seed data
│  │   ├── sample_statement.csv    # Demo CSV for testing
│  │   └── feedback_overrides.json # Created at runtime (manual overrides)
│  ├── modules/
│  │   ├── ingestion.py            # CSV parsing & standardization
│  │   ├── classifier.py           # Hybrid categorization
│  │   └── analytics.py            # KPIs, trends, anomalies, budgets
│  ├── app.py                      # Streamlit web dashboard
│  ├── qt_app.py                   # PySide6 desktop GUI
│
└── requirements.txt
└── README.md
└── test_*.csv                     # Generated CSV Stataments
└── generate_statement_en.py      # Statement Generator (English)
└── generate_statement_am.py      # Statement Generator (Armenian)
└── Real_Statements                # Real Statements for testing (in .gitignore to keep privacy)
```

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Installation

```bash
cd bank_analyzer
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the Streamlit Dashboard

```bash
cd bank_analyzer
streamlit run app.py
```

The app opens in your browser (default: http://localhost:8501).

### Streamlit Features

| Section                        | Description                                            |
| ------------------------------ | ------------------------------------------------------ |
| **Sidebar**              | CSV upload, bank selector, date range filter           |
| **Metrics**              | Total income, expenses, net savings, transaction count |
| **Visual Analytics**     | Donut chart, monthly trends, top 5 expenses            |
| **Transaction Explorer** | Filterable table + manual category override            |
| **Budget & Anomalies**   | Budget progress bars, z-score anomaly detection        |

## Running the PySide6 Desktop GUI

```bash
cd bank_analyzer
python qt_app.py
```

The desktop app mirrors the Streamlit functionality with native Qt charts and tables.

## Quick Test with Sample Data

1. Launch either `streamlit run app.py` or `python qt_app.py`
2. Upload `data/sample_statement.csv`
3. Select **Auto Detect** or any configured bank
4. Explore categorized transactions and charts

## Supported Banks (Config)

Pre-configured mappings in `config/bank_configs.json`:

- **Ameriabank** — comma-separated, UTF-8
- **ACBA Bank** — semicolon-separated, skip header row
- **Inecobank** — UTF-8 BOM, dot-date format
- **Auto Detect** — heuristic column matching

Add new banks by extending `bank_configs.json`.

## Categorization Logic

1. **Tier 1 — Rules:** Keyword match against `config/category_rules.json` (high confidence)
2. **Tier 2 — ML:** TF-IDF + Logistic Regression trained on `data/seed_transactions.csv`
3. **Feedback loop:** Manual overrides in the UI update rules and retrain the model

## Unified Transaction Schema

| Column                  | Type     | Description                                  |
| ----------------------- | -------- | -------------------------------------------- |
| `date`                | datetime | Transaction date                             |
| `description`         | str      | Raw bank text                                |
| `cleaned_description` | str      | Sanitized text for matching                  |
| `amount`              | float    | Absolute value                               |
| `transaction_type`    | str      | `income` or `expense`                    |
| `category`            | str      | Assigned category                            |
| `confidence`          | float    | Classification confidence                    |
| `method`              | str      | `rule`, `ml`, `manual`, or `default` |

## Extending Category Rules

Edit `config/category_rules.json`:

```json
{
  "Transport": ["yandex", "gg", "bolt"],
  "Supermarket": ["city", "sas", "carrefour"]
}
```

Or use the **Manual Category Override** feature in either UI — new keywords are persisted automatically.


# Web-Page

Visit [this link](https://statementanalyser-vahagmardyan.streamlit.app/) for using this application from any other device.
