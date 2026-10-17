# Bank Statement Analyzer

A modular application for parsing, standardizing, categorizing, and visualizing bank statements from CSV, XLS, XLSX and PDF files. Supports Armenian, English and Russian bank exports, with a hybrid **rules + machine learning** categorization pipeline that learns from your own statements and corrections.

## Key Design Patterns

The project follows SOLID principles and implements these design patterns:

| Pattern                                   | Where                                                                                 | Purpose                                                                                                                                             |
| ----------------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Facade**                          | `BankAnalyzerFacade` (`facade.py`)                                                | One high-level entry point (load → classify → learn → analyze) for the UI layer (Streamlit)                                                      |
| **Chain of Responsibility**         | `ClassificationHandler` (`classification_handlers.py`)                            | Sequential categorization: `MerchantRuleHandler` → `KeywordRuleHandler` → `LearnedModelHandler` → `VectorSimilarityHandler` → `Other` |
| **Strategy**                        | `StatementReader`, `ExportStrategy`, `AnomalyDetector`                          | Interchangeable algorithms for file reading (PDF / Excel / CSV), exporting (CSV / Excel) and anomaly detection (Z-score)                            |
| **Factory**                         | `StatementReaderFactory`, `ExportStrategyFactory`, `build_classification_chain` | Picks the right reader / exporter from the file or UI choice and assembles the classification chain                                                 |
| **Template Method**                 | `BankStatementLoader.load`                                                          | Fixed pipeline: detect bank → read file → standardize schema                                                                                      |
| **Dependency Injection / Protocol** | `StatementLoaderProtocol`, `ClassifierProtocol`                                   | High-level code depends on interfaces, so the loader, classifier and ML model can be swapped or mocked in tests                                     |

## Project Structure

```
.
├── bank_analyzer/
│   ├── config/
│   │   ├── bank_configs.json            # Column mapping per bank
│   │   └── category_rules.json          # Categories → keywords + merchants (with aliases)
│   ├── data/
│   │   ├── training_data.csv            # Created at runtime: what the ML model has learned
│   │   ├── labelled_all.xlsx            # Hand-checked labels (849 real transactions) for evaluation
│   │   └── fake_statement_ML_demo.csv   # Demo statement with merchants unknown to the rules
│   ├── modules/
│   │   ├── readers.py                   # File-reading strategies (PDF, Excel, CSV) + table/header detection
│   │   ├── ingestion.py                 # Bank detection, column mapping, schema standardization
│   │   ├── text_utils.py                # Shared text cleaning / ML noise-token removal
│   │   ├── classifier.py                # TransactionClassifier: rules loading, chain, learning, overrides
│   │   ├── classification_handlers.py   # Chain of Responsibility handlers + rule index + fuzzy matching
│   │   ├── ml_model.py                  # LearnedCategoryModel: TF-IDF + Logistic Regression
│   │   ├── analytics.py                 # KPIs, trends, budgets (AnalyticsService)
│   │   ├── anomaly.py                   # Anomaly detection strategies (Z-score)
│   │   ├── exporters.py                 # Export strategies (CSV, Excel)
│   │   ├── facade.py                    # BankAnalyzerFacade
│   │   ├── models.py                    # Enums, dataclasses, protocols
│   │   └── constants.py                 # Default budgets, income categories
│   ├── app.py                           # Streamlit web dashboard
│   ├── qt_app.py                        # Legacy PySide6 desktop GUI (kept for reference, not maintained)
│   └── evaluate_classifier.py           # Accuracy evaluation + threshold tuning
├── requirements.txt
├── README.md
├── test_*.csv                           # Generated CSV statements
├── generate_statement_en.py             # Statement generator (English)
├── generate_statement_am.py             # Statement generator (Armenian)
├── Real_Statements/                     # Real statements for testing (in .gitignore for privacy)
└── UML/                                 # Project UML diagrams
```

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt` (pandas, scikit-learn, scipy, pdfplumber, openpyxl, streamlit, plotly)

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

| Section                        | Description                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------------------- |
| **Sidebar**              | Statement upload (CSV / XLS / XLSX / PDF), bank selector, date range filter                   |
| **Metrics**              | Total income, expenses, net savings, transaction count                                        |
| **Visual Analytics**     | Donut chart, monthly trends, top 5 expenses                                                   |
| **Transaction Explorer** | Filterable table with `confidence` and `method` columns, export, manual category override |
| **Budget & Anomalies**   | Budget progress bars, Z-score anomaly detection                                               |

## Quick Test with Sample Data

1. Run `streamlit run app.py`
2. Upload a few real statements first (the model learns from them)
3. Upload `data/fake_statement_ML_demo.csv`. Its merchants are not in the rules, so many rows are decided by the ML model (`method = ML model`)
4. Correct a wrong row with **Manual Category Override**, then reload the file: the correction is remembered

## Supported Banks and Formats

Tested on real statements:

| Bank                  | Formats     | Notes                                                                                                                                                |
| --------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ameriabank**  | CSV, XLSX   | CSV payee column (`Շահառու/Վճարող`) is used; amounts are taken in the account currency, so USD card purchases are converted correctly |
| **ACBA Bank**   | XLSX        | The merchant is read from `Գործարքի վայրը` (ACBA's description is only `Վճարում`)                                            |
| **Idram**       | XLSX        | Debit / credit columns                                                                                                                               |
| **Evocabank**   | XLSX        |                                                                                                                                                      |
| **Inecobank**   | config only |                                                                                                                                                      |
| **Auto Detect** | any         | Heuristic header and column matching                                                                                                                 |

Add new banks by extending `config/bank_configs.json`.

## Categorization Logic

Each transaction goes through the chain until one step is confident:

1. **Merchant rule:** known merchant alias from `category_rules.json` (longest match wins), e.g. `ART LUNCH` → Cafes and Restaurants. Confidence 1.0.
2. **Keyword rule:** category keyword (Armenian / English / Russian), e.g. `ուսման վարձ` → Education. Confidence 0.95.
   The bank's own description is checked first and the payee/place second, so `Commission | ՀԱՊՀ` is a Fee, not tuition.
3. **ML model** (`ml_model.py`), used only when no rule matches:
   - **Features:** TF-IDF on character n-grams (2–5, robust to typos and transliteration) + TF-IDF on word uni/bi-grams, after removing bank boilerplate (`POS`, `YEREVAN`, `Ամերիաբանկ ՓԲԸ`, dates, card numbers…)
   - **Classifier:** multinomial **Logistic Regression** (scikit-learn), `class_weight="balanced"`, `C=8`
   - **Training data:** the JSON rules (cold start) + rows the rules labelled in your uploaded statements (self-training) + manual overrides (highest weight, ×6)
   - **Answers only when confident:** probability ≥ 0.25 **and** a lead of ≥ 0.20 over the second-best class; otherwise the row falls through
   - Retrains automatically after every upload and every override; saved in `data/training_data.csv`
4. **Fuzzy match:** char n-gram TF-IDF shortlist + word-by-word similarity check on short spans of the description; catches typos such as `STARBUKS` → Starbucks.
5. **Other:** nothing was confident enough.

Expense rows can never be classified as `Income` or `Salary`.

**Feedback loop:** a manual override adds a keyword to `category_rules.json`, re-labels identical rows in the current statement, and retrains the model, so the correction applies to future statements too.

## Evaluation

```bash
python evaluate_classifier.py data/labelled_all.xlsx --sweep
```

It uses cross-validation with folds grouped by merchant, so the same merchant never appears in both train and test (statements overlap). It reports:

- accuracy per method (rule / ML / fuzzy / Other) and sample mistakes
- an **unseen-merchants test**, where all rules matching the test rows are deleted, so the ML model is measured alone
- a **threshold sweep** for `min_probability` / `min_margin`

Results on 849 real transactions (Ameriabank, ACBA, Idram):

| Test                                             | Accuracy | Other | Wrong guesses  |
| ------------------------------------------------ | -------- | ----- | -------------- |
| Full pipeline                                    | 98.9%    | 0.7%  | 0.4%           |
| Unseen merchants, old thresholds (0.40 / 0.10)   | 62.6%    | 17.0% | 20.4%          |
| Unseen merchants, tuned thresholds (0.25 / 0.20) | 63.4%    | 32.7% | **3.9%** |

The tuned thresholds keep accuracy the same but cut wrong guesses about five-fold: the model says "Other" instead of guessing, and a wrong category would silently distort budgets.

## Unified Transaction Schema

| Column                  | Type     | Description                                                                                 |
| ----------------------- | -------- | ------------------------------------------------------------------------------------------- |
| `date`                | datetime | Transaction date                                                                            |
| `description`         | str      | Bank text; merchant/payee appended after `\|` when the bank stores it in a separate column |
| `cleaned_description` | str      | Sanitized text (used for manual-override keywords)                                          |
| `amount`              | float    | Absolute value in the account currency                                                      |
| `transaction_type`    | str      | `income` or `expense`                                                                   |
| `transaction_id`      | str      | Row identifier                                                                              |
| `category`            | str      | Assigned category                                                                           |
| `confidence`          | float    | Classification confidence (0–1)                                                            |
| `method`              | str      | `Merchant rule`, `Keyword rule`, `ML model`, `Vector`, `manual`, or `None`      |

## Extending Category Rules

Edit `config/category_rules.json`:

```json
{
  "Transport": {
    "keywords": ["taxi", "տաքսի", "такси"],
    "merchants": [
      { "name": "Yandex Go", "aliases": ["yandex go", "yandexgo", "յանդեքս"] }
    ]
  }
}
```

The old flat format (`"Transport": ["taxi", "yandex"]`) is still accepted. You can also use **Manual Category Override** in the UI: new keywords are saved automatically and the model retrains.

## Web Page

Visit [this link](https://statementanalyser-vahagmardyan.streamlit.app/) to use the application from any other device.
