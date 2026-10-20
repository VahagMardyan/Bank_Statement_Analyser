# Bank Statement Analyzer

A modular application for parsing, standardizing, categorizing, and visualizing bank statements from CSV, XLS, XLSX and PDF files. It supports Armenian, English and Russian bank exports, with a hybrid **rules + machine learning** categorization pipeline that learns from your own statements and corrections. The dashboard is fully multilingual (Armenian and English included; more languages are added by dropping in a JSON file).

## Key Design Patterns

The project follows SOLID principles and implements these design patterns:

| Pattern                                   | Where                                                                                                 | Purpose                                                                                                                                                  |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Facade**                          | `BankAnalyzerFacade` (`facade.py`)                                                                | One entry point (load → classify → learn → analyze) for the UI. The UI never reaches into the subsystems.                                             |
| **Template Method**                 | `StatementLoaderTemplate.load` (`ingestion.py`), `ExportStrategy.export` (`exporters.py`)     | Fixed pipeline: read file → standardize → drop invalid rows → finalize. Subclasses supply the steps.                                                  |
| **Strategy**                        | `StatementReader`, `PdfLayoutStrategy`, `ExportStrategy`, `AnomalyDetector`                   | Interchangeable algorithms for file formats (PDF / Excel / CSV), PDF layouts (borderless / table), exports (CSV / Excel) and anomaly detection (Z-score) |
| **Factory**                         | `StatementReaderFactory`, `ExportStrategyFactory`                                                 | Picks the right reader for a file and the right exporter for a UI choice                                                                                 |
| **Registry (decorator)**            | `@ExportStrategyFactory.register("CSV")`                                                            | A new export format is one new class; the factory is never edited (Open/Closed)                                                                          |
| **Chain of Responsibility**         | `ClassificationHandler` (`classification_handlers.py`)                                            | Primary description → merchant → keyword → individual entrepreneur → ML model → fuzzy match → `Other`                                            |
| **Builder**                         | `ClassificationChainBuilder`                                                                        | Assembles the chain in one readable place; optional steps (the ML model) are skipped with `None`                                                       |
| **Observer**                        | `RuleSet.subscribe` → `RuleMatcher`, `FuzzyRuleIndex`, `LearnedCategoryModel` (`rules.py`) | When an override adds a rule, every index and the ML model rebuild automatically                                                                         |
| **Repository**                      | `JsonRuleRepository` / `InMemoryRuleRepository`, `TrainingExampleStore`, `LocaleRepository`   | Storage (JSON / CSV files) is hidden behind `load` / `save`; evaluation runs on an in-memory copy and never touches your files                       |
| **Parameter Object**                | `ClassificationRequest` (`models.py`)                                                             | Raw text, cleaned text, bank description and excluded categories travel together; text is cleaned once per row                                           |
| **Value Object**                    | `ClassificationResult`, `KPIMetrics`, `ParsedAmounts`, `Prediction` (frozen dataclasses)      | Immutable results                                                                                                                                        |
| **Dependency Injection / Protocol** | `StatementLoaderProtocol`, `ClassifierProtocol`, `SelfTrainingClassifier`, `Translator`       | High-level code depends on interfaces, so the loader, classifier, model and language can be swapped or mocked in tests                                   |

How SOLID shows up:

- **Single Responsibility:** e.g. `TransactionClassifier` only wires collaborators together. Rule storage, matching, fuzzy search and ML each have their own class, and so do column resolution, amount parsing and date parsing in the loader.
- **Open/Closed:** new readers, PDF layouts, exporters, anomaly detectors, handlers, dashboard tabs and languages are all added without editing existing classes.
- **Interface Segregation:** `ClassifierProtocol` is the minimum the UI needs. Self-training is a separate optional protocol (`SelfTrainingClassifier`).
- **Dependency Inversion:** the facade depends on protocols, not concrete classes.

UML class diagrams are in `UML/`:

| File                              | Shows                                                               |
| --------------------------------- | ------------------------------------------------------------------- |
| `01_ingestion_system.mmd`       | Loader template, column/amount/date parsing, readers, PDF layouts   |
| `02_classification_system.mmd`  | Classifier, rules + repository + observers, handler chain, ML model |
| `03_analytics_and_exporter.mmd` | Analytics service, anomaly strategy, export strategies + registry   |
| `04_high_level_facade.mmd`      | Facade, protocols, enums                                            |
| `05_ui_and_i18n.mmd`            | Dashboard tabs, translator, locale files                            |

## Project Structure

```
.
├── bank_analyzer/
│   ├── config/
│   │   └── category_rules.json          # Categories → keywords + merchants (with aliases)
│   ├── locales/
│   │   ├── en.json                      # All UI text in English (reference file)
│   │   └── hy.json                      # All UI text in Armenian
│   ├── data/
│   │   ├── training_data.csv            # Created at runtime: what the ML model has learned
│   │   ├── labelled_all.xlsx            # Hand-checked labels (849 real transactions) for evaluation
│   │   └── fake_statement_ML_demo.csv   # Demo statement with merchants unknown to the rules
│   ├── modules/
│   │   ├── readers.py                   # Reader strategies (PDF, Excel, CSV), PDF layout strategies, header detection
│   │   ├── ingestion.py                 # Loader template, column/amount/date parsing (built-in header hints)
│   │   ├── text_utils.py                # TextNormalizer: the one text cleaner used everywhere
│   │   ├── rules.py                     # RuleSet (observable), CategoryRule, rule repositories
│   │   ├── classification_handlers.py   # Chain of Responsibility handlers, chain builder, exact + fuzzy rule indexes
│   │   ├── classifier.py                # TransactionClassifier: wires rules, indexes, model and chain
│   │   ├── ml_model.py                  # LearnedCategoryModel (TF-IDF + Logistic Regression), training store
│   │   ├── analytics.py                 # AnalyticsService: KPIs, trends, budgets, anomalies
│   │   ├── anomaly.py                   # Anomaly detection strategies (Z-score)
│   │   ├── exporters.py                 # Export strategies (CSV, Excel) + registry factory
│   │   ├── facade.py                    # BankAnalyzerFacade
│   │   ├── i18n.py                      # Translator, LocaleRepository, missing-translation checker
│   │   ├── models.py                    # Enums, dataclasses, protocols
│   │   └── constants.py                 # Default budgets, income categories
│   ├── app.py                           # Streamlit dashboard (DashboardApp + one class per tab)
│   └── evaluate_classifier.py           # Accuracy evaluation + threshold tuning
├── requirements.txt
├── README.md
├── test_*.csv                           # Generated CSV statements
├── generate_statement_en.py             # Statement generator (English)
├── generate_statement_am.py             # Statement generator (Armenian)
├── Real_Statements/                     # Real statements for testing (in .gitignore for privacy)
└── UML/                                 # Project UML diagrams (Mermaid)
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
| **Sidebar**              | Language switch, statement upload (CSV / XLS / XLSX / PDF), date range filter                 |
| **Metrics**              | Total income, expenses, net savings, transaction count                                        |
| **Visual Analytics**     | Donut chart, monthly trends, top 5 expenses                                                   |
| **Transaction Explorer** | Filterable table with `confidence` and `method` columns, export, manual category override |
| **Budget & Anomalies**   | Budget progress bars, Z-score anomaly detection                                               |

## Languages

Everything on screen comes from `locales/<code>.json`: labels, messages, table headers, category names, transaction types (Մուտք / Ելք), method names, bank names and the currency word. The data itself always stays in English (category keys, `income` / `expense`), so analytics, budgets and the ML model never depend on the language.

| Task                   | How                                                                                                                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Add a language         | Copy `locales/en.json` to `locales/<code>.json` (e.g. `ru.json`), set `"meta": {"name": "Русский"}`, translate the values. It appears in the language switch automatically. |
| Edit text              | Change the value in that language's file. No code changes are needed.                                                                                                                      |
| Remove a language      | Delete its file.                                                                                                                                                                           |
| Find untranslated text | `python -m modules.i18n` lists missing and unknown keys per language                                                                                                                     |
| Default language       | `DEFAULT_LANGUAGE` in `modules/i18n.py` (Armenian)                                                                                                                                     |

Anything missing in a language falls back to English, then to the key itself, so a half-finished translation never crashes the app. Text placeholders such as `{count}` or `{currency}` must be kept as they are.

Exports use the on-screen language. `evaluate_classifier.py` maps headers and values from any language back to the English keys, so an Armenian export can be used as a labelled file directly.

Not translatable from the locale files: Streamlit's built-in upload-box text and the date-picker calendar, and error details from the file loader.

## Quick Test with Sample Data

1. Run `streamlit run app.py`
2. Upload a few real statements first (the model learns from them)
3. Upload `data/fake_statement_ML_demo.csv`. Its merchants are not in the rules, so many rows are decided by the ML model (`method = ML model`)
4. Correct a wrong row with **Manual Category Override**, then reload the file: the correction is remembered

## Supported Banks and Formats

Tested on real statements:

| Bank                  | Formats   | Notes                                                                                                                                                |
| --------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ameriabank**  | CSV, XLSX | CSV payee column (`Շահառու/Վճարող`) is used; amounts are taken in the account currency, so USD card purchases are converted correctly |
| **ACBA Bank**   | XLSX      | The merchant is read from `Գործարքի վայրը` (ACBA's description is only `Վճարում`)                                            |
| **Idram**       | XLSX      | Debit / credit columns                                                                                                                               |
| **Evocabank**   | XLSX      |                                                                                                                                                      |
| **Other banks** | any       | Not tested, but any statement whose headers match the built-in hints is read                                                                         |

There is no per-bank configuration and no bank selection. Every file goes through the same steps:

- **Header row:** found by date and amount words.
- **Columns:** found by the Armenian and English header hints in `COLUMN_HINTS` (`modules/ingestion.py`). A new header wording is supported by adding the word to that list.
- **Dates:** parsed so ISO dates (`2026-03-04`) and day-first dates (`04.03.2026`) are never mixed up.

## Categorization Logic

Each transaction goes through the chain until one step is confident:

1. **Bank description first:** the loader appends the merchant/payee after `|`. The merchant and keyword rules are tried on the bank's own description before the payee, so `Commission | ՀԱՊՀ` is a Fee, not tuition.
2. **Merchant rule:** known merchant alias from `category_rules.json` (longest match wins), e.g. `ART LUNCH` → Cafes and Restaurants. Confidence 1.0.
3. **Keyword rule:** category keyword (Armenian / English / Russian), e.g. `ուսման վարձ` → Education. Confidence 0.95.
4. **Individual entrepreneur detector:** in Armenia many small shops, salons and kiosks are registered as individual entrepreneurs (ԱՁ / IE / PE), so the statement shows only the owner's name, e.g. `Վճարում | K. A. MELKONYAN` or `Ք: ARTAK BEGYAN IE`. A card payment is put into **Individual Entrepreneurs** if:

   - it carries an ԱՁ/IE/PE/ИП marker, or
   - the payee is 2–4 name-like words with an Armenian surname ending (`-yan`, `-ian`, `-uni`, `-յան`…) and no business words (`market`, `pharm`, `club`…).

   Personal transfers are not affected, because the Transfer rule fires first. Confidence 0.7. On the real data it caught 164 rows with 0 false positives.
5. **ML model** (`ml_model.py`), used only when nothing above matches:

   - **Features:** TF-IDF on character n-grams (2–5, robust to typos and transliteration) + TF-IDF on word uni/bi-grams, after removing bank boilerplate (`POS`, `YEREVAN`, `Ամերիաբանկ ՓԲԸ`, dates, card numbers…)
   - **Classifier:** multinomial **Logistic Regression** (scikit-learn), `class_weight="balanced"`, `C=8`
   - **Training data:** the JSON rules (cold start) + rows the rules labelled in your uploaded statements (self-training) + manual overrides (highest weight, ×6)
   - **Answers only when confident:** probability ≥ 0.25 **and** a lead of ≥ 0.20 over the second-best class; otherwise the row falls through
   - **Retraining:** automatic after every upload and every override; the training data is saved in `data/training_data.csv`
6. **Fuzzy match:** char n-gram TF-IDF shortlist + word-by-word similarity check on short spans of the description; catches typos such as `STARBUKS` → Starbucks.
7. **Other:** nothing was confident enough.

Expense rows can never be classified as `Income` or `Salary`.

**Feedback loop:** a manual override does three things:

- adds a keyword to `category_rules.json`
- re-labels identical rows in the current statement
- retrains the model

The rule change notifies the matcher, the fuzzy index and the model, so the correction applies to future statements too.

## Evaluation

```bash
python evaluate_classifier.py data/labelled_all.xlsx --sweep
```

It uses cross-validation with folds grouped by merchant, so the same merchant never appears in both train and test (statements overlap). It reports:

- accuracy per method (rule / ML / fuzzy / Other) and sample mistakes
- an **unseen-merchants test**, where all rules matching the test rows are deleted, so the ML model is measured alone
- a **threshold sweep** for `min_probability` / `min_margin`

The labelled file can be an export from the Transaction Explorer in any UI language.

Results on 849 real transactions (Ameriabank, ACBA, Idram):

| Test                                             | Accuracy | Other | Wrong guesses  |
| ------------------------------------------------ | -------- | ----- | -------------- |
| Full pipeline                                    | 98.9%    | 0.7%  | 0.4%           |
| Unseen merchants, old thresholds (0.40 / 0.10)   | 62.6%    | 17.0% | 20.4%          |
| Unseen merchants, tuned thresholds (0.25 / 0.20) | 63.4%    | 32.7% | **3.9%** |

The tuned thresholds keep accuracy the same but cut wrong guesses about five-fold: the model says "Other" instead of guessing, and a wrong category would silently distort budgets.

## Unified Transaction Schema

| Column                  | Type     | Description                                                                                                                                                 |
| ----------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `date`                | datetime | Transaction date                                                                                                                                            |
| `description`         | str      | Bank text; merchant/payee appended after `\|` when the bank stores it in a separate column                                                                 |
| `cleaned_description` | str      | Same text after `TextNormalizer.clean` (lower-case; dates, times, card numbers, IDs and punctuation removed). Manual-override keywords are saved from it. |
| `amount`              | float    | Absolute value in the account currency                                                                                                                      |
| `transaction_type`    | str      | `income` or `expense`                                                                                                                                   |
| `transaction_id`      | str      | Row identifier                                                                                                                                              |
| `category`            | str      | Assigned category key (English; shown translated in the UI)                                                                                                 |
| `confidence`          | float    | Classification confidence (0–1)                                                                                                                            |
| `method`              | str      | `Merchant rule`, `Keyword rule`, `Name heuristic`, `ML model`, `Vector`, `manual`, or `None`                                                  |

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

The key (`"Transport"`) is the category's identifier; keep it in English. To show it under another name, add it to the `categories` section of each `locales/<code>.json`. A new category with no translation is shown by its key.

Older formats are still accepted:

- the flat list form: `"Transport": ["taxi", "yandex"]`
- keys with an inline label: `"Transport | Տրանսպորտ"` (only the part before `|` is used)

You can also use **Manual Category Override** in the UI: new keywords are saved automatically and the model retrains.

## Web Page

Visit [this link](https://statementanalyser-vahagmardyan.streamlit.app/) to use the application from any other device.
