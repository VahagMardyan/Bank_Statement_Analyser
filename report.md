# Design Pattern Refactor Report

This document describes the refactor of **Bank Statement Analyzer**: what changed, why, and which design patterns were applied. Runtime behavior of parsing, hybrid categorization, charts, budgets, and export is intended to stay the same; the code is structured so each concern has a clear home.

## Why the original code needed patterns 

The application already worked, but responsibilities were mixed:

- `BankStatementLoader` was a large class that detected file type, read CSV/Excel/PDF, found table headers, mapped bank columns, and standardized the schema.
- `TransactionClassifier` encoded merchant, keyword, and TF-IDF matching as private methods with a hard-coded `if` sequence.
- Streamlit (`app.py`) and Qt (`qt_app.py`) each constructed the loader and classifier, duplicated override logic, and reached into private fields such as `loader._configs`.
- CSV vs Excel download logic lived as two functions in the Streamlit app.
- Analytics lived as loose functions with no extension point for other anomaly algorithms.

The refactor introduces small typed objects and classic GoF-style patterns so new banks, file formats, classifiers, exporters, or detectors can be added without rewriting the UIs.

## New / changed files

| File                                                 | Role                                                  |
| ---------------------------------------------------- | ----------------------------------------------------- |
| `bank_analyzer/modules/models.py`                  | Typed domain objects and Protocols                    |
| `bank_analyzer/modules/constants.py`               | Shared default budgets                                |
| `bank_analyzer/modules/readers.py`                 | File-read Strategy + Factory                          |
| `bank_analyzer/modules/classification_handlers.py` | Classification Chain of Responsibility                |
| `bank_analyzer/modules/anomaly.py`                 | Anomaly-detection Strategy                            |
| `bank_analyzer/modules/exporters.py`               | Export Strategy + Factory                             |
| `bank_analyzer/modules/facade.py`                  | Facade + dependency injection for both UIs            |
| `bank_analyzer/modules/ingestion.py`               | Slimmed loader: bank detect + column mapping + schema |
| `bank_analyzer/modules/classifier.py`              | Orchestrates the classification chain                 |
| `bank_analyzer/modules/analytics.py`               | Analytics service; budget-loop bug fix                |
| `bank_analyzer/app.py` / `qt_app.py`             | Talk to the facade instead of internals               |

Public names (`BankStatementLoader`, `TransactionClassifier`, `compute_kpis`, etc.) remain importable from `modules`.

---

## Design patterns used

### 1. Strategy — file readers

**Problem:** CSV, Excel, and PDF each need a different read path, but `load()` should not branch on format forever.

**Where:** `modules/readers.py`

- Abstract `StatementReader` with `can_read()` / `read()`.
- Concrete strategies: `CsvStatementReader`, `ExcelStatementReader`, `PdfStatementReader`.
- Shared header discovery lives in `TableExtractor` so readers do not duplicate it.

Adding a new format (for example OFX) means a new reader class, not a larger `if` in the loader.

### 2. Factory Method — choosing a reader and an exporter

**Problem:** Callers should not construct the right strategy themselves.

**Where:**

- `StatementReaderFactory.create(path)` walks PDF → Excel → CSV (`can_read`) and returns the matching reader.
- `ExportStrategyFactory.create(label)` maps UI labels (`"CSV"`, `"Excel (.xlsx)"`) to `CsvExportStrategy` / `ExcelExportStrategy`.

`BankStatementLoader.load()` only does `factory.create(path).read(...)`.

### 3. Template Method — standardize after read

**Problem:** Every file still needs the same pipeline: resolve bank → read table → map columns → dates/amounts → clean description → drop balance rows → assign IDs.

**Where:** `BankStatementLoader.load()` then `_standardize()`.

The variable step is *how* the raw table is obtained (Strategy). The fixed steps stay on the loader.

### 4. Chain of Responsibility — hybrid classification

**Problem:** Matching order was implicit in nested `if`s: merchant, then keyword, then TF-IDF.

**Where:** `modules/classification_handlers.py`

- `ClassificationHandler.handle()` tries `try_classify()`; on `None` it forwards to the next handler.
- Chain: `MerchantRuleHandler` → `KeywordRuleHandler` → `VectorSimilarityHandler`.
- `build_classification_chain()` assembles the chain (small factory for the pipeline).
- Shared rules and the TF-IDF index live in `RuleIndex`.

A new matcher (for example a neural model) can be inserted as another handler without changing the UI.

### 5. Strategy — anomaly detection

**Problem:** Z-score logic was baked into `detect_anomalies()`.

**Where:** `modules/anomaly.py`

- Abstract `AnomalyDetector`.
- `ZScoreAnomalyDetector` is the default.
- `detect_anomalies(..., detector=...)` accepts another strategy later (IQR, isolation forest, etc.).

### 6. Strategy — table export

**Problem:** Streamlit had two parallel `convert_df_to_*_bytes` functions.

**Where:** `modules/exporters.py` — `ExportStrategy.export()`, used by the Transaction Explorer download button.

### 7. Facade — one API for both GUIs

**Problem:** Streamlit and Qt duplicated “load then classify”, category override, bank labels, and KPI wiring.

**Where:** `BankAnalyzerFacade`

- `load_and_classify()`
- `override_category()`
- `apply_date_filter()`
- `bank_labels()` / `get_categories()` / `compute_kpis()`

Both `app.py` and `qt_app.py` hold one facade instance instead of a loader + classifier pair. They no longer read `loader._configs`.

### 8. Dependency Inversion (and injection)

**Problem:** High-level UI code imported concrete classes and private attributes.

**Where:** `ClassifierProtocol` and `StatementLoaderProtocol` in `models.py`.

`BankAnalyzerFacade.__init__` accepts optional loader, classifier, and `AnalyticsService` (defaults constructed inside). Tests or a future API can inject fakes.

### 9. Service object — analytics

**Where:** `AnalyticsService` wraps category summary, trends, budgets, anomalies, and KPIs so the UIs depend on one object. Module-level functions remain for backward compatibility.

---

## SOLID principles

The patterns above were used to apply SOLID, not as decoration. Mapping of each principle to this codebase:

### S — Single Responsibility Principle

Each module/class has one main reason to change:

| Unit | Responsibility |
|------|----------------|
| `StatementReader` implementations | Read one file format into a table |
| `TableExtractor` | Find headers in a raw grid |
| `BankStatementLoader` | Bank detection, column mapping, unified schema |
| `ClassificationHandler` subclasses | One matching tactic each (merchant / keyword / vector) |
| `RuleIndex` | Rule corpus + TF-IDF index |
| `TransactionClassifier` | Load/save rules and run the chain |
| `ExportStrategy` implementations | One download format |
| `AnomalyDetector` implementations | One anomaly algorithm |
| `AnalyticsService` | KPI / trend / budget queries |
| `BankAnalyzerFacade` | Orchestrate load → classify → override for UIs |
| `app.py` / `qt_app.py` | Presentation only |

Previously, ingestion, classification, export, and UI wiring sat in the same few files.

### O — Open/Closed Principle

Core pipelines stay closed for modification and open for extension:

- New file format → new `StatementReader` (do not grow `load()` with more `if` branches).
- New classifier stage → new `ClassificationHandler` linked in `build_classification_chain()`.
- New export format → new `ExportStrategy` registered in `ExportStrategyFactory`.
- New anomaly method → new `AnomalyDetector` passed into `detect_anomalies()`.

`BankStatementLoader.load()` and both GUIs do not need to change for those extensions.

### L — Liskov Substitution Principle

Subtypes are usable wherever the abstraction is expected, with the same contract:

- Any `StatementReader` must implement `can_read(path)` and `read(path) → DataFrame`.
- Any `ExportStrategy` must provide `export()`, `file_extension`, and `mime_type`.
- Any `AnomalyDetector.detect(expenses)` returns a DataFrame of flagged rows (empty if none).
- Any `ClassificationHandler.handle(text)` returns a `ClassificationResult`.
- `BankAnalyzerFacade` accepts `ClassifierProtocol` / `StatementLoaderProtocol`; a stub that satisfies the protocol can replace the real classes.

Callers never special-case “if this is the PDF reader…” after the factory has chosen a strategy.

### I — Interface Segregation Principle

Clients depend on small surfaces, not on one mega-class:

- `StatementReader` only has `can_read` / `read` (UIs never see PDF word-position helpers).
- `ExportStrategy` only has export metadata + `export()`.
- `AnomalyDetector` only has `detect()`.
- `ClassifierProtocol` is `get_categories`, `classify`, `manual_override`.
- `StatementLoaderProtocol` is `list_banks`, `bank_labels`, `load`.

Streamlit/Qt talk to `BankAnalyzerFacade` and do not import reader or handler internals.

### D — Dependency Inversion Principle

High-level policy depends on abstractions, not concretions:

- `BankAnalyzerFacade` is typed against `ClassifierProtocol` and `StatementLoaderProtocol`, not only the concrete classes.
- Loader depends on `StatementReaderFactory` / `StatementReader`, not on CSV/Excel/PDF classes directly.
- Classifier depends on `ClassificationHandler` + `RuleIndex`, not on inlined matching `if`s.
- Analytics depends on `AnomalyDetector`, with `ZScoreAnomalyDetector` as the default implementation.
- Constructor injection: `BankAnalyzerFacade(loader=..., classifier=..., analytics=...)` so tests can pass fakes.

This is the same idea as pattern §8 (DIP + injection) above.

### Honest limits

SOLID is applied at the **module and pipeline** level. `PdfStatementReader` is still a large parser (borderless layout + tables) because that logic is inherently format-specific. UIs still construct a default `BankAnalyzerFacade()` rather than a DI container. The goal was a course-project structure that is extendable without pretending every helper is a one-method class.

---

## Type hints and annotations

- `from __future__ import annotations` on modules.
- Dataclasses: `ClassificationResult`, `KPIMetrics`.
- Enums: `TransactionType`, `ClassificationMethod`.
- Protocols for loader/classifier contracts.
- Explicit types on method arguments and returns (`Path`, `pd.DataFrame`, `dict[str, float]`, `Iterator[...]`, `Optional[...]` / `X | None`).
- Shared `DEFAULT_BUDGETS` so Streamlit and Qt no longer keep divergent dicts.

---

## Behavioral fixes included in the refactor

1. **`check_budget_limits` returned inside the `for` loop**, so only the first category was reported. The `return` is now after the loop, so every category appears in the budget table.
2. **Dead code after `return` in PDF reading** was removed when PDF logic moved to `PdfStatementReader`.
3. **OLE/`.xls` magic bytes** used `b"\xd0\xcf\11\xe0"` (`\11` is a tab). Readers now use `b"\xd0\xcf\x11\xe0"`.
4. **Encapsulation:** `bank_display_name()` / `bank_labels()` replace UI access to `_configs`.

---

## How to extend (pattern-aligned)

| Change                   | What to add                                                                 |
| ------------------------ | --------------------------------------------------------------------------- |
| New file format          | `StatementReader` subclass + register in `StatementReaderFactory`       |
| New classification stage | `ClassificationHandler` subclass + link in `build_classification_chain` |
| New export format        | `ExportStrategy` subclass + entry in `ExportStrategyFactory._REGISTRY`  |
| New anomaly algorithm    | `AnomalyDetector` subclass passed into `detect_anomalies`               |
| New UI                   | Call `BankAnalyzerFacade` only                                            |

---

## What was deliberately not changed

- JSON configs (`bank_configs.json`, `category_rules.json`) and seed/sample CSVs.
- Plotly / Qt chart presentation.
- Hybrid matching order and confidence values (merchant 1.0, keyword 0.95, vector score).
- Streamlit Cloud / `streamlit run app.py` and `python qt_app.py` entry points.
