# Bank Statement Analyzer — design patterns and OOP map

## Layers

```
app.py (Streamlit UI) ── Translator (modules/i18n.py) ── locales/*.json
  DashboardApp ── DashboardTab (ABC)
                    ├ VisualAnalyticsTab
                    ├ TransactionExplorerTab
                    └ BudgetAnomaliesTab
        │ talks only to
        ▼
modules/facade.py      BankAnalyzerFacade
        │
        ├── ingestion.py    BankStatementLoader (StatementLoaderTemplate)
        │     ├ BankConfigRegistry, BankDetector
        │     ├ ColumnResolver, AmountParser, DateParser, DescriptionBuilder
        │     └ readers.py  StatementReaderFactory → Pdf/Excel/Csv readers
        │                     PdfStatementReader → BorderlessLayoutStrategy, TableLayoutStrategy
        │
        ├── classifier.py   TransactionClassifier
        │     ├ rules.py    JsonRuleRepository → RuleSet ──notifies──┐
        │     │                                                     ├ RuleMatcher
        │     │                                                     ├ FuzzyRuleIndex
        │     │                                                     └ LearnedCategoryModel (ml_model.py)
        │     └ classification_handlers.py  handler chain (built by ClassificationChainBuilder)
        │
        └── analytics.py    AnalyticsService ── AnomalyDetector (anomaly.py)
```

## Patterns

| Pattern | Where | Why |
|---|---|---|
| Facade | `BankAnalyzerFacade` | UIs call one object, never the subsystems directly. |
| Dependency Injection | constructors of the facade, classifier, loader and model | Every collaborator can be replaced (tests, evaluation, another bank). |
| Template Method | `StatementLoaderTemplate.load()`; `ExportStrategy.export()` | The order of steps is fixed in the base class; subclasses supply the steps. |
| Strategy | `StatementReader` subclasses, `PdfLayoutStrategy`, `ExportStrategy`, `AnomalyDetector` | Swap the algorithm (file format, PDF layout, export format, anomaly rule) without touching callers. |
| Factory | `StatementReaderFactory` (picks a reader by file), `ExportStrategyFactory` (by label) | Callers ask for "a reader for this file" and get the right class. |
| Registry + decorator | `@ExportStrategyFactory.register("CSV")` | A new export format is one new class; the factory is never edited (Open/Closed). |
| Chain of Responsibility | `ClassificationHandler` subclasses | Primary description → merchant → keyword → entrepreneur name → ML model → fuzzy match → Other. Each step either decides or passes on. |
| Builder | `ClassificationChainBuilder` | The chain order is written once, readably; optional steps (ML model) are skipped with `None`. |
| Observer | `RuleSet.subscribe()`; `on_rules_changed()` in `RuleMatcher`, `FuzzyRuleIndex`, `LearnedCategoryModel` | When a manual override adds a rule, every index and the model rebuild automatically. |
| Repository | `JsonRuleRepository` / `InMemoryRuleRepository`, `TrainingExampleStore` | Storage (JSON/CSV) is hidden behind load/save; evaluation uses the in-memory one so it never writes to disk. |
| Parameter Object | `ClassificationRequest` | Raw, cleaned and primary text plus excluded categories travel together; cleaning happens once. |
| Repository + injected Translator | `LocaleRepository`, `Translator` (`modules/i18n.py`) | No UI text in code. Every language is one `locales/<code>.json` file: add a file to add a language, delete it to remove one. Missing keys fall back to English. |
| Value Object | `ClassificationResult`, `KPIMetrics`, `ParsedAmounts`, `BorderlessColumns` (frozen dataclasses) | Immutable results that can't be changed by accident. |

## SOLID

- **Single Responsibility.** Each class has one job. For example, `TransactionClassifier` only wires things together. Rule storage moved to `rules.py`, matching to `RuleMatcher`/`FuzzyRuleIndex`, and loader steps to `ColumnResolver`, `AmountParser`, `DateParser` and `DescriptionBuilder`.
- **Open/Closed.** New export formats, PDF layouts, anomaly detectors, dashboard tabs and classification handlers are all added as new classes, without editing existing ones.
- **Liskov Substitution.** Every reader, handler, exporter and tab can stand in for its base class.
- **Interface Segregation.** `ClassifierProtocol` is the minimum the UI needs. `SelfTrainingClassifier` is a separate, optional capability, and the facade checks for it with `isinstance` instead of `getattr`.
- **Dependency Inversion.** The facade depends on `ClassifierProtocol` and `StatementLoaderProtocol`, not on concrete classes.

## One text cleaner

`TextNormalizer.clean` is the only cleaning function. The loader's `cleaned_description`, the rule matcher and manual overrides all use it. As a result, a keyword saved from an override always matches the text the classifier sees.

## Languages

- **Add a language:** copy `locales/en.json` to `locales/<code>.json` (e.g. `ru.json`), set `meta.name` to the language's own name, and translate the values. It appears in the sidebar switch automatically.
- **Check for gaps:** `python -m modules.i18n` lists missing or unknown keys per language.
- **What's translated:** UI text, table headers, category names, transaction types, method names, bank names, and the currency word.
- **What stays English:** the data itself (category keys, `income`/`expense`), so analytics, budgets and training never depend on the language.
- **Exports** use the on-screen language. `evaluate_classifier.py` maps headers and values back from any language.
