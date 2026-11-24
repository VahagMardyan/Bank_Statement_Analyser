# Bank Statement Analyzer

A Streamlit app that reads bank statements (CSV, XLS, XLSX, PDF), puts every transaction into a spending category and shows where the money went. It understands Armenian, English and Russian bank exports. Categories come from rules plus a machine learning model that learns from your own statements and corrections.

Each person has their own account. Uploaded transactions are saved in MySQL, so the dashboard shows your whole history, not just the last file, and uploading the same statement twice doesn't create duplicates. From that history the app finds recurring payments (subscriptions, rent, utilities) and the date each one is due next. It also forecasts the month's spending, warns when a category is heading over budget, flags payments that look suspicious, and can email reminders before payments are due.

Statements don't have to be in AMD. A USD, EUR, RUB or other foreign statement is converted to AMD with saved exchange rates, and the transaction table can be shown and downloaded in any of the supported currencies.

The interface is in Armenian and English. Another language is one more JSON file.

## Design patterns

The project follows SOLID principles and implements these design patterns:


| Pattern                             | Where                                                                                                                                                                                          | Purpose                                                                                                                                                                        |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Facade**                          | `BankAnalyzerFacade` (`facade.py`)                                                                                                                                                             | One entry point (load → classify → learn → analyze) for the UI. The UI never reaches into the subsystems.                                                                   |
| **Template Method**                 | `StatementLoaderTemplate.load` (`ingestion.py`), `ExportStrategy.export` (`exporters.py`)                                                                                                      | Fixed pipeline: read file → standardize → drop invalid rows → finalize. Subclasses supply the steps.                                                                        |
| **Strategy**                        | `StatementReader`, `PdfLayoutStrategy`, `ExportStrategy`, `AnomalyDetector`, `RecurringDetector`, `AlertRule`, `NotificationChannel`                                                           | Interchangeable algorithms for file formats (PDF / Excel / CSV), PDF layouts (borderless / table), exports (CSV / Excel) and anomaly detection (Z-score)                       |
| **Factory**                         | `StatementReaderFactory`, `ExportStrategyFactory`                                                                                                                                              | Picks the right reader for a file and the right exporter for a UI choice                                                                                                       |
| **Registry (decorator)**            | `@ExportStrategyFactory.register("CSV")`                                                                                                                                                       | A new export format is one new class; the factory is never edited (Open/Closed)                                                                                                |
| **Chain of Responsibility**         | `ClassificationHandler` (`classification_handlers.py`)                                                                                                                                         | Override keywords (yours, then community) → primary description → merchant → keyword → individual entrepreneur → ML model → fuzzy match → `Other` |
| **Builder**                         | `ClassificationChainBuilder`                                                                                                                                                                   | Assembles the chain in one readable place; optional steps (the ML model) are skipped with `None`                                                                               |
| **Observer**                        | `RuleSet.subscribe` → `RuleMatcher`, `FuzzyRuleIndex`, `LearnedCategoryModel` (`rules.py`)                                                                                                    | When an override adds a rule, every index and the ML model rebuild automatically                                                                                               |
| **Repository**                      | `JsonRuleRepository` / `InMemoryRuleRepository`, `TrainingExampleStore`, `LocaleRepository`, `RateStore` (`currency.py`); MySQL versions in `mysql_repositories.py`, `users.py`, `remember.py` | Storage is hidden behind `load` / `save`. The same classifier works with JSON / CSV files or with MySQL, and evaluation runs on an in-memory copy that never touches your data |
| **Factory Method**                  | `BankAnalyzerFacade.for_user(user_id)`                                                                                                                                                         | Builds a facade whose rules, ML examples, history and recurring payments all belong to one user. The UI doesn't need to know how it's assembled                                |
| **Parameter Object**                | `ClassificationRequest` (`models.py`)                                                                                                                                                          | Raw text, cleaned text, bank description and excluded categories travel together; text is cleaned once per row                                                                 |
| **Value Object**                    | `ClassificationResult`, `KPIMetrics`, `ParsedAmounts`, `Prediction`, `User`, `RecurringPayment`, `Period` (frozen dataclasses)                                                                 | Immutable results. `User` has no password hash on purpose, since it lives in the Streamlit session                                                                             |
| **Dependency Injection / Protocol** | `StatementLoaderProtocol`, `ClassifierProtocol`, `SelfTrainingClassifier`, `Translator`, `RateSource`                                                                                          | High-level code depends on interfaces, so the loader, classifier, model, language and exchange rate source can be swapped or mocked in tests                                   |

How SOLID shows up:

- **Single Responsibility:** e.g. `TransactionClassifier` only wires collaborators together. Rule storage, matching, fuzzy search and ML each have their own class, and so do column resolution, amount parsing and date parsing in the loader.
- **Open/Closed:** new readers, PDF layouts, exporters, anomaly detectors, handlers, dashboard tabs and languages are all added without editing existing classes.
- **Interface Segregation:** `ClassifierProtocol` is the minimum the UI needs. Self-training is a separate optional protocol (`SelfTrainingClassifier`).
- **Dependency Inversion:** the facade depends on protocols, not concrete classes.

UML class diagrams are in `UML/Mmd_Files/` (Mermaid source) and `UML/Png_Files/` (rendered). They don't cover the database, account, recurring payment and currency classes yet:


| File                            | Shows                                                               |
| --------------------------------- | --------------------------------------------------------------------- |
| `01_ingestion_system.mmd`       | Loader template, column/amount/date parsing, readers, PDF layouts   |
| `02_classification_system.mmd`  | Classifier, rules + repository + observers, handler chain, ML model |
| `03_analytics_and_exporter.mmd` | Analytics service, anomaly strategy, export strategies + registry   |
| `04_high_level_facade.mmd`      | Facade, protocols, enums                                            |
| `05_ui_and_i18n.mmd`            | Dashboard tabs, translator, locale files                            |

## Project structure

```
.
├── bank_analyzer/
│   ├── config/
│   │   └── category_rules.json          # Categories → keywords + merchants (with aliases)
│   ├── locales/
│   │   ├── en.json                      # All UI text in English (reference file)
│   │   └── hy.json                      # All UI text in Armenian
│   ├── db/
│   │   └── schema.sql                   # MySQL database: 8 tables
│   ├── data/
│   │   ├── training_data.csv            # Created at runtime when running without a database
│   │   ├── exchange_rates.json          # Saved exchange rates, created at runtime (see Foreign currencies)
│   │   ├── labelled_all.xlsx            # Hand-checked labels (849 real transactions) for evaluation
│   │   ├── fake_statement_ML_demo.csv   # Demo statement with merchants unknown to the rules
│   │   └── fake_statement_recurring_demo.csv  # 6.5 months of made-up payments for testing recurring detection
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
│   │   ├── db.py                        # One shared SQLAlchemy engine; settings come from .env
│   │   ├── mysql_repositories.py        # MySQL storage: per-user rules, ML examples, transactions, recurring payments
│   │   ├── history.py                   # TransactionRepository interface, duplicate-proof row fingerprints
│   │   ├── recurring.py                 # RecurringDetector, Period, RecurringPayment, RecurringRepository
│   │   ├── users.py                     # User storage, bcrypt password hashing, AuthService
│   │   ├── remember.py                  # "Keep me signed in" tokens
│   │   ├── budgets.py                   # BudgetRepository interface (monthly limits per category)
│   │   ├── advisor.py                   # SpendingAdvisor: month status, advice, end-of-month forecast
│   │   ├── alerts.py                    # SuspiciousActivityDetector and its AlertRule strategies
│   │   ├── notifications.py             # Email reminders: channels, notification log, ReminderService
│   │   ├── report.py                    # Monthly .xlsx report (one sheet per table)
│   │   ├── currency.py                  # Statement currency detection, exchange rates, conversion to and from AMD
│   │   ├── i18n.py                      # Translator, LocaleRepository, missing-translation checker
│   │   ├── models.py                    # Enums, dataclasses, protocols
│   │   └── constants.py                 # Default budgets, income categories
│   ├── .env                             # Database settings (not in git)
│   ├── ui/
│   │   ├── auth_page.py                 # AuthPage: sign in / create account
│   │   ├── dashboard.py                 # DashboardApp: sidebar, KPI row, tab container
│   │   ├── tabs.py                      # DashboardTab (ABC) + one class per tab
│   │   ├── account_panel.py             # AccountPanel: current user, account settings, sign out
│   │   └── common.py                    # Session keys, cookies, flash messages, language switch
│   ├── app.py                           # Streamlit entry point: shows the login page or the dashboard
│   ├── send_reminders.py                # Daily job: sends reminder emails (or prints them with --dry-run)
│   ├── migrate_currency.py              # One-time update for databases made before currency support
│   ├── promote_rules.py                 # Copies community rules into config/category_rules.json
│   └── evaluate_classifier.py           # Accuracy evaluation + threshold tuning
├── requirements.txt
├── README.md
├── test_*.csv                           # Generated CSV statements
├── generate_statements_en.py            # Statement generator (English)
├── generate_statements_am.py            # Statement generator (Armenian)
├── Real_Statements/                     # Real statements for testing (in .gitignore for privacy)
└── UML/                                 # Project UML diagrams (Mermaid)
```

## Requirements

- Python 3.10+
- MySQL 8 or TiDB Cloud. The upsert queries use `ON DUPLICATE KEY UPDATE col = VALUES(col)`, which both accept. TiDB doesn't run the newer `INSERT ... AS new` form, so the code doesn't use it.
- Python packages from `requirements.txt` (in the repository root): pandas, scikit-learn, scipy, pdfplumber, openpyxl, xlrd, streamlit, plotly, and for the database and accounts sqlalchemy, pymysql, cryptography, certifi, python-dotenv, bcrypt
- A free [currencyapi.com](https://currencyapi.com) key, only if you load statements in currencies other than AMD

## Installation

```bash
cd bank_analyzer
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r ../requirements.txt
```

`requirements.txt` sits in the repository root because Streamlit Cloud reads it from there.

### Database

Create `bank_analyzer/.env` with your MySQL login. It's in `.gitignore`, so the password stays off GitHub:

```
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=bank_analyser

# Only for statements in other currencies
CURRENCY_API_KEY=your_currencyapi_key
```

For a hosted database that requires an encrypted connection (TiDB Cloud does), also add `DB_SSL=true`.

Then create the database and tables from the `bank_analyzer/db` folder:

```powershell
mysql -u root -p --default-character-set=utf8mb4 -e "source schema.sql"
```

PowerShell doesn't support `<`, which is why the command uses `source`. Check that it worked with `mysql -u root -p -e "USE bank_analyser; SHOW TABLES;"`, which should list 8 tables. To start over with an empty database, run `DROP DATABASE bank_analyser;` and load the schema again. That deletes every account too.

A database made before currency support needs two new columns in `transactions`. Run this once from `bank_analyzer`. It does nothing if the columns are already there:

```powershell
python migrate_currency.py
```

Without a database the classifier still works on its own (`BankAnalyzerFacade()` uses the JSON and CSV files), which is how `evaluate_classifier.py` runs.

## Running the dashboard

```bash
cd bank_analyzer
streamlit run app.py
```

The app opens in your browser (default: http://localhost:8501). On Windows, `runner.bat` does the same.

### What's on screen


| Section                  | Description                                                                                                                                                                            |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Login page**           | Sign in or create an account, "Keep me signed in", language switch                                                                                                                     |
| **Sidebar**              | Language switch, your name and email, sign out, account settings, statement upload (CSV / XLS / XLSX / PDF), which exchange rates were used for a foreign statement, date range filter |
| **Metrics**              | Total income, expenses, net savings, transaction count                                                                                                                                 |
| **Visual Analytics**     | Donut chart, monthly trends, top 5 expenses                                                                                                                                            |
| **Transaction Explorer** | Filterable table with `confidence` and `method` columns, currency picker, CSV / Excel export in the chosen currency, manual category override                                          |
| **Budget & Anomalies**   | Month selector, saved monthly limits per category, progress bars, monthly .xlsx report, Z-score anomaly detection                                                                      |
| **Recurring Payments**   | Detected regular payments with amount, period, last and next payment date, days left; wrong ones can be hidden, missing ones added by hand                                             |
| **Advisor**              | This month so far, recurring payments still due, end-of-month estimate, budget warnings                                                                                                |
| **Alerts**               | Possible duplicate charges, large first payments to new merchants, price increases, night payments                                                                                     |

## Accounts and stored data

Passwords are hashed with bcrypt and never stored as text. Signing in with a wrong email and with a wrong password gives the same message, so the login form can't be used to check who has an account.

Everything a user does is stored under their id:


| Table                | What's in it                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| `users`              | Email, name, password hash, account language                                                          |
| `transactions`       | Every uploaded transaction with its category, the amount in AMD, and the original amount and currency |
| `user_rules`         | Keywords from your manual overrides. When enough users agree on one, it becomes a community rule for everyone (see [Overrides and community rules](#overrides-and-community-rules)) |
| `training_examples`  | What the ML model has learned for you                                                                 |
| `recurring_payments` | Detected regular payments, including the ones you hid                                                 |
| `auth_tokens`        | "Keep me signed in" tokens                                                                            |
| `budgets`            | Monthly limits per category                                                                           |
| `notifications_log`  | Reminders and warnings already emailed, so none is sent twice                                         |

So one person's corrections never change how another person's transactions are categorized.

Uploading a statement saves only the rows that aren't stored yet. Each row gets a SHA-256 fingerprint from its date, original amount, type and description (plus the currency code when it isn't AMD), and `(user_id, fingerprint)` is unique in the table. If the same payment appears twice in one file (two bus tickets in the same minute), the second copy also gets its position in the file mixed in, so both are kept, and re-uploading the file still matches. One limit: the same month exported once as PDF and once as CSV can be saved twice, because the description text or the time can differ between formats.

The upload itself goes to a temporary file that is deleted right after reading, so real statements don't stay on disk.

The facade keeps the loaded history and recurring list in memory between Streamlit reruns, and drops them whenever something changes them (an upload, an override, clearing history, adding or hiding a recurring payment). So the Advisor, Alerts and report don't query MySQL again on every click. If the same account is open in two browsers, an upload in one shows up in the other after signing in again.

### Account settings

Under **Account settings** in the sidebar you can:

- change your name
- change the account language: the app opens in it after you sign in, and reminder emails use it
- change your password (asks for the current one)
- clear your history: deletes your transactions and detected recurring payments, but keeps your category corrections
- delete your account: asks for your password, then `ON DELETE CASCADE` removes everything stored for you

Clearing and deleting both need a "can't be undone" box ticked first.

### Keep me signed in

With the box ticked, the browser gets a cookie that holds two random strings, `selector:validator`. It doesn't contain your user id, because anyone could edit an id in the browser and become someone else. The database stores the selector and only a SHA-256 hash of the validator, so a copy of the database can't be turned into working cookies. A token lasts 30 days. Signing out deletes it, changing the password signs out every other remembered device, and a cookie with the right selector but a wrong validator is treated as forged and its token is deleted.

Streamlit can read cookies (`st.context.cookies`) but not set them, so the cookie is written by a small script. That means it can't be `HttpOnly`. The cookie uses `SameSite=Lax`. `Strict` didn't work, because the browser then left the cookie out of some requests and the app never saw it.

## Recurring payments

After every upload the app looks at your whole stored history (expenses only), grouped by the cleaned description. A merchant counts as a recurring payment when:

1. it was paid on at least 3 different days
2. the median gap between payments is close to 7, 30, 91 or 365 days, and at least 75% of the gaps fit that period, so one skipped month is fine
3. the amount is stable: standard deviation / mean ≤ 0.35. `Utilities` skip this check, because a winter gas bill can be several times the summer one
4. the last payment is no more than two periods before your latest transaction. Older ones are treated as cancelled. The latest transaction is used instead of today, because statements are often uploaded weeks late

The expected amount is the median of the last 3 payments. For monthly, quarterly and yearly payments the next due date keeps the same day of the month (paid on the 15th, next due on the 15th).

If something isn't really a regular payment, hide it in the tab. Hidden payments stay hidden after later uploads. All the thresholds are constructor arguments of `RecurringDetector`, so they can be tuned without touching the code that uses it.

`data/fake_statement_recurring_demo.csv` is a made-up statement for checking this (May 1 to Nov 15, 2026). The app should find 9 recurring payments: rent, Netflix, Telegram Premium, Ucom internet, ENA electricity, Gazprom gas, YouTube Premium (with August skipped), a weekly gym class and quarterly car insurance. Spotify (stopped in July), the salary and the random supermarket, taxi and cafe spending should not show up.

## Budgets

Limits are monthly and saved per user (`budgets` table). The Budget tab compares them with one month at a time, the latest by default, because comparing a monthly limit with several months of history would put every category over budget. Categories you never changed use the defaults from `DEFAULT_BUDGETS` in `modules/constants.py`.

The same tab has a **Download the report** button. It builds an Excel file for the selected month with these sheets: summary, categories, budgets, transactions, recurring payments, and the month's alerts and advice when there are any. The file is in the language you have on, and it's built only when you click.

## Advisor

The Advisor tab looks at the month of your latest transaction (statements usually arrive late, so this isn't always the calendar month).

The forecast has three parts:

- what you've spent so far
- everyday spending for the rest of the month, at the current daily pace
- recurring payments still due this month, each at its expected amount

Recurring payments and large one-off purchases (above both 30,000 AMD and the 95th percentile of your payments) are counted once and aren't extended at the daily pace. Otherwise a rent paid on the 1st, or one laptop, would be "projected" again for every remaining day. Transfers and cash withdrawals aren't counted as spending. Before day 7 there's no pace projection, because a few days of data say very little.

Advice comes in three kinds, most serious first:

1. already over the monthly budget
2. not over yet, but will be at this pace (by at least 5,000 AMD)
3. heading 25% or more above your usual spending for that category, where "usual" is the average of up to 3 earlier months

The thresholds are constructor arguments of `SpendingAdvisor`.

## Alerts

The Alerts tab shows payments from the last 90 days of your history that are worth a second look. Each check is its own `AlertRule` class, so a new check is one new class:


| Rule                   | Fires when                                                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `DuplicateChargeRule`  | The same merchant charges the same amount (5,000 AMD or more) again within 10 minutes, or on the same day when the statement has no times |
| `NewMerchantLargeRule` | The first payment to a merchant is above both 30,000 AMD and the 95th percentile of your payments. Known recurring payments don't count   |
| `PriceIncreaseRule`    | A recurring payment's latest amount is 15% or more above its earlier median (utilities skipped, their amounts always vary)                |
| `NightPaymentRule`     | A payment of 10,000 AMD or more between 01:00 and 05:00 (only when the statement has times)                                               |

These are hints, not proof of fraud, and the tab says so.

## Foreign currencies

Budgets, the advisor and the alerts all work in AMD, so a statement in another currency is converted to AMD when it's loaded. Supported currencies: AMD, USD, EUR, RUB, GBP, GEL, AED (UAE dirham) and CNY (Chinese yuan).

The statement currency is found in this order:

1. The header above the table: `Հաշվի արժույթ` in Ameriabank, Evocabank and ACBA statements, or `Արժույթ` with `ՀՀ դրամ` under it in Idram. In the Ameriabank PDF it's read from the totals lines (`... 123,456.00 AMD`).
2. A code or symbol next to the amounts, like `-300.00 USD` in Ameriabank CSV exports or `-£23.40`.
3. A currency column, but only when every row has the same value.
4. AMD, when none of the above is found.

The `Արժույթ` column in Ameriabank xlsx files is the currency of each purchase, not of the account, so it isn't trusted on its own. A Netflix payment in USD on an AMD card stays an AMD row with the amount the bank charged.

Exchange rates come from [currencyapi.com](https://currencyapi.com) (free plan: 300 requests a month) with the key from `CURRENCY_API_KEY`. They're saved in `data/exchange_rates.json`. The app downloads them only when a foreign statement or currency needs them, and at most every 3 days. A new dated snapshot is added only if some currency moved 3% or more against AMD since the last one. Otherwise the file just records when it was checked. That keeps it at roughly 10 requests a month. Both numbers can be changed in `.env` with `CURRENCY_REFRESH_DAYS` and `CURRENCY_CHANGE_PCT`.

Each transaction is converted with the newest snapshot on or before its date. Transactions older than the first snapshot use the first one, and the app says their AMD amounts are approximate. If the API can't be reached, the last saved rates are used. With no saved rates at all, the upload stops and asks for the key. After a failed download the app waits 6 hours before trying again (5 minutes if a needed currency is missing from the file), so Streamlit reruns don't eat the monthly limit.

The values from the statement are kept in `original_amount` and `original_currency`. The duplicate check uses them, not the AMD amount, so loading the same USD statement again after the rates changed adds nothing.

The Transaction Explorer has a currency picker. The table and its CSV or Excel download use the chosen currency, and the file name ends with its code (`transactions_export_USD.xlsx`). Each row is converted at the rate of its own date, and rows that were already in that currency show their original amount exactly. The saved data stays in AMD.

On Streamlit Cloud the rates file is lost when the app restarts, which costs one extra request.

## Email reminders

Turn reminders on under **Account settings → Email reminders** and choose how many days before a payment to be reminded (3 by default). Emails are written in your account language.

`send_reminders.py` sends one email per user with:

- recurring payments due within that many days
- budget warnings for the current month (over budget, or heading over)

`notifications_log` records every reminder and warning sent, so nothing is sent twice even if the script runs several times a day. Due dates of manual payments always move to the next one ahead. Detected payments that are a few days overdue keep their date until the statement with that payment is uploaded.

Add your mail settings to `bank_analyzer/.env`. For Gmail you need an [app password](https://support.google.com/accounts/answer/185833), not your normal password:

```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM=you@gmail.com
```

Try it without sending anything first. `--dry-run` prints the emails and doesn't record them, and `--today` pretends it's another day:

```powershell
python send_reminders.py --dry-run
python send_reminders.py --dry-run --today 2026-11-28
```

To run it every day on Windows, create a scheduled task (adjust the paths):

```powershell
schtasks /Create /SC DAILY /ST 09:00 /TN "BankAnalyzerReminders" /TR "C:\Users\user\Desktop\Course_Project\bank_analyzer\.venv\Scripts\python.exe C:\Users\user\Desktop\Course_Project\bank_analyzer\send_reminders.py"
```

The computer has to be on at that time. MySQL has to be running too, since the script reads the same database as the app.

## Languages

Everything on screen comes from `locales/<code>.json`: labels, messages, table headers, category names, transaction types (Մուտք / Ելք), method names, bank names and the currency word. The data itself always stays in English (category keys, `income` / `expense`), so analytics, budgets and the ML model never depend on the language.


| Task                   | How                                                                                                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Add a language         | Copy `locales/en.json` to `locales/<code>.json` (e.g. `ru.json`), set `"meta"` (`"flag": "🇷🇺"` for the sidebar switch, `"label": "Русский"` for the account language list), translate the values. It appears in both automatically. |
| Edit text              | Change the value in that language's file. No code changes are needed.                                                                                                              |
| Remove a language      | Delete its file.                                                                                                                                                                   |
| Find untranslated text | `python -m modules.i18n` lists missing and unknown keys per language                                                                                                               |
| Default language       | `DEFAULT_LANGUAGE` in `modules/i18n.py` (Armenian). Used on the login page until someone signs in                                                                             |

Anything missing in a language falls back to English, then to the key itself, so a half-finished translation never crashes the app. Text placeholders such as `{count}` or `{currency}` must be kept as they are.

Exports use the on-screen language. `evaluate_classifier.py` maps headers and values from any language back to the English keys, so an Armenian export can be used as a labelled file directly.

Not translatable from the locale files: Streamlit's built-in upload-box text and the date-picker calendar, and error details from the file loader.

Every account has its own language. You pick it when you create the account (the list starts at the language the page is in), and you can change it later in **Account settings**. After signing in, including with "Keep me signed in", the app switches to the account language. The switch at the top of the sidebar changes only the current session; it doesn't change the account. After signing out, the login page stays in the language you last had on.

## Quick test with sample data

1. Run `streamlit run app.py` and create an account
2. Upload a few real statements first (the model learns from them)
3. Upload `data/fake_statement_ML_demo.csv`. Its merchants are not in the rules, so many rows are decided by the ML model (`method = ML model`)
4. Correct a wrong row with **Manual Category Override**, then reload the file: the correction is remembered
5. Upload the same file again: the message should say 0 new transactions
6. Upload `data/fake_statement_recurring_demo.csv` and open the **Recurring Payments** tab (see [Recurring payments](#recurring-payments) for the expected list)
7. Open **Advisor**: with the default budgets it warns about Insurance (a 45,000 quarterly payment against a 20,000 monthly limit) and Supermarket
8. In **Budget & Anomalies**, pick a month and download its report
9. With `CURRENCY_API_KEY` set, upload a statement in another currency. The sidebar shows which rates were used. In the **Transaction Explorer**, switch the currency and download the table

## Supported banks and formats

Tested on real statements:


| Bank            | Formats        | Notes                                                                                                                                                                                                                                                                                                   |
| ----------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ameriabank**  | CSV, XLSX, PDF | CSV payee column (`Շահառու/Վճարող`) is used. Amounts are taken in the account currency, so a USD purchase on an AMD card is stored at the AMD amount the bank charged. USD account statements are recognized from the header and converted (see [Foreign currencies](#foreign-currencies)) |
| **ACBA Bank**   | XLSX           | The merchant is read from `Գործարքի վայրը` (ACBA's description is only `Վճարում`)                                                                                                                                                                                                   |
| **Idram**       | XLSX           | Debit / credit columns                                                                                                                                                                                                                                                                                  |
| **Evocabank**   | XLSX           |                                                                                                                                                                                                                                                                                                         |
| **Other banks** | any            | Not tested, but any statement whose headers match the built-in hints is read                                                                                                                                                                                                                            |

There is no per-bank configuration and no bank selection. Every file goes through the same steps:

- **Header row:** found by date and amount words.
- **Columns:** found by the Armenian and English header hints in `COLUMN_HINTS` (`modules/ingestion.py`). A new header wording is supported by adding the word to that list.
- **Dates:** parsed so ISO dates (`2026-03-04`) and day-first dates (`04.03.2026`) are never mixed up.

## Categorization logic

Each transaction goes through the chain until one step is confident:

0. **Your corrections and community rules:** keywords saved from manual overrides are checked before everything else, so a correction also beats a wrong merchant rule from `category_rules.json`. Your own rule wins over a community rule for the same keyword. Confidence 1.0, method `User rule` or `Community rule` (see [Overrides and community rules](#overrides-and-community-rules)).
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
   - **Retraining:** automatic after every upload and every override. The training data is saved per user in the `training_examples` table (in `data/training_data.csv` when running without a database)
6. **Fuzzy match:** char n-gram TF-IDF shortlist + word-by-word similarity check on short spans of the description; catches typos such as `STARBUKS` → Starbucks.
7. **Other:** nothing was confident enough.

Expense rows can never be classified as `Income` or `Salary`.

### Overrides and community rules

A manual override does three things:

- saves a keyword to your own rules (the `user_rules` table; `category_rules.json` when running without a database)
- re-labels the rows that keyword matches, on screen and in your saved history
- retrains the model

The keyword is the meaningful part of the description, not the whole line. For `Ք: VILEX LLC VILEX LLC 051 Ամերիաբանկ ՓԲԸ Գլխամասային գրասենյակ` it saves `vilex`: the bank boilerplate and numbers are dropped, and the longest remaining run of words (up to 4) is kept. Runs that are only category words, like `transfer to`, are skipped when there's something better, so one correction doesn't catch every transfer. The keyword is always a continuous piece of the cleaned text, so it matches the next statement with the same merchant. Keywords saved before this change keep working, and correcting the same text again moves them to the new category too.

When several users correct the same keyword the same way, it becomes a **community rule** that applies to everyone:

- at least 3 different users chose that category for the keyword, and
- they are at least 75% of all users who corrected that keyword. So 3 users saying Shopping and 2 saying Cafes (60%) gives no rule.

Both numbers can be changed in `.env` with `COMMUNITY_MIN_USERS` and `COMMUNITY_MIN_SHARE`. Community rules are computed from `user_rules` when an account is opened, so there's no extra table, and they work on Streamlit Cloud too. A community rule never overrides your own correction, and it isn't saved as your vote.

To make community rules part of the shared JSON file (and of git), run `promote_rules.py` from `bank_analyzer`:

```powershell
python promote_rules.py            # shows what would change, changes nothing
python promote_rules.py --write    # writes config/category_rules.json
```

It adds each keyword to its category and removes it from other categories. When a merchant in another category matches the same text, it doesn't change the merchant but prints a `check` line, because inside the JSON merchant rules come before keywords. Look at the result with `git diff config/category_rules.json` before committing. For TiDB, set the `DB_*` values in the PowerShell session first, the same way as for `migrate_currency.py` (see [Testing](#testing)).

The rule change notifies the matcher, the fuzzy index and the model, so the correction applies to future statements right away.

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


| Test                                             | Accuracy | Other | Wrong guesses |
| -------------------------------------------------- | ---------- | ------- | --------------- |
| Full pipeline                                    | 98.9%    | 0.7%  | 0.4%          |
| Unseen merchants, old thresholds (0.40 / 0.10)   | 62.6%    | 17.0% | 20.4%         |
| Unseen merchants, tuned thresholds (0.25 / 0.20) | 63.4%    | 32.7% | **3.9%**      |

The tuned thresholds keep accuracy the same but cut wrong guesses about five-fold: the model says "Other" instead of guessing, and a wrong category would silently distort budgets.

## Unified transaction schema


| Column                | Type     | Description                                                                                                                                               |
| ----------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `date`                | datetime | Transaction date                                                                                                                                          |
| `description`         | str      | Bank text; merchant/payee appended after `|` when the bank stores it in a separate column                                                                 |
| `cleaned_description` | str      | Same text after `TextNormalizer.clean` (lower-case; dates, times, card numbers, IDs and punctuation removed). Manual-override keywords are saved from it. |
| `amount`              | float    | Absolute value in AMD. Statements in other currencies are converted when loaded                                                                           |
| `original_amount`     | float    | Absolute value as written in the statement                                                                                                                |
| `original_currency`   | str      | Currency code of the statement (`AMD`, `USD`, `EUR` and so on)                                                                                            |
| `transaction_type`    | str      | `income` or `expense`                                                                                                                                     |
| `transaction_id`      | str      | Row identifier: the row number in a freshly loaded file, the database id for saved history                                                                |
| `category`            | str      | Assigned category key (English; shown translated in the UI)                                                                                               |
| `confidence`          | float    | Classification confidence (0–1)                                                                                                                          |
| `method`              | str      | `User rule`, `Community rule`, `Merchant rule`, `Keyword rule`, `Name heuristic`, `ML model`, `Vector`, `manual`, or `None`                           |
| `fingerprint`         | str      | SHA-256 used to skip duplicates (only on rows loaded from the database)                                                                                   |

## Extending category rules

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

You can also use **Manual Category Override** in the UI. Those keywords are saved for your account, and the model retrains. Once enough users agree on a correction it applies to everyone, and `promote_rules.py` copies it into this file (see [Overrides and community rules](#overrides-and-community-rules)). Editing `category_rules.json` by hand changes the rules for everyone right away.

## Testing

Visit [this link](https://statementanalyser-vahagmardyan.streamlit.app/) to use the application from any other device. Streamlit Cloud deletes local files on restart, so the online copy uses a free TiDB Cloud database. Its `DB_*` values, `DB_SSL = "true"` and `CURRENCY_API_KEY` go in the app's Secrets instead of `.env`. Only made-up statements are uploaded there; real ones stay in the local MySQL.

To update the TiDB schema from your own computer without editing `.env`, set the TiDB values for one PowerShell session. `.env` doesn't override variables that are already set:

```powershell
$env:DB_HOST = 'your-tidb-host'
$env:DB_PORT = '4000'
$env:DB_USER = 'your_tidb_user'
$env:DB_PASSWORD = 'your_tidb_password'
$env:DB_NAME = 'bank_analyser'
$env:DB_SSL = 'true'
python migrate_currency.py
```

## Credits

Flag emoji font: [Twemoji Country Flags](https://github.com/talkjs/country-flag-emoji-polyfill), built from [Twemoji](https://github.com/twitter/twemoji) graphics © Twitter, Inc. and other contributors, licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). Used unchanged.
