"""PySide6 desktop GUI for Bank Statement Analyzer."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QPieSeries,
    QValueAxis,
)
from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QAction, QFont, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from modules.analytics import (
    check_budget_limits,
    compute_kpis,
    detect_anomalies,
    get_category_summary,
    get_monthly_trends,
)
from modules.classifier import TransactionClassifier
from modules.ingestion import BankStatementLoader

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BUDGETS = {
    "Transport": 50000,
    "Supermarket": 80000,
    "Restaurants": 40000,
    "Utilities": 30000,
    "Entertainment": 20000,
    "Shopping": 50000,
}


def format_amd(value: float) -> str:
    return f"{value:,.0f} AMD"


class MetricCard(QGroupBox):
    """Small KPI card widget."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        self.value_label = QLabel("—")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.value_label.setFont(font)
        layout = QVBoxLayout(self)
        layout.addWidget(self.value_label)

    def set_value(self, text: str) -> None:
        self.value_label.setText(text)


class DataFrameTable(QTableWidget):
    """Reusable table bound to a pandas DataFrame."""

    def load_dataframe(self, df: pd.DataFrame) -> None:
        self.clear()
        if df is None or df.empty:
            self.setRowCount(0)
            self.setColumnCount(0)
            return

        display = df.copy()
        for col in display.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
            display[col] = display[col].dt.strftime("%Y-%m-%d")

        self.setRowCount(len(display))
        self.setColumnCount(len(display.columns))
        self.setHorizontalHeaderLabels([str(c) for c in display.columns])

        for row_idx, row in enumerate(display.itertuples(index=False)):
            for col_idx, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.setItem(row_idx, col_idx, item)

        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)


class OverrideDialog(QDialog):
    """Dialog for manual category override."""

    def __init__(
        self,
        transaction_id: str,
        description: str,
        categories: list[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manual Category Override")
        layout = QFormLayout(self)
        layout.addRow("Transaction ID:", QLabel(transaction_id))
        layout.addRow("Description:", QLabel(description[:80]))
        self.category_combo = QComboBox()
        self.category_combo.addItems(categories)
        layout.addRow("New Category:", self.category_combo)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def selected_category(self) -> str:
        return self.category_combo.currentText()


class BankAnalyzerWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Bank Statement Analyzer")
        self.resize(1280, 820)

        self.loader = BankStatementLoader()
        self.classifier = TransactionClassifier()
        self.df: Optional[pd.DataFrame] = None
        self.filtered_df: Optional[pd.DataFrame] = None
        self.budgets = dict(DEFAULT_BUDGETS)

        self._build_menu()
        self._build_ui()
        self.statusBar().showMessage("Ready — open a CSV or Excel bank statement to begin.")

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("File")
        open_action = QAction("Open Statement…", self)
        open_action.triggered.connect(self.open_statement)
        menu.addAction(open_action)
        quit_action = QAction("Exit", self)
        quit_action.triggered.connect(self.close)
        menu.addAction(quit_action)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        controls = QHBoxLayout()
        self.bank_combo = QComboBox()
        self.bank_combo.addItem("Auto Detect", "auto_detect")
        for bank_key in self.loader.list_banks():
            display = self.loader._configs[bank_key]["display_name"]
            self.bank_combo.addItem(display, bank_key)

        self.start_date = QDateEdit(calendarPopup=True)
        self.end_date = QDateEdit(calendarPopup=True)
        self.start_date.setEnabled(False)
        self.end_date.setEnabled(False)
        self.start_date.dateChanged.connect(self.apply_filters)
        self.end_date.dateChanged.connect(self.apply_filters)

        open_btn = QPushButton("Open Statement")
        open_btn.clicked.connect(self.open_statement)

        controls.addWidget(QLabel("Bank:"))
        controls.addWidget(self.bank_combo)
        controls.addWidget(open_btn)
        controls.addStretch()
        controls.addWidget(QLabel("From:"))
        controls.addWidget(self.start_date)
        controls.addWidget(QLabel("To:"))
        controls.addWidget(self.end_date)
        root.addLayout(controls)

        metrics_row = QHBoxLayout()
        self.income_card = MetricCard("Total Income")
        self.expense_card = MetricCard("Total Expenses")
        self.savings_card = MetricCard("Net Savings")
        self.txn_card = MetricCard("Transactions")
        for card in (self.income_card, self.expense_card, self.savings_card, self.txn_card):
            metrics_row.addWidget(card)
        root.addLayout(metrics_row)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self._build_analytics_tab()
        self._build_explorer_tab()
        self._build_budget_tab()

    def _build_analytics_tab(self) -> None:
        tab = QWidget()
        layout = QGridLayout(tab)

        self.pie_chart_view = QChartView()
        self.pie_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.trend_chart_view = QChartView()
        self.trend_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.top_expenses_table = DataFrameTable()
        layout.addWidget(self.pie_chart_view, 0, 0)
        layout.addWidget(self.trend_chart_view, 0, 1)
        layout.addWidget(QLabel("Top 5 Largest Expenses"), 1, 0, 1, 2)
        layout.addWidget(self.top_expenses_table, 2, 0, 1, 2)

        self.tabs.addTab(tab, "Visual Analytics")

    def _build_explorer_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        filters = QHBoxLayout()
        self.category_filter = QComboBox()
        self.category_filter.addItem("All")
        self.category_filter.currentTextChanged.connect(self.refresh_explorer)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search description…")
        self.search_input.textChanged.connect(self.refresh_explorer)
        override_btn = QPushButton("Override Category")
        override_btn.clicked.connect(self.override_category)

        filters.addWidget(QLabel("Category:"))
        filters.addWidget(self.category_filter)
        filters.addWidget(QLabel("Search:"))
        filters.addWidget(self.search_input)
        filters.addWidget(override_btn)
        layout.addLayout(filters)

        self.explorer_table = DataFrameTable()
        self.explorer_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.explorer_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.explorer_table)

        self.tabs.addTab(tab, "Transaction Explorer")

    def _build_budget_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        budget_group = QGroupBox("Budget Limits (AMD)")
        budget_layout = QGridLayout(budget_group)
        self.budget_spinboxes: dict[str, QSpinBox] = {}
        for idx, (cat, default) in enumerate(DEFAULT_BUDGETS.items()):
            spin = QSpinBox()
            spin.setRange(0, 10_000_000)
            spin.setSingleStep(1000)
            spin.setValue(default)
            spin.valueChanged.connect(self.refresh_budget)
            self.budget_spinboxes[cat] = spin
            row, col = divmod(idx, 3)
            budget_layout.addWidget(QLabel(cat), row * 2, col)
            budget_layout.addWidget(spin, row * 2 + 1, col)
        layout.addWidget(budget_group)

        self.budget_bars: dict[str, QProgressBar] = {}
        bars_group = QGroupBox("Budget Usage")
        bars_layout = QVBoxLayout(bars_group)
        for cat in DEFAULT_BUDGETS:
            row = QHBoxLayout()
            label = QLabel(cat)
            bar = QProgressBar()
            bar.setRange(0, 100)
            status = QLabel("")
            row.addWidget(label, 1)
            row.addWidget(bar, 4)
            row.addWidget(status, 2)
            bars_layout.addLayout(row)
            self.budget_bars[cat] = bar
            setattr(self, f"_budget_status_{cat}", status)
        layout.addWidget(bars_group)

        anomaly_controls = QHBoxLayout()
        anomaly_controls.addWidget(QLabel("Z-Score Threshold:"))
        self.anomaly_slider = QSlider(Qt.Orientation.Horizontal)
        self.anomaly_slider.setRange(15, 40)
        self.anomaly_slider.setValue(25)
        self.anomaly_label = QLabel("2.5")
        self.anomaly_slider.valueChanged.connect(
            lambda v: (self.anomaly_label.setText(f"{v / 10:.1f}"), self.refresh_anomalies())
        )
        anomaly_controls.addWidget(self.anomaly_slider)
        anomaly_controls.addWidget(self.anomaly_label)
        layout.addLayout(anomaly_controls)

        self.anomaly_table = DataFrameTable()
        layout.addWidget(QLabel("Detected Anomalies"))
        layout.addWidget(self.anomaly_table)

        self.tabs.addTab(tab, "Budget & Anomalies")

    def open_statement(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Bank Statement",
            str(BASE_DIR.parent / "Statements"),
            "Bank Statements (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)",
        )
        if not path:
            return

        bank_key = self.bank_combo.currentData()
        try:
            raw = self.loader.load(path, bank_key=bank_key)
            self.df = self.classifier.classify(raw)
            self._setup_date_filters()
            self.apply_filters()
            self.statusBar().showMessage(f"Loaded {len(self.df)} transactions from {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))

    def _setup_date_filters(self) -> None:
        if self.df is None or self.df.empty:
            return
        min_d = self.df["date"].min().date()
        max_d = self.df["date"].max().date()
        self.start_date.setEnabled(True)
        self.end_date.setEnabled(True)
        self.start_date.setDate(QDate(min_d.year, min_d.month, min_d.day))
        self.end_date.setDate(QDate(max_d.year, max_d.month, max_d.day))

    def apply_filters(self) -> None:
        if self.df is None:
            self.filtered_df = None
            self.refresh_all()
            return

        start = self.start_date.date().toPython()
        end = self.end_date.date().toPython()
        mask = (self.df["date"].dt.date >= start) & (self.df["date"].dt.date <= end)
        self.filtered_df = self.df.loc[mask].copy()
        self.refresh_all()

    def refresh_all(self) -> None:
        self.refresh_metrics()
        self.refresh_analytics()
        self.refresh_explorer_filters()
        self.refresh_explorer()
        self.refresh_budget()
        self.refresh_anomalies()

    def _active_df(self) -> pd.DataFrame:
        if self.filtered_df is not None:
            return self.filtered_df
        if self.df is not None:
            return self.df
        return pd.DataFrame()

    def refresh_metrics(self) -> None:
        kpis = compute_kpis(self._active_df())
        self.income_card.set_value(format_amd(kpis["total_income"]))
        self.expense_card.set_value(format_amd(kpis["total_expenses"]))
        self.savings_card.set_value(format_amd(kpis["net_savings"]))
        self.txn_card.set_value(str(kpis["transaction_count"]))

    def refresh_analytics(self) -> None:
        df = self._active_df()
        summary = get_category_summary(df)

        pie_series = QPieSeries()
        if not summary.empty:
            for _, row in summary.iterrows():
                pie_series.append(str(row["category"]), float(row["total_spend"]))
        pie_chart = QChart()
        pie_chart.addSeries(pie_series)
        pie_chart.setTitle("Expense Distribution by Category")
        pie_chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.pie_chart_view.setChart(pie_chart)

        trends = get_monthly_trends(df)
        bar_set = QBarSet("Spend")
        categories: list[str] = []
        if not trends.empty:
            for _, row in trends.iterrows():
                categories.append(str(row["period"])[:7])
                bar_set.append(float(row["total_spend"]))
        bar_series = QBarSeries()
        bar_series.append(bar_set)
        trend_chart = QChart()
        trend_chart.addSeries(bar_series)
        trend_chart.setTitle("Monthly Spending Trends")
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        trend_chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        bar_series.attachAxis(axis_x)
        axis_y = QValueAxis()
        trend_chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        bar_series.attachAxis(axis_y)
        self.trend_chart_view.setChart(trend_chart)

        expenses = df[df["transaction_type"] == "expense"] if not df.empty else df
        if not expenses.empty:
            top5 = expenses.nlargest(5, "amount")[["date", "description", "category", "amount"]]
        else:
            top5 = pd.DataFrame()
        self.top_expenses_table.load_dataframe(top5)

    def refresh_explorer_filters(self) -> None:
        df = self._active_df()
        current = self.category_filter.currentText()
        self.category_filter.blockSignals(True)
        self.category_filter.clear()
        self.category_filter.addItem("All")
        if not df.empty and "category" in df.columns:
            for cat in sorted(df["category"].dropna().unique()):
                self.category_filter.addItem(str(cat))
        idx = self.category_filter.findText(current)
        if idx >= 0:
            self.category_filter.setCurrentIndex(idx)
        self.category_filter.blockSignals(False)

    def refresh_explorer(self) -> None:
        df = self._active_df()
        if df.empty:
            self.explorer_table.load_dataframe(df)
            return

        filtered = df.copy()
        cat = self.category_filter.currentText()
        if cat != "All":
            filtered = filtered[filtered["category"] == cat]
        search = self.search_input.text().strip()
        if search:
            filtered = filtered[
                filtered["description"].str.contains(search, case=False, na=False)
                | filtered["cleaned_description"].str.contains(search, case=False, na=False)
            ]

        cols = [
            "transaction_id",
            "date",
            "description",
            "category",
            "amount",
            "transaction_type",
            "confidence",
            "method",
        ]
        self.explorer_table.load_dataframe(filtered[cols])

    def override_category(self) -> None:
        row = self.explorer_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Override", "Select a transaction row first.")
            return

        txn_id_item = self.explorer_table.item(row, 0)
        desc_item = self.explorer_table.item(row, 2)
        if txn_id_item is None:
            return

        txn_id = txn_id_item.text()
        dialog = OverrideDialog(
            txn_id,
            desc_item.text() if desc_item else "",
            self.classifier.get_categories(),
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        new_cat = dialog.selected_category()
        if self.df is None:
            return

        match = self.df[self.df["transaction_id"] == txn_id]
        if match.empty:
            return

        cleaned = match.iloc[0]["cleaned_description"]
        self.classifier.manual_override(txn_id, new_cat, cleaned)
        mask = self.df["transaction_id"] == txn_id
        self.df.loc[mask, "category"] = new_cat
        self.df.loc[mask, "method"] = "manual"
        self.df.loc[mask, "confidence"] = 1.0
        self.apply_filters()
        QMessageBox.information(self, "Override", f"Category updated to '{new_cat}'.")

    def refresh_budget(self) -> None:
        for cat, spin in self.budget_spinboxes.items():
            self.budgets[cat] = float(spin.value())

        report = check_budget_limits(self._active_df(), self.budgets)
        for cat in DEFAULT_BUDGETS:
            bar = self.budget_bars[cat]
            status: QLabel = getattr(self, f"_budget_status_{cat}")
            row = report[report["category"] == cat] if not report.empty else pd.DataFrame()
            if row.empty:
                bar.setValue(0)
                status.setText("No spend")
                continue
            pct = min(float(row.iloc[0]["pct_used"]), 100)
            bar.setValue(int(pct))
            actual = row.iloc[0]["actual"]
            budget = row.iloc[0]["budget"]
            status.setText(f"{actual:,.0f} / {budget:,.0f}")

    def refresh_anomalies(self) -> None:
        threshold = self.anomaly_slider.value() / 10.0
        anomalies = detect_anomalies(self._active_df(), threshold_std=threshold)
        if anomalies.empty:
            self.anomaly_table.load_dataframe(pd.DataFrame())
        else:
            display = anomalies[["date", "description", "category", "amount", "z_score"]]
            self.anomaly_table.load_dataframe(display)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = BankAnalyzerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
