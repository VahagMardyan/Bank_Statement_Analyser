"""Streamlit dashboard for Bank Statement Analyzer."""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

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
    "Cafes and Restaurants": 40000,
    "Utilities": 30000,
    "Entertainment": 20000,
    "Shopping": 50000,
}


def convert_df_to_csv_bytes(
    df: pd.DataFrame, columns: list[str] | None = None
) -> bytes:
    """Converts DataFrame to UTF-8-SIG encoded CSV bytes for specified or all columns."""
    target_df = df[columns] if columns is not None else df
    return target_df.to_csv(index=False).encode("utf-8-sig")


def convert_df_to_excel_bytes(
    df: pd.DataFrame, columns: list[str] | None = None
) -> bytes:
    """Converts DataFrame to Excel (.xlsx) bytes using openpyxl for specified or all columns."""
    target_df = df[columns] if columns is not None else df
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        target_df.to_excel(writer, index=False, sheet_name="Transactions")
    return output.getvalue()


def init_session_state() -> None:
    defaults = {
        "df": None,
        "classifier": TransactionClassifier(),
        "loader": BankStatementLoader(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_date_filter(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    return df.loc[mask].copy()


def render_metrics(kpis: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Income", f"{kpis['total_income']:,.0f} AMD")
    c2.metric("Total Expenses", f"{kpis['total_expenses']:,.0f} AMD")
    c3.metric("Net Savings", f"{kpis['net_savings']:,.0f} AMD")
    c4.metric("Transactions", kpis["transaction_count"])


def tab_visual_analytics(df: pd.DataFrame) -> None:
    expenses = df[df["transaction_type"] == "expense"]
    summary = get_category_summary(df)

    col1, col2 = st.columns(2)
    with col1:
        if not summary.empty:
            fig_pie = px.pie(
                summary,
                names="category",
                values="total_spend",
                hole=0.45,
                title="Expense Distribution by Category",
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No expense data to display.")

    with col2:
        trends = get_monthly_trends(df)
        if not trends.empty:
            fig_line = px.bar(
                trends,
                x="period",
                y="total_spend",
                title="Monthly Spending Trends",
                labels={"total_spend": "Spend (AMD)", "period": "Month"},
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No trend data available.")

    st.subheader("Top 5 Largest Expenses")
    if not expenses.empty:
        top5 = expenses.nlargest(5, "amount")[
            ["date", "description", "category", "amount"]
        ].copy()
        top5["date"] = top5["date"].dt.strftime("%Y-%m-%d, %H:%M")
        st.dataframe(top5, use_container_width=True, hide_index=True)
    else:
        st.info("No expenses found.")


def tab_transaction_explorer(df: pd.DataFrame) -> None:
    categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_cat = st.selectbox("Filter by Category", categories)
    with col2:
        search = st.text_input("Search Description", "")

    filtered = df.copy()
    if selected_cat != "All":
        filtered = filtered[filtered["category"] == selected_cat]
    if search.strip():
        filtered = filtered[
            filtered["description"].str.contains(search, case=False, na=False)
            | filtered["cleaned_description"].str.contains(search, case=False, na=False)
        ]

    available_cols = [
        "transaction_id",
        "date",
        "description",
        "category",
        "amount",
        "transaction_type",
        "confidence",
        "method",
    ]

    selected_cols = st.multiselect(
        "Select Columns to Display & Export",
        options=available_cols,
        default=available_cols[0 : len(available_cols)-2], # "confidence" and "method" are not selected by default
    )

    if not selected_cols:
        st.warning("Please select at least one column to display and export.")
        return

    export_df = filtered.copy()
    if "date" in export_df.columns:
        export_df["date"] = export_df["date"].dt.strftime("%Y-%m-%d, %H:%M")

    st.dataframe(
        export_df[selected_cols],
        use_container_width=True,
        hide_index=True,
    )

    if not export_df.empty:
        col_fmt, col_btn = st.columns([1, 2], vertical_alignment="bottom")
        with col_fmt:
            export_format = st.selectbox("Export Format", ["Excel (.xlsx)", "CSV"])
        with col_btn:
            st.write("")  # Alignment spacing
            if export_format == "CSV":
                file_bytes = convert_df_to_csv_bytes(export_df, columns=selected_cols)
                file_name = "transactions_export.csv"
                mime = "text/csv"
            else:
                file_bytes = convert_df_to_excel_bytes(export_df, columns=selected_cols)
                file_name = "transactions_export.xlsx"
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

            st.download_button(
                label=f"📥 Download as {export_format}",
                data=file_bytes,
                file_name=file_name,
                mime=mime,
                use_container_width=True,
            )

    st.divider()
    st.subheader("Manual Category Override")
    if not filtered.empty:
        txn_id = st.selectbox(
            "Transaction ID",
            filtered["transaction_id"].tolist(),
            format_func=lambda x: f"{x} - {filtered.loc[filtered['transaction_id'] == x, 'description'].iloc[0][:50]}",
        )
        new_category = st.selectbox(
            "New Category",
            st.session_state.classifier.get_categories(),
        )
        if st.button("Apply Override"):
            row = filtered.loc[filtered["transaction_id"] == txn_id].iloc[0]
            st.session_state.classifier.manual_override(
                transaction_id=txn_id,
                new_category=new_category,
                cleaned_description=row["cleaned_description"],
            )
            mask = st.session_state.df["transaction_id"] == txn_id
            st.session_state.df.loc[mask, "category"] = new_category
            st.session_state.df.loc[mask, "method"] = "manual"
            st.session_state.df.loc[mask, "confidence"] = 1.0
            st.success(f"Category updated to '{new_category}'. Rules/dataset updated for retraining.")
            st.rerun()


def tab_budget_anomalies(df: pd.DataFrame) -> None:
    st.subheader("Budget Limits")
    budget_input: dict[str, float] = {}
    cols = st.columns(3)
    for idx, (cat, default) in enumerate(DEFAULT_BUDGETS.items()):
        with cols[idx % 3]:
            budget_input[cat] = st.number_input(
                f"{cat} Budget (AMD)",
                min_value=0,
                value=default,
                step=1000,
                key=f"budget_{cat}",
            )

    budget_report = check_budget_limits(df, budget_input)
    if not budget_report.empty:
        for _, row in budget_report.iterrows():
            pct = min(row["pct_used"], 100)
            st.write(f"**{row['category']}** — {row['actual']:,.0f} / {row['budget']:,.0f} AMD")
            st.progress(min(pct / 100, 1.0))
            if row["over_budget"]:
                st.warning(f"Over budget by {abs(row['remaining']):,.0f} AMD")

        st.dataframe(budget_report, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Anomaly Detection")
    threshold = st.slider("Z-Score Threshold", 1.5, 4.0, 2.5, 0.1)
    anomalies = detect_anomalies(df, threshold_std=threshold)
    if anomalies.empty:
        st.info("No anomalies detected at this threshold.")
    else:
        display = anomalies[
            ["date", "description", "category", "amount", "z_score"]
        ].copy()
        display["date"] = display["date"].dt.strftime("%Y-%m-%d, %H:%M")
        st.dataframe(display, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Bank Statement Analyzer",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    st.title("🏦 Bank Statement Analyzer")
    st.caption("Parse, categorize, and visualize bank statements from Armenian and international banks.")

    with st.sidebar:
        st.header("Configuration")
        uploaded = st.file_uploader(
            "Upload Bank Statement (CSV / Excel / PDF)",
            type=["csv", "xlsx", "xls", "pdf"],
        )
        loader: BankStatementLoader = st.session_state.loader
        banks = ["auto_detect"] + loader.list_banks()
        bank_labels = {
            "auto_detect": "Auto Detect",
            **{k: loader._configs[k]["display_name"] for k in loader.list_banks()},
        }
        selected_bank = st.selectbox(
            "Bank",
            banks,
            format_func=lambda x: bank_labels.get(x, x),
        )

        if uploaded is not None:
            if st.button("Load & Analyze", type="primary"):
                file_ext = Path(uploaded.name).suffix.lower()
                tmp_path = BASE_DIR / "data" / f"_upload{file_ext}"
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_bytes(uploaded.getvalue())
                try:
                    raw = loader.load(tmp_path, bank_key=selected_bank)
                    classified = st.session_state.classifier.classify(raw)
                    st.session_state.df = classified
                    st.success(f"Loaded {len(classified)} transactions.")
                except Exception as exc:
                    st.error(f"Failed to load statement: {exc}")

        df: pd.DataFrame | None = st.session_state.df
        if df is not None and not df.empty:
            min_date = df["date"].min().date()
            max_date = df["date"].max().date()
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                df = apply_date_filter(df, date_range[0], date_range[1])
        else:
            df = None

    if df is None or df.empty:
        st.info("Upload a statement file (CSV, XLSX, XLS, PDF) using the sidebar to get started.")
        st.markdown(
            """
            **Supported features:**
            - Multi-bank parsing (Ameriabank, ACBA, Inecobank, Evocabank, auto-detect)
            - Supports CSV, XLSX, XLS, and PDF formats
            - Hybrid rule + ML categorization
            - Interactive charts, budget tracking, and export functionality
            """
        )
        return

    render_metrics(compute_kpis(df))

    tab1, tab2, tab3 = st.tabs(["Visual Analytics", "Transaction Explorer", "Budget & Anomalies"])
    with tab1:
        tab_visual_analytics(df)
    with tab2:
        tab_transaction_explorer(df)
    with tab3:
        tab_budget_anomalies(df)


if __name__ == "__main__":
    main()