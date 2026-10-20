"""Measure and tune the transaction classifier on a hand-labelled file.

Make the labelled file once:
  1. Load a statement in the app, export the Transaction Explorer as CSV
     (keep at least: description, category, transaction_type).
  2. Open it in Excel and correct the `category` column by hand.
  3. Save as data/labelled.csv (UTF-8).

Run:
    python evaluate_classifier.py data/labelled.csv
    python evaluate_classifier.py data/labelled.csv --sweep   # tune thresholds (a few minutes)

Uses k-fold cross-validation: for each fold the model learns ONLY from the
training folds (as if those were earlier statements), then predicts the held-
out fold. Nothing is written to config/ or data/training_data.csv.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from modules.classification_handlers import RuleMatcher
from modules.classifier import TransactionClassifier
from modules.i18n import LocaleRepository
from modules.ml_model import LearnedCategoryModel
from modules.models import ClassificationMethod
from modules.rules import InMemoryRuleRepository, JsonRuleRepository
from modules.text_utils import ml_text

BASE = Path(__file__).resolve().parent
RULES = BASE / "config" / "category_rules.json"


COLUMN_ALIASES = {
    "description": "description", "նկարագրություն": "description",
    "details": "description", "cleaned_description": "description",
    "category": "category", "true_category": "category", "կատեգորիա": "category",
    "transaction_type": "transaction_type", "type": "transaction_type",
}


def load_labelled(path: Path) -> pd.DataFrame:
    """Read CSV or Excel; auto-detect ',' vs ';' and encoding; normalise headers."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = None
        for enc in ("utf-8-sig", "cp1251", "cp1252"):
            try:
                # sep=None sniffs the delimiter (Excel on Windows often saves ';')
                df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        if df is None:
            raise SystemExit(f"Could not read {path} - save it as 'CSV UTF-8' or .xlsx.")

    df.columns = [str(c).strip().lstrip("﻿").lower() for c in df.columns]
    # Exports from the app use the UI language: map headers and values back
    # to the English keys, whatever language the file was exported in.
    locales = LocaleRepository()
    headers = {**locales.reverse_lookup("columns"), **COLUMN_ALIASES}
    df = df.rename(columns={c: headers[c] for c in df.columns if c in headers})
    for col, section in (("category", "categories"), ("transaction_type", "transaction_types")):
        if col in df.columns:
            reverse = locales.reverse_lookup(section)
            df[col] = df[col].map(
                lambda v: reverse.get(str(v).strip().lower(), v) if pd.notna(v) else v
            )
    missing = {"description", "category"} - set(df.columns)
    if missing:
        raise SystemExit(
            f"Missing column(s) {sorted(missing)} in {path.name}.\n"
            f"Columns found: {list(df.columns)}\n"
            "The file needs a 'description' column (raw transaction text) and a "
            "'category' column (the correct label)."
        )
    return df


def make_classifier(tmp: Path, min_prob: float, min_margin: float) -> TransactionClassifier:
    # Fresh in-memory copy of the rules (never saved) and an empty training store.
    repository = InMemoryRuleRepository(JsonRuleRepository(RULES).load())
    model = LearnedCategoryModel(
        store_path=tmp / "train.csv", min_probability=min_prob, min_margin=min_margin
    )
    return TransactionClassifier(model=model, repository=repository)


def strip_rules_matching(clf: TransactionClassifier, texts: pd.Series) -> int:
    """Delete every keyword / merchant alias that fires on ``texts``.

    Simulates merchants the JSON has never heard of, so the test measures the
    ML fallback instead of the rules. Returns how many rule entries were removed.
    """
    cleaned = [clf.normalizer.clean(t) for t in texts]
    # The RuleSet notifies the matcher, fuzzy index and model after pruning.
    return clf.rules.prune(
        lambda alias: any(RuleMatcher.alias_matches(alias, t) for t in cleaned)
    )


def run_cv_multi(
    df: pd.DataFrame,
    folds: int,
    thresholds: list[tuple[float, float]],
    use_labels_for_training: bool,
    unseen: bool = False,
) -> dict[tuple[float, float], pd.DataFrame]:
    """Cross-validate once, scoring several (min_probability, min_margin) pairs.

    Thresholds only affect prediction, so each fold's model is trained once and
    re-used for every pair - this keeps --sweep about as fast as a single run.
    """
    # Group folds by merchant (description without numbers/boilerplate) so the
    # same merchant never sits in both train and test - statements overlap, and
    # a random split would let the model "remember" the answer.
    groups = (
        df["description"].map(ml_text).str.replace(r"\d+", "", regex=True)
        .str.split().str[:4].str.join(" ")
    )
    uniq = groups.unique()
    rng = np.random.default_rng(42)
    group_fold = dict(zip(uniq, rng.integers(0, folds, len(uniq))))
    fold_id = groups.map(group_fold).to_numpy()

    parts: dict[tuple[float, float], list[pd.DataFrame]] = {t: [] for t in thresholds}
    for k in range(folds):
        train, test = df[fold_id != k], df[fold_id == k]
        if test.empty:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            clf = make_classifier(Path(tmp), *thresholds[0])
            if unseen:
                strip_rules_matching(clf, test["description"])
            # Learn from the "earlier statements": rule-labelled rows always,
            # plus true labels if simulating manual overrides.
            clf.learn_from_dataframe(clf.classify_dataframe(train), persist=False)
            if use_labels_for_training and clf.model is not None:
                clf.model.add_examples(
                    zip(train["description"], train["true_category"]),
                    source="manual", persist=False,
                )
                clf.model.fit()
            for p, m in thresholds:
                if clf.model is not None:
                    clf.model.min_probability, clf.model.min_margin = p, m
                pred = clf.classify_dataframe(test)
                parts[(p, m)].append(pred.assign(true_category=test["true_category"].values))
    return {t: pd.concat(v, ignore_index=True) for t, v in parts.items()}


def run_cv(df: pd.DataFrame, folds: int, min_prob: float, min_margin: float,
           use_labels_for_training: bool, unseen: bool = False) -> pd.DataFrame:
    return run_cv_multi(
        df, folds, [(min_prob, min_margin)], use_labels_for_training, unseen
    )[(min_prob, min_margin)]


def report(res: pd.DataFrame, title: str) -> None:
    res = res.assign(correct=res["category"] == res["true_category"])
    print(f"\n=== {title} ===")
    print(f"rows: {len(res)}   accuracy: {res['correct'].mean():.1%}   "
          f"'Other' rate: {(res['category'] == 'Other').mean():.1%}")
    by = res.groupby("method").agg(rows=("correct", "size"), accuracy=("correct", "mean"))
    by["share"] = by["rows"] / len(res)
    print(by.sort_values("rows", ascending=False).to_string(
        formatters={"accuracy": "{:.1%}".format, "share": "{:.1%}".format}))
    wrong = res[~res["correct"] & (res["category"] != "Other")]
    if not wrong.empty:
        print("\nSample confident mistakes (fix rules or relabel):")
        print(wrong[["description", "true_category", "category", "method", "confidence"]]
              .head(10).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labelled_csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sweep", action="store_true", help="grid-search model thresholds")
    args = ap.parse_args()

    df = load_labelled(Path(args.labelled_csv))
    df = df.rename(columns={"category": "true_category"})
    df["true_category"] = df["true_category"].fillna("").astype(str).str.strip()
    unlabelled = (df["true_category"] == "") | (df["true_category"].str.lower() == "nan")
    if unlabelled.any():
        print(f"Skipping {int(unlabelled.sum())} rows with an empty category.")
    df = df[~unlabelled & df["description"].notna()].reset_index(drop=True)
    df["description"] = df["description"].astype(str)
    df["true_category"] = df["true_category"].astype(str).str.strip()
    print(f"Loaded {len(df)} labelled rows, {df['true_category'].nunique()} categories.")
    known = set(json.loads(RULES.read_text(encoding="utf-8"))) | {"Other"}
    unknown = df.loc[~df["true_category"].isin(known), "true_category"].value_counts()
    if not unknown.empty:
        print("\nWARNING: these labels are not categories in category_rules.json, so the")
        print("classifier can never predict them (they count as errors):")
        for name, n in unknown.items():
            print(f"  {name!r}: {n} rows")
        print(f"Valid names: {sorted(known)}\n")
    texts = df["description"].str.strip()
    top = texts.value_counts()
    if len(df) and top.iloc[0] / len(df) > 0.3:
        print(f"WARNING: {top.iloc[0]} of {len(df)} rows have the identical description "
              f"{top.index[0]!r}. Text this generic can't be classified - check that the "
              "statement's merchant/details column is being read.\n")
    if "transaction_type" not in df:
        df["transaction_type"] = "expense"

    report(run_cv(df, args.folds, 0.25, 0.20, False),
           "Rules + self-training (no manual labels)")
    report(run_cv(df, args.folds, 0.25, 0.20, True),
           "Plus manual corrections from earlier statements")

    unseen = run_cv(df, args.folds, 0.25, 0.20, True, unseen=True)
    report(unseen, "UNSEEN MERCHANTS: rules for the test rows deleted, "
                   "model trained on your other corrected statements")

    if args.sweep:
        print("\n=== Threshold sweep (unseen merchants) ===")
        print("Thresholds only matter for the ML step, so they are tuned on the")
        print("unseen-merchant test. Higher precision = fewer wrong guesses;")
        print("higher 'other_rate' = more rows left for you to label.\n")
        grid = [(p, m) for p in (0.25, 0.35, 0.45, 0.60) for m in (0.0, 0.10, 0.20)]
        results = run_cv_multi(df, args.folds, grid, True, unseen=True)
        rows = []
        for (p, m), r in results.items():
            ml = r[r["method"] == ClassificationMethod.ML_MODEL.value]
            rows.append({
                "min_probability": p, "min_margin": m,
                "accuracy": (r["category"] == r["true_category"]).mean(),
                "other_rate": (r["category"] == "Other").mean(),
                "ml_precision": (ml["category"] == ml["true_category"]).mean()
                if len(ml) else float("nan"),
            })
        out = pd.DataFrame(rows)
        # Accuracy counts 'Other' as wrong, so it rewards guessing. Prefer the
        # setting with the best accuracy among those whose ML guesses are >=90%
        # precise; fall back to plain accuracy if none qualify.
        good = out[out["ml_precision"] >= 0.90]
        out = (good if not good.empty else out).sort_values(
            ["accuracy", "ml_precision"], ascending=False)
        print(out.head(10).to_string(index=False, float_format="{:.3f}".format))
        best = out.iloc[0]
        print(f"\nBest: LearnedCategoryModel(min_probability={best.min_probability}, "
              f"min_margin={best.min_margin})")


if __name__ == "__main__":
    main()
