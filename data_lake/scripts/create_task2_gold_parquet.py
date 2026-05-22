import argparse

import pandas as pd


def parse_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if pd.isna(v):
            return None
        return bool(int(v))
    s = str(v).strip().lower()
    if not s or s == "nan":
        return None
    if s in {"1", "true", "t", "yes", "y", "si", "sí"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver_parquet", required=True, help="Silver parquet (ya binarizado) de Task2")
    ap.add_argument("--labeled_csv", required=True, help="CSV con salida de revision/anotacion (debe tener fragment_id,gold_is_contribution)")
    ap.add_argument("--out_parquet", required=True, help="Output gold parquet (mismas columnas que silver)")
    ap.add_argument(
        "--keep_unlabeled",
        action="store_true",
        help="Si está activo, mantiene filas sin gold_is_contribution (usa el valor silver). Por defecto solo guarda filas con gold.",
    )
    args = ap.parse_args()

    silver = pd.read_parquet(args.silver_parquet)
    if "fragment_id" not in silver.columns:
        raise ValueError("Silver parquet debe contener columna 'fragment_id'.")
    if "is_contribution" not in silver.columns:
        raise ValueError("Silver parquet debe contener columna 'is_contribution'.")

    ann = pd.read_csv(args.labeled_csv)
    required = {"fragment_id", "gold_is_contribution"}
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(f"CSV labeled missing required columns: {sorted(missing)}")

    silver = silver.copy()
    ann = ann.copy()
    silver["fragment_id"] = silver["fragment_id"].astype(str)
    ann["fragment_id"] = ann["fragment_id"].astype(str)

    ann["gold_is_contribution_parsed"] = ann["gold_is_contribution"].apply(parse_bool)

    if not args.keep_unlabeled:
        ann = ann[ann["gold_is_contribution_parsed"].notna()].copy()

    gold_ids = set(ann["fragment_id"].tolist())
    gold = silver[silver["fragment_id"].isin(gold_ids)].copy()

    # Map gold labels to silver rows
    gold_map = dict(zip(ann["fragment_id"], ann["gold_is_contribution_parsed"]))
    gold["is_contribution"] = gold["fragment_id"].map(gold_map).fillna(gold["is_contribution"]).astype(bool)

    # Enforce exact same columns & order as silver
    gold = gold[silver.columns.tolist()]

    gold.to_parquet(args.out_parquet, index=False)

    counts = gold["is_contribution"].value_counts(dropna=False).to_dict()
    print("Gold parquet created")
    print("Saved:", args.out_parquet)
    print("Rows:", len(gold))
    print("Binary counts (is_contribution):", counts)
    print("Columns:", gold.columns.tolist())


if __name__ == "__main__":
    main()
