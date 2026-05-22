import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="CSV output from assisted labeling")
    ap.add_argument("--out_csv", required=True, help="Cleaned CSV without missing rows")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # Normalize columns if present
    if "gold_label" not in df.columns:
        raise ValueError(f"Missing column gold_label. Found: {df.columns.tolist()}")
    if "notes" not in df.columns:
        # if notes doesn't exist, treat as empty
        df["notes"] = ""

    # Drop rows with missing outputs
    mask_bad = (
        df["gold_label"].isna()
        | (df["gold_label"].astype(str).str.strip() == "")
        | (df["notes"].astype(str).str.strip() == "MISSING_FROM_MODEL_OUTPUT")
    )

    removed = int(mask_bad.sum())
    kept = int((~mask_bad).sum())

    df_clean = df[~mask_bad].copy()
    df_clean.to_csv(args.out_csv, index=False)

    print(f"Removed rows: {removed}")
    print(f"Kept rows:    {kept}")
    print(f"Saved: {args.out_csv}")

if __name__ == "__main__":
    main()
