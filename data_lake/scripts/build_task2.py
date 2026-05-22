from __future__ import annotations

import argparse
import random
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REQUIRED_SILVER_COLS = ["chunk_id", "doc_id", "source_path", "label", "heading", "text"]


@dataclass(frozen=True)
class BuildConfig:
    silver_parquet: Path
    out_parquet: Path
    n_pos: int
    n_neg: int
    neg_min_words: int
    neg_max_words: int
    min_pos_score: int
    seed: int


def compile_contribution_patterns() -> tuple[list[re.Pattern], list[re.Pattern]]:
    """
    Return (strong_patterns, weak_patterns) for contribution detection.
    """
    strong_patterns = [
        r"\b(en\s+este\s+trabajo\s+)?(presentamos|proponemos|introducimos|planteamos)\b",
        r"\b(nuestra|nuestro)\s+(propuesta|aporte|contribuci[oó]n)\b",
        r"\b(este\s+trabajo|este\s+art[ií]culo)\s+(propone|presenta|introduce)\b",
        r"\b(aportamos|contribuimos)\b",
        r"\b(a\s+diferencia\s+de)\b",
        r"\b(primer(a)?\s+vez|por\s+primera\s+vez)\b",
        r"\b(ponemos\s+a\s+disposici[oó]n|liberamos|publicamos)\b",
    ]
    weak_patterns = [
        r"\b(nuevo|nueva)\s+(m[eé]todo|enfoque|algoritmo|sistema|marco|modelo|recurso|corpus|dataset)\b",
        r"\b(mejora(mos)?|supera(mos)?|incrementa(mos)?)\b",
        r"\b(prototipo|framework|pipeline)\b",
    ]
    return (
        [re.compile(p, flags=re.IGNORECASE) for p in strong_patterns],
        [re.compile(p, flags=re.IGNORECASE) for p in weak_patterns],
    )


def contribution_score(text: str, heading: str, rhetorical_label: str, strong: list[re.Pattern], weak: list[re.Pattern]) -> int:
    t = (text or "").strip()
    if not t:
        return 0

    score = 0
    rlabel = (rhetorical_label or "").strip().upper()
    heading_l = (heading or "").strip().lower()

    strong_hits = sum(1 for rx in strong if rx.search(t))
    weak_hits = sum(1 for rx in weak if rx.search(t))
    score += 3 * strong_hits
    score += 1 * weak_hits

    if "contribuci" in heading_l or "aporte" in heading_l:
        score += 2

    if rlabel == "CONTR":
        score += 2
    elif rlabel in {"METH", "RESU", "DISC"}:
        score += 1

    return score


def looks_like_contribution(
    text: str,
    heading: str,
    rhetorical_label: str,
    strong: list[re.Pattern],
    weak: list[re.Pattern],
    min_pos_score: int,
) -> tuple[bool, int]:
    sc = contribution_score(text, heading, rhetorical_label, strong, weak)
    return (sc >= min_pos_score), sc


def validate_silver_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_SILVER_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Silver parquet missing columns: {missing}. Required: {REQUIRED_SILVER_COLS}")


def sample_diverse(df: pd.DataFrame, n: int, seed: int, key_col: str = "doc_id") -> pd.DataFrame:
    if df.empty or n <= 0:
        return df.iloc[0:0].copy()

    shuffled = df.sample(frac=1, random_state=seed)
    first_per_doc = shuffled.drop_duplicates(subset=[key_col], keep="first")
    picked = first_per_doc.head(n)
    if len(picked) >= n:
        return picked

    remaining = shuffled[~shuffled.index.isin(picked.index)]
    extra = remaining.head(n - len(picked))
    return pd.concat([picked, extra], axis=0)


def make_negative_window(text: str, min_words: int, max_words: int, seed: int, max_tries: int = 5) -> tuple[str, int] | None:
    words = (text or "").split()
    if len(words) < min_words:
        return None

    rng = random.Random(seed ^ len(words))
    for _ in range(max_tries):
        target_len = rng.randint(min_words, min(max_words, len(words)))
        start = rng.randint(0, max(0, len(words) - target_len))
        window_words = words[start : start + target_len]
        window_text = " ".join(window_words).strip()
        if window_text:
            return window_text, len(window_words)
    return None


def build_task2(cfg: BuildConfig) -> None:
    cfg.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(cfg.silver_parquet)
    validate_silver_df(df)

    df = df.copy()
    df["chunk_id"] = df["chunk_id"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    df["source_path"] = df["source_path"].astype(str)
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df["heading"] = df["heading"].astype(str)
    df["text"] = df["text"].astype(str)
    if "n_words" not in df.columns or df["n_words"].isna().any():
        df["n_words"] = df["text"].str.split().str.len()

    strong, weak = compile_contribution_patterns()
    scored = df.apply(
        lambda r: looks_like_contribution(
            text=str(r["text"]),
            heading=str(r.get("heading", "")),
            rhetorical_label=str(r.get("label", "")),
            strong=strong,
            weak=weak,
            min_pos_score=cfg.min_pos_score,
        ),
        axis=1,
        result_type="expand",
    )
    df["is_contribution_match"] = scored[0].astype(bool)
    df["contrib_score"] = scored[1].astype(int)

    # Positives
    pos_strict = df[df["is_contribution_match"]].sort_values("contrib_score", ascending=False).copy()
    pos_strict["pos_reason"] = "pattern_or_score"
    pos_picked = sample_diverse(pos_strict, cfg.n_pos, cfg.seed)

    if len(pos_picked) < cfg.n_pos:
        remaining_n = cfg.n_pos - len(pos_picked)
        already = set(pos_picked["chunk_id"].tolist())
        pos_relaxed = df[(df["label"] == "CONTR") & (~df["chunk_id"].isin(already))].copy()
        pos_relaxed = pos_relaxed.sort_values("contrib_score", ascending=False)
        pos_relaxed["pos_reason"] = "label_CONTR"
        pos_fill = sample_diverse(pos_relaxed, remaining_n, cfg.seed + 1)
        pos_picked = pd.concat([pos_picked, pos_fill], axis=0)

    pos_picked = pos_picked.head(cfg.n_pos).copy()
    if len(pos_picked) < cfg.n_pos:
        raise RuntimeError(
            f"No hay suficientes positivos. Se pidieron {cfg.n_pos} y se obtuvieron {len(pos_picked)}. "
            "Baja --min-pos-score o reduce --n-pos."
        )

    # Negatives
    pos_ids = set(pos_picked["chunk_id"].tolist())
    neg_candidates = df[~df["chunk_id"].isin(pos_ids)].copy()
    neg_candidates = neg_candidates[~neg_candidates["is_contribution_match"]].copy()
    neg_candidates = neg_candidates[neg_candidates["label"].isin(["BACK", "INTRO", "LIM", "CONC", "DISC", "RESU"])].copy()
    neg_candidates = neg_candidates[neg_candidates["n_words"] >= cfg.neg_min_words].copy()
    neg_candidates = neg_candidates.sample(frac=1, random_state=cfg.seed + 7)

    neg_rows = []
    neg_seen = 0
    for _, r in tqdm(neg_candidates.iterrows(), total=len(neg_candidates), desc="Selecting negatives"):
        if neg_seen >= cfg.n_neg:
            break

        raw_text = str(r["text"])
        n_words = int(r["n_words"])
        if cfg.neg_min_words <= n_words <= cfg.neg_max_words:
            window_text = raw_text
            window_n = n_words
        else:
            window = make_negative_window(
                raw_text,
                min_words=cfg.neg_min_words,
                max_words=min(cfg.neg_max_words, n_words),
                seed=cfg.seed + neg_seen,
            )
            if window is None:
                continue
            window_text, window_n = window

        if looks_like_contribution(
            text=window_text,
            heading=str(r.get("heading", "")),
            rhetorical_label=str(r.get("label", "")),
            strong=strong,
            weak=weak,
            min_pos_score=cfg.min_pos_score,
        )[0]:
            continue

        neg_rows.append(
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "source_path": r["source_path"],
                "label": r["label"],
                "heading": r["heading"],
                "text": window_text,
                "n_words": window_n,
                "contrib_score": int(r.get("contrib_score", 0)),
            }
        )
        neg_seen += 1

    neg_picked = pd.DataFrame(neg_rows)
    if len(neg_picked) < cfg.n_neg:
        raise RuntimeError(
            f"No hay suficientes negativos. Se pidieron {cfg.n_neg} y se obtuvieron {len(neg_picked)}. "
            "Ajusta --neg-min-words/--neg-max-words o reduce --n-neg."
        )

    def to_rows(frame: pd.DataFrame, is_pos: bool) -> list[dict]:
        rows: list[dict] = []
        for _, r in frame.iterrows():
            rows.append(
                {
                    "fragment_id": str(uuid.uuid4()),
                    "source_chunk_id": str(r["chunk_id"]),
                    "doc_id": str(r["doc_id"]),
                    "source_path": str(r["source_path"]),
                    "heading": str(r["heading"]),
                    "rhetorical_label": str(r["label"]),
                    "is_contribution": bool(is_pos),
                    "n_words": int(r["n_words"]),
                    "heuristic_score": int(r.get("contrib_score", 0)),
                    "text": str(r["text"]),
                }
            )
        return rows

    out_rows = []
    out_rows.extend(to_rows(pos_picked, True))
    out_rows.extend(to_rows(neg_picked, False))

    out_df = pd.DataFrame(out_rows).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    out_df.to_parquet(cfg.out_parquet, index=False)

    print("Saved:", cfg.out_parquet)
    print("Positives:", int(out_df["is_contribution"].sum()))
    print("Negatives:", int((~out_df["is_contribution"]).sum()))
    print("Total:", len(out_df))
    print("Positive reasons:", dict(pos_picked.get("pos_reason", pd.Series(dtype=str)).value_counts()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--silver-parquet",
        default="data_lake/datasets/task1_silver_final_train_no_golden.parquet",
        help="Task1 silver parquet with chunk_id/doc_id/source_path/label/heading/text",
    )
    ap.add_argument(
        "--out",
        default="data_lake/datasets/task2_contributions_silver.parquet",
        help="Output parquet for Task2 silver",
    )
    ap.add_argument("--n-pos", type=int, default=4000, help="Number of positive fragments")
    ap.add_argument("--n-neg", type=int, default=4000, help="Number of negative fragments")
    ap.add_argument("--neg-min-words", type=int, default=250)
    ap.add_argument("--neg-max-words", type=int, default=500)
    ap.add_argument("--min-pos-score", type=int, default=3, help="Minimum heuristic score to mark positive")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = BuildConfig(
        silver_parquet=Path(args.silver_parquet),
        out_parquet=Path(args.out),
        n_pos=int(args.n_pos),
        n_neg=int(args.n_neg),
        neg_min_words=int(args.neg_min_words),
        neg_max_words=int(args.neg_max_words),
        min_pos_score=int(args.min_pos_score),
        seed=int(args.seed),
    )
    build_task2(cfg)


if __name__ == "__main__":
    main()

