from __future__ import annotations

import re
import unicodedata
import uuid
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

LABELS = ["INTRO", "BACK", "METH", "RESU", "DISC", "CONTR", "LIM", "CONC"]
SKIP_LABEL = "__SKIP__"

# Skip sections that are out-of-scope for rhetorical labeling.
SKIP_HEADING_RE = re.compile(
    r"\b("
    r"referencias?|bibliografia|anexos?|apendice|appendix|annex|"
    r"agradecimientos?|acknowledg(e)?ments?|funding|financiacion|"
    r"declaracion de datos|disponibilidad de datos|conflicto de interes|ethics?|"
    r"material suplementario|supplementary material"
    r")\b",
    re.IGNORECASE,
)

# Heading patterns by label. Anchored to heading start to avoid
# matching random in-text words like "aporte" or "limitacion".
HEADING_PATTERNS = [
    (re.compile(r"^(introduccion|introduction|planteamiento del problema|objetivos?)\b", re.IGNORECASE), "INTRO"),
    (re.compile(r"^(antecedentes|marco teorico|estado del arte|trabajos relacionados|related work|literature review)\b", re.IGNORECASE), "BACK"),
    (re.compile(r"^(metodologia|materiales y metodos|metodos|methodology|methods?|experimental setup)\b", re.IGNORECASE), "METH"),
    (re.compile(r"^(resultados?|results?)\b", re.IGNORECASE), "RESU"),
    (re.compile(r"^(discusion|discussion|analisis de resultados?)\b", re.IGNORECASE), "DISC"),
    (re.compile(r"^(contribucion(es)?|aportes?|contribution(s)?|principal contributions?)\b", re.IGNORECASE), "CONTR"),
    (re.compile(r"^(limitacion(es)?|amenazas? a la validez|threats? to validity|limitations?)\b", re.IGNORECASE), "LIM"),
    (re.compile(r"^(conclusion(es)?|concluding remarks|future work|trabajo futuro)\b", re.IGNORECASE), "CONC"),
]

# Numbered heading forms:
# 1. Introduccion
# 2) Related Work
# III - Metodologia
NUMBERED_HEADING = re.compile(r"^\s*((\d+(\.\d+)*)|([ivxlcdm]+))\s*[)\.\-:]?\s*(.+?)\s*$", re.IGNORECASE)

SHORT_HEADING_MAX_WORDS = 12
SHORT_HEADING_MAX_CHARS = 160
SENTENCE_VERB_RE = re.compile(
    r"\b("
    r"es|son|fue|fueron|sera|seran|hay|hubo|tiene|tienen|presenta|presentan|"
    r"proponemos|proponen|analiza|analizan|utiliza|utilizan|incluye|incluyen|"
    r"debe|deben|puede|pueden|permite|permiten"
    r")\b",
    re.IGNORECASE,
)

# Chunking config
MIN_WORDS = 250
MAX_WORDS = 1000
TARGET_WORDS = 600
MIN_TAIL_WORDS = 80
MIN_SECTION_WORDS = 50

SCIENTIFIC_CUE_RE = re.compile(
    r"\b("
    r"investig|estudio|trabajo|articulo|paper|research|study|metod|modelo|propuesta|"
    r"sistema|enfoque|analisis|experimento|resultad|discusion|conclusion|cient"
    r")\w*\b",
    re.IGNORECASE,
)
CONTR_SENTENCE_RE = re.compile(
    r"\b("
    r"no|pretende|ayudara|ayudar[aá]|permite|permitira|permitir[aá]|"
    r"analiza|describe|explica|muestra|demuestra|presenta|proponemos|propone"
    r")\b",
    re.IGNORECASE,
)


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def looks_like_heading(raw: str) -> bool:
    if not raw:
        return False

    words = raw.split()
    if len(words) > SHORT_HEADING_MAX_WORDS:
        return False
    if len(raw) > SHORT_HEADING_MAX_CHARS:
        return False

    # Long sentence-like lines are usually not headings.
    if raw.endswith(".") and len(words) > 4:
        return False
    if raw.count(";") > 0 or raw.count(",") > 0:
        return False
    if "?" in raw or "!" in raw:
        return False

    # If it's long and has verbs, it is probably a sentence.
    if len(words) >= 7 and SENTENCE_VERB_RE.search(raw):
        return False

    # Headings usually start with uppercase or numbering.
    has_numbering = bool(NUMBERED_HEADING.match(raw))
    first_alpha = next((ch for ch in raw if ch.isalpha()), "")
    if first_alpha and first_alpha.islower() and not has_numbering:
        return False

    return True


def canonical_heading(raw: str) -> str:
    m = NUMBERED_HEADING.match(raw)
    candidate = m.group(5) if m else raw
    candidate = normalize_line(candidate).strip(":- ")
    return candidate


def heading_label(line: str) -> tuple[str | None, str | None]:
    """
    Return (label, heading_text) if line looks like a heading, else (None, None).
    """
    raw = line.strip()
    if not raw:
        return None, None
    if not looks_like_heading(raw):
        return None, None

    heading_text = canonical_heading(raw)
    if not heading_text:
        return None, None

    norm = strip_accents(heading_text).lower()
    if SKIP_HEADING_RE.search(norm):
        return SKIP_LABEL, heading_text

    for rx, label in HEADING_PATTERNS:
        if rx.match(norm):
            if not is_label_plausible(norm, label):
                return None, None
            return label, heading_text

    return None, None


def is_label_plausible(norm_heading: str, label: str) -> bool:
    """
    Extra guardrails for noisy labels to reduce false positives.
    """
    if label == "CONTR":
        word_count = len(norm_heading.split())
        if word_count > 10:
            return False
        if CONTR_SENTENCE_RE.search(norm_heading):
            return False
        # "Contribucion..." style headings are generally fine.
        if norm_heading.startswith("contribucion") or norm_heading.startswith("contribution"):
            return True
        # "Aporte(s)..." headings are only accepted if they look academic.
        if norm_heading.startswith("aporte") or norm_heading.startswith("aportes"):
            return bool(SCIENTIFIC_CUE_RE.search(norm_heading))
        return False

    if label == "LIM":
        # Keep generic limitation headings.
        if norm_heading in {"limitaciones", "limitacion", "limitations", "limitation"}:
            return True
        # Otherwise require scientific context cue.
        return bool(SCIENTIFIC_CUE_RE.search(norm_heading))

    return True


def split_into_sections(text: str) -> list[dict]:
    """
    Split full doc text into labeled sections using detected headings.
    Returns list of {label, heading, section_text}.
    Unlabeled text before first heading is ignored.
    """
    lines = text.split("\n")
    sections: list[dict] = []
    current = None

    for line in lines:
        norm = normalize_line(line)
        label, heading = heading_label(norm)

        if label == SKIP_LABEL:
            if current and current["lines"]:
                current["section_text"] = "\n".join(current["lines"]).strip()
                sections.append(current)
            current = None
            continue

        if label:
            if current and current["lines"]:
                current["section_text"] = "\n".join(current["lines"]).strip()
                sections.append(current)
            current = {"label": label, "heading": heading, "lines": []}
            continue

        if current is not None and norm:
            current["lines"].append(norm)

    if current and current["lines"]:
        current["section_text"] = "\n".join(current["lines"]).strip()
        sections.append(current)

    cleaned = []
    for s in sections:
        if len(s["section_text"].split()) >= MIN_SECTION_WORDS:
            cleaned.append(s)
    return cleaned


def chunk_words(words: list[str], min_w: int, max_w: int) -> list[list[str]]:
    """
    Chunk list of words into [min_w, max_w] chunks when possible.
    Keeps medium tails instead of dropping all leftovers.
    """
    chunks: list[list[str]] = []
    i = 0
    n = len(words)

    while i < n:
        remaining = n - i
        if remaining <= max_w:
            if remaining >= min_w:
                chunks.append(words[i:])
            elif remaining >= MIN_TAIL_WORDS:
                if chunks and len(chunks[-1]) + remaining <= int(max_w * 1.25):
                    chunks[-1].extend(words[i:])
                else:
                    chunks.append(words[i:])
            break

        take = min(max_w, max(min_w, TARGET_WORDS))
        chunks.append(words[i : i + take])
        i += take

    return chunks


def iter_parquet_files(root: Path) -> Iterator[Path]:
    for p in sorted(root.rglob("*.parquet")):
        yield p


def build_task1(
    clean_parquet_root: str,
    out_path: str,
    per_label_cap: int = 5000,
    min_quality_score: int = 0,
    require_sections: bool = False,
    max_docs: int | None = None,
) -> None:
    """
    Build Task1 dataset from clean parquet shards.
    """
    root = Path(clean_parquet_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {lbl: 0 for lbl in LABELS}
    rows = []

    docs_seen = 0
    parquet_files = list(iter_parquet_files(root))
    for fp in tqdm(parquet_files, desc="Reading clean_parquet"):
        table = pq.read_table(fp)
        df = table.to_pandas()

        for _, r in df.iterrows():
            if max_docs is not None and docs_seen >= max_docs:
                break
            docs_seen += 1

            if "quality_score" in df.columns and int(r.get("quality_score", 0)) < min_quality_score:
                continue
            if require_sections and "has_sections" in df.columns and not bool(r.get("has_sections", False)):
                continue

            text = r["text"]
            doc_id = r.get("doc_id")
            path = r.get("path")

            sections = split_into_sections(text)
            if not sections:
                continue

            for s in sections:
                label = s["label"]
                if label not in counts:
                    continue
                if counts[label] >= per_label_cap:
                    continue

                words = s["section_text"].split()
                chunks = chunk_words(words, MIN_WORDS, MAX_WORDS)
                for ch_words in chunks:
                    if counts[label] >= per_label_cap:
                        break
                    chunk_text = " ".join(ch_words).strip()
                    if not chunk_text:
                        continue

                    rows.append(
                        {
                            "chunk_id": str(uuid.uuid4()),
                            "doc_id": doc_id,
                            "source_path": path,
                            "label": label,
                            "heading": s["heading"],
                            "n_words": len(ch_words),
                            "text": chunk_text,
                        }
                    )
                    counts[label] += 1

        if max_docs is not None and docs_seen >= max_docs:
            break

        if all(counts.get(lbl, 0) >= per_label_cap for lbl in LABELS):
            break

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print("Saved:", out_path)
    print("Counts per label:", counts)
    print("Total rows:", len(out_df))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-root", required=True, help="Root folder with clean parquet shards")
    ap.add_argument("--out", required=True, help="Output parquet path for task1 dataset")
    ap.add_argument("--per-label-cap", type=int, default=5000)
    ap.add_argument("--min-quality-score", type=int, default=0)
    ap.add_argument("--require-sections", action="store_true")
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()

    build_task1(
        clean_parquet_root=args.clean_root,
        out_path=args.out,
        per_label_cap=args.per_label_cap,
        min_quality_score=args.min_quality_score,
        require_sections=args.require_sections,
        max_docs=args.max_docs,
    )
