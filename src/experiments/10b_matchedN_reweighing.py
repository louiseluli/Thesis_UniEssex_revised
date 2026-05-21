# -*- coding: utf-8 -*-
"""
10b_matchedN_reweighing.py
==========================

Purpose
-------
Matched-N ablation for the reweighing step: verify that the 95.3% accuracy
reported in step 10 is not inflated by the White Women group having 14x more
training examples than Black Women (31,459 vs 2,272).

Method
------
- Take the canonical step-06 train split.
- Cap White Women training examples to N_BW (= number of Black Women in train).
- All other groups unchanged.
- Run the same Kamiran-Calders reweighing + RF pipeline.
- Compare accuracy/fairness with the full-sample reweighing result.

Output
------
outputs/data/10b_matchedN_overall_metrics.csv
outputs/data/10b_matchedN_group_metrics.csv
outputs/data/10b_matchedN_disparities.csv
outputs/data/10b_matchedN_comparison.csv   <-- side-by-side with step 10
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project path ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve()
for _ in range(5):
    if (_ROOT / "src").is_dir():
        sys.path.append(str(_ROOT))
        break
    _ROOT = _ROOT.parent

# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_ww(df: pd.DataFrame) -> pd.Series:
    return ((df.get("race_ethnicity_white", 0) == 1) &
            (df.get("gender_female", 0) == 1))

def _is_bw(df: pd.DataFrame) -> pd.Series:
    return ((df.get("race_ethnicity_black", 0) == 1) &
            (df.get("gender_female", 0) == 1))


def _cap_white_women(train: pd.DataFrame, n_cap: int, seed: int) -> pd.DataFrame:
    """Return train with White Women capped to n_cap rows, stratified by label."""
    ww_mask = _is_ww(train)
    ww_df   = train[ww_mask]
    rest_df = train[~ww_mask]

    if len(ww_df) <= n_cap:
        print(f"[MATCHED-N] WW already <= {n_cap}; no capping needed.")
        return train

    # Stratified cap: preserve Amateur/non-Amateur ratio within WW
    from sklearn.model_selection import train_test_split
    cats = ww_df["categories"].fillna("").str.split(",").str[0].str.strip()
    labels = (cats == "Amateur").astype(int)
    _, ww_kept = train_test_split(
        ww_df, test_size=n_cap, random_state=seed, stratify=labels
    )
    result = pd.concat([rest_df, ww_kept], ignore_index=True)
    print(f"[MATCHED-N] WW capped: {len(ww_df):,} → {len(ww_kept):,} "
          f"(target N_BW={n_cap:,}). Total train: {len(result):,}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_all = time.perf_counter()
    print("--- Starting Step 10b: Matched-N Reweighing Ablation ---")

    # ── 1. Load corpus + splits ───────────────────────────────────────────────
    from src.utils.theme_manager import load_config  # noqa: PLC0415
    CONFIG = load_config()
    SEED = int(CONFIG.get("reproducibility", {}).get("seed", 95))
    data_dir  = Path(CONFIG["paths"]["data"])
    corpus_path = data_dir / "01_ml_corpus.parquet"

    print(f"[READ] {corpus_path}")
    df = pd.read_parquet(corpus_path)

    # Ensure baseline columns
    if "combined_text_clean" not in df.columns:
        df["combined_text_clean"] = (
            df.get("title", "").fillna("").astype(str) + " " +
            df.get("tags",  "").fillna("").astype(str)
        )
    for c in ("rating", "views"):
        if c not in df.columns:
            df[c] = 0.0
    if "categories" not in df.columns:
        df["categories"] = ""

    # Splits from step-06
    def read_ids(p: Path):
        if p.exists():
            return pd.Index(pd.read_csv(p)["video_id"])
        return None

    tr_ids = read_ids(data_dir / "06_train_ids.csv")
    va_ids = read_ids(data_dir / "06_val_ids.csv")
    te_ids = read_ids(data_dir / "06_test_ids.csv")

    if tr_ids is None:
        raise RuntimeError("Step-06 train IDs not found. Run step 06 first.")

    train_df = df[df["video_id"].isin(tr_ids)].reset_index(drop=True)
    val_df   = df[df["video_id"].isin(va_ids)].reset_index(drop=True)
    test_df  = df[df["video_id"].isin(te_ids)].reset_index(drop=True)

    # ── 2. Count groups in train ───────────────────────────────────────────────
    n_bw = int(_is_bw(train_df).sum())
    n_ww = int(_is_ww(train_df).sum())
    print(f"[STATS] Train — Black Women: {n_bw:,}  |  White Women: {n_ww:,}  |  ratio WW/BW: {n_ww/max(n_bw,1):.1f}x")

    train_matched = _cap_white_women(train_df, n_cap=n_bw, seed=SEED)  # SEED defined above

    # ── 3. Reweighing + RF on matched train ───────────────────────────────────
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    TEXT_COL = "combined_text_clean"
    NUM_COLS = ["rating", "views"]

    def group_labels(d):
        lbl = np.full(len(d), "Other", dtype=object)
        lbl[(_is_ww(d)).values] = "White Women"
        lbl[(_is_bw(d)).values] = "Black Women"
        lbl[((d.get("race_ethnicity_asian",0)==1)&(d.get("gender_female",0)==1)).values] = "Asian Women"
        lbl[((d.get("race_ethnicity_latina",0)==1)&(d.get("gender_female",0)==1)).values] = "Latina Women"
        return pd.Series(lbl, index=d.index, name="Group")

    def binary_target(d):
        pri = d["categories"].fillna("").str.split(",").str[0].str.strip()
        return (pri == "Amateur").astype(int).to_numpy()

    def reweigh_weights(y, groups):
        a = groups.to_numpy()
        dfw = pd.DataFrame({"a": a, "y": y})
        pa  = dfw["a"].value_counts(normalize=True)
        py  = dfw["y"].value_counts(normalize=True)
        pay = dfw.value_counts(normalize=True)
        w = np.ones(len(y), dtype=float)
        for (ai, yi), p_ay in pay.items():
            w_ay = (pa[ai] * py[yi]) / max(p_ay, 1e-12)
            w[(a == ai) & (y == yi)] = w_ay
        w = w * (len(y) / w.sum())
        return w

    text_pipe = Pipeline([
        ("hash", HashingVectorizer(n_features=2**18, alternate_sign=False,
                                    ngram_range=(1,2), norm="l2", stop_words="english")),
        ("svd",  TruncatedSVD(n_components=256, random_state=SEED))
    ])
    pre = ColumnTransformer(
        [("text", text_pipe, TEXT_COL), ("num", "passthrough", NUM_COLS)],
        remainder="drop", sparse_threshold=0.0
    )
    model = Pipeline([("prep", pre),  # type: ignore[assignment]
                      ("rf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))])

    y_tr = binary_target(train_matched)
    g_tr = group_labels(train_matched)
    w_tr = reweigh_weights(y_tr, g_tr)

    y_va = binary_target(val_df)
    y_te = binary_target(test_df)
    g_te = group_labels(test_df)

    t0 = time.perf_counter()
    print("Fitting matched-N RF with reweighing ...")
    model.fit(train_matched, y_tr, rf__sample_weight=w_tr)
    print(f"[TIME] fit: {time.perf_counter()-t0:.1f}s")

    ypv = model.predict(val_df)
    ypt = model.predict(test_df)

    def metrics_row(split, yt, yp):
        return {"Split": split,
                "Accuracy": round(accuracy_score(yt,yp),3),
                "Precision": round(precision_score(yt,yp,zero_division=0),3),
                "Recall": round(recall_score(yt,yp,zero_division=0),3),
                "F1": round(f1_score(yt,yp,zero_division=0),3)}

    overall = pd.DataFrame([metrics_row("Val",y_va,ypv), metrics_row("Test",y_te,ypt)])

    # Group metrics (Test)
    test_meta = test_df.copy(); test_meta["Group"] = g_te.values
    grp_rows = []
    for grp in sorted(g_te.unique()):
        idx = np.where(g_te.values == grp)[0]
        yt_, yp_ = y_te[idx], ypt[idx]
        grp_rows.append({"Group": grp, "N": len(idx),
            "Accuracy": round(accuracy_score(yt_,yp_),3),
            "Precision": round(precision_score(yt_,yp_,zero_division=0),3),
            "Recall": round(recall_score(yt_,yp_,zero_division=0),3),
            "F1": round(f1_score(yt_,yp_,zero_division=0),3)})
    grp_df = pd.DataFrame(grp_rows).sort_values("Group")

    base_row = grp_df[grp_df["Group"]=="White Women"].iloc[0]
    disp_rows = []
    for _, r in grp_df.iterrows():
        if r["Group"] == "White Women": continue
        disp_rows.append({"Comparison Group": r["Group"],
            "Accuracy Disparity": round(base_row["Accuracy"]-r["Accuracy"],3),
            "Equal Opportunity Difference": round(base_row["Recall"]-r["Recall"],3),
            "Precision Disparity": round(base_row["Precision"]-r["Precision"],3)})
    disp_df = pd.DataFrame(disp_rows).sort_values("Comparison Group")

    # ── 4. Compare with full-sample step-10 results ───────────────────────────
    full_overall = pd.read_csv(data_dir / "10_reweigh_overall_metrics.csv")
    full_grp     = pd.read_csv(data_dir / "10_reweigh_group_metrics.csv")
    full_disp    = pd.read_csv(data_dir / "10_reweigh_disparities.csv")

    full_test_acc = full_overall[full_overall["Split"]=="Test"]["Accuracy"].iloc[0]
    match_test_acc = overall[overall["Split"]=="Test"]["Accuracy"].iloc[0]

    comparison = pd.DataFrame([
        {"Scenario": "Full corpus (N_WW=31,459)", **full_overall[full_overall["Split"]=="Test"].iloc[0].drop("Split").to_dict()},
        {"Scenario": f"Matched-N (N_WW=N_BW={n_bw:,})", **overall[overall["Split"]=="Test"].iloc[0].drop("Split").to_dict()},
    ])
    acc_drop = full_test_acc - match_test_acc
    print(f"\n[RESULT] Full accuracy: {full_test_acc:.3f}  |  Matched-N accuracy: {match_test_acc:.3f}  |  Drop: {acc_drop:+.3f}")

    disp_compare = []
    for grp in ["Black Women", "Asian Women", "Latina Women"]:
        full_eod   = full_disp[full_disp["Comparison Group"]==grp]["Equal Opportunity Difference"].values
        match_eod  = disp_df[disp_df["Comparison Group"]==grp]["Equal Opportunity Difference"].values
        disp_compare.append({"Group": grp,
            "EOD_full": round(float(full_eod[0]),3) if len(full_eod) else None,
            "EOD_matchedN": round(float(match_eod[0]),3) if len(match_eod) else None})
    disp_compare_df = pd.DataFrame(disp_compare)
    print("\n[EOD COMPARISON]")
    print(disp_compare_df.to_string(index=False))

    # ── 5. Save ───────────────────────────────────────────────────────────────
    out_dir = data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(out_dir / "10b_matchedN_overall_metrics.csv", index=False)
    grp_df.to_csv(out_dir / "10b_matchedN_group_metrics.csv", index=False)
    disp_df.to_csv(out_dir / "10b_matchedN_disparities.csv", index=False)
    comparison.to_csv(out_dir / "10b_matchedN_comparison.csv", index=False)
    disp_compare_df.to_csv(out_dir / "10b_matchedN_eod_comparison.csv", index=False)

    print("\n[SAVED] 10b_matchedN_*.csv")
    print(f"[TIME] total: {time.perf_counter()-t_all:.1f}s")
    print("--- Step 10b: Matched-N Ablation Completed ---")


if __name__ == "__main__":
    main()
