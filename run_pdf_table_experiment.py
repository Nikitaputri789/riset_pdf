"""
PDF table extraction experiment: baseline vs layout-aware, schema-validated pipeline.

Outputs (in ./results):
  experiment_detailed_results.csv   per-document, per-pipeline metrics
  experiment_summary_results.csv    per-scenario means / SD
  ablation_results.csv              component ablation (tau = 7 px)
  sensitivity_analysis_results.csv  tolerance sweep tau in {1,3,5,7}
"""
import os
import time
import numpy as np
import pandas as pd
import pdfplumber
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

BENCH = "benchmark_data"
OUT = "results"
os.makedirs(BENCH, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
styles = getSampleStyleSheet()
N_DOCS = 5          # documents per scenario
N_REPEATS = 7       # timing repetitions per document (first run discarded as warm-up)
DEFAULT_TAU = 3.0   # pdfplumber's default snapping tolerance (px = PDF points)


# ----------------------------------------------------------------------------
# 1. Benchmark generation (deterministic)
# ----------------------------------------------------------------------------
def _para(t):
    return Paragraph(str(t), styles["Normal"])


def build_scenario_a():
    gt = []
    header = ["Employee_ID", "Department", "Role", "Hours", "Hourly_Rate"]
    for idx in range(1, N_DOCS + 1):
        path = f"{BENCH}/scenario_A_{idx}.pdf"
        rows = [header]
        for r in range(1, 16):
            rows.append([f"EMP-{idx}{r:02d}", f"Dept_{(r % 4) + 1}",
                         f"Engineer_L{(r % 3) + 1}", str(35 + (r % 10)), f"{25 + r}.00"])
        t = Table(rows, colWidths=[85, 95, 110, 65, 85])
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.8, colors.black),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
        SimpleDocTemplate(path, pagesize=A4).build(
            [Paragraph(f"Scenario A Benchmark Table #{idx}", styles["Heading2"]), Spacer(1, 12), t])
        gt.append(dict(scenario="A", file=path, rows=rows))
    return gt


def _borderless_table(header, body, widths, valign, repeat=0):
    flow = [[_para(h) for h in header]] + [[_para(c) for c in r] for r in body]
    t = Table(flow, colWidths=widths, repeatRows=repeat)
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
                           ("LINEBELOW", (0, 0), (-1, 0), 1.0, colors.black),
                           ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
                           ("VALIGN", (0, 0), (-1, -1), valign)]))
    return t


def build_scenario_b(tag="B", valign="TOP"):
    gt = []
    header = ["Task_ID", "Module_Name", "Specification_Note", "Latency_ms"]
    for idx in range(1, N_DOCS + 1):
        path = f"{BENCH}/scenario_{tag}_{idx}.pdf"
        body = [[f"TSK-{idx}{r:02d}", f"CoreUnit_{r}",
                 f"Cache coherence verification across core cluster {r} under burst load",
                 f"{12.4 + r:.1f}"] for r in range(1, 13)]
        t = _borderless_table(header, body, [70, 95, 195, 75], valign)
        SimpleDocTemplate(path, pagesize=A4).build(
            [Paragraph(f"Scenario {tag} Benchmark Table #{idx}", styles["Heading2"]), Spacer(1, 12), t])
        gt.append(dict(scenario=tag, file=path, rows=[header] + body))
    return gt


def build_scenario_c():
    gt = []
    header = ["Trace_ID", "Core", "Event_Description", "Cycles", "Status"]
    for idx in range(1, N_DOCS + 1):
        path = f"{BENCH}/scenario_C_{idx}.pdf"
        body = [[f"TRC-{idx}{r:03d}", f"Core_{(r % 8) + 1}",
                 f"Pipeline stall detected while resolving memory dependency chain number {r} "
                 f"during speculative execution",
                 str(1000 + 37 * r), "OK" if r % 5 else "RETRY"] for r in range(1, 31)]
        t = _borderless_table(header, body, [60, 50, 215, 50, 55], "TOP", repeat=1)
        SimpleDocTemplate(path, pagesize=A4).build(
            [Paragraph(f"Scenario C Benchmark Table #{idx}", styles["Heading2"]), Spacer(1, 12), t])
        gt.append(dict(scenario="C", file=path, rows=[header] + body))
    return gt


# ----------------------------------------------------------------------------
# 2. Extraction pipelines
# ----------------------------------------------------------------------------
def norm(s):
    return " ".join(str(s if s is not None else "").split())


def clean_table(tbl):
    return [[norm(c) for c in row] for row in tbl]


def baseline(path, schema=None, **_):
    """Default pdfplumber extract_tables() (line strategy), whitespace normalisation only."""
    rows = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for tbl in page.extract_tables() or []:
                rows.extend(clean_table(tbl))
    return rows


def text_baseline(path, schema=None, **_):
    """pdfplumber extract_tables() with the built-in text strategy (stronger off-the-shelf baseline)."""
    rows = []
    ts = {"vertical_strategy": "text", "horizontal_strategy": "text"}
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for tbl in page.extract_tables(table_settings=ts) or []:
                rows.extend(clean_table(tbl))
    return rows


def _valid_grid(rows, schema):
    return bool(rows) and rows[0] == schema and all(len(r) == len(schema) for r in rows)


def _cluster_lines(words, tau):
    """Chain-link words into visual lines when their 'top' gap is <= tau."""
    words = sorted(words, key=lambda w: w["top"])
    lines, cur, last = [], [], None
    for w in words:
        if last is not None and w["top"] - last > tau:
            lines.append(cur)
            cur = []
        cur.append(w)
        last = w["top"]
    if cur:
        lines.append(cur)
    return lines


def proposed(path, schema, tau=DEFAULT_TAU, fallback=True, merge=True, filt_hdr=True, **_):
    """
    Layout-aware, schema-validated pipeline.
    (1) line-based extraction; accept only if it satisfies the schema, otherwise
    (2) coordinate-based fallback: column starts from header word x0, words snapped to
        columns (x tolerance tau), grouped into visual lines (y tolerance tau);
    (3) continuation-row merging (line without identifier-column word joins previous row);
    (4) schema validation: repeated header removal, orphan/malformed row removal.
    """
    n = len(schema)
    with pdfplumber.open(path) as pdf:
        grid = []
        for page in pdf.pages:
            for tbl in page.extract_tables() or []:
                grid.extend(clean_table(tbl))
        if _valid_grid(grid, schema) or not fallback:
            rows = grid
        else:
            starts = None
            rows = []
            for page in pdf.pages:
                words = page.extract_words()
                anchor = [w["top"] for w in words if w["text"] == schema[0]]
                if anchor:                          # schema anchoring: ignore text above the table
                    words = [w for w in words if w["top"] >= min(anchor) - tau]
                if starts is None:
                    hw = {w["text"]: w for w in words if w["text"] in schema}
                    if len(hw) != n:
                        return []
                    starts = [hw[h]["x0"] for h in schema]
                for line in _cluster_lines(words, tau):
                    cells = [[] for _ in range(n)]
                    for w in line:
                        cand = [j for j in range(n) if starts[j] - tau <= w["x0"]]
                        if cand:
                            cells[max(cand)].append(w)
                    if cells[0] or not merge:
                        rows.append(cells)          # identifier present -> new logical row
                    elif rows:
                        for j in range(n):          # continuation line -> merge into previous row
                            rows[-1][j].extend(cells[j])
                    # orphan continuation before any row: dropped by schema validation
            rows = [[norm(" ".join(w["text"] for w in sorted(c, key=lambda w: (round(w["top"], 1), w["x0"]))))
                     for c in r] for r in rows]
        if filt_hdr:
            out, seen = [], False
            for r in rows:
                if r == schema:
                    if seen:
                        continue
                    seen = True
                out.append(r)
            rows = out
        return rows


# ----------------------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------------------
def cell_set(rows):
    return {(i, j, v) for i, r in enumerate(rows) for j, v in enumerate(r)}


def prf(pred, gt):
    cp, cg = cell_set(pred), cell_set(gt)
    tp = len(cp & cg)
    p = tp / len(cp) if cp else 0.0
    r = tp / len(cg) if cg else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return 100 * p, 100 * r, 100 * f


def struct_err(pred, gt, n_cols):
    mis = sum(1 for r in pred if len(r) != n_cols)
    return min(100.0, 100 * (mis + abs(len(pred) - len(gt))) / len(gt))


def row_exact(pred, gt):
    return 100 * sum(1 for a, b in zip(pred, gt) if a == b) / len(gt)


def timed(fn, path, schema, **kw):
    fn(path, schema, **kw)  # warm-up
    ts = []
    for _ in range(N_REPEATS - 1):
        t0 = time.perf_counter()
        out = fn(path, schema, **kw)
        ts.append((time.perf_counter() - t0) * 1000)
    return out, float(np.mean(ts))


def evaluate(docs, pipelines):
    recs = []
    for d in docs:
        schema, n = d["rows"][0], len(d["rows"][0])
        for name, (fn, kw) in pipelines.items():
            pred, ms = timed(fn, d["file"], schema, **kw)
            p, r, f = prf(pred, d["rows"])
            recs.append(dict(scenario=d["scenario"], file=os.path.basename(d["file"]), pipeline=name,
                             precision=p, recall=r, f1=f, row_exact=row_exact(pred, d["rows"]),
                             struct_err=struct_err(pred, d["rows"], n), time_ms=ms,
                             pred_rows=len(pred), gt_rows=len(d["rows"])))
    return pd.DataFrame(recs)


if __name__ == "__main__":
    docs = build_scenario_a() + build_scenario_b("B") + build_scenario_c() + build_scenario_b("D", "MIDDLE")
    main_pipes = {"Baseline (lines)": (baseline, {}),
                  "Baseline (text)": (text_baseline, {}),
                  "Proposed": (proposed, {"tau": DEFAULT_TAU})}
    det = evaluate(docs, main_pipes)
    det.to_csv(f"{OUT}/experiment_detailed_results.csv", index=False)
    summ = det.groupby(["scenario", "pipeline"]).agg(
        precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"),
        row_exact=("row_exact", "mean"), struct_err=("struct_err", "mean"),
        time_mean=("time_ms", "mean"), time_sd=("time_ms", "std")).reset_index()
    summ.to_csv(f"{OUT}/experiment_summary_results.csv", index=False)
    abl_pipes = {"Full": (proposed, {"tau": 7.0}),
                 "No fallback": (proposed, {"tau": 7.0, "fallback": False}),
                 "No row merging": (proposed, {"tau": 7.0, "merge": False}),
                 "No header filter": (proposed, {"tau": 7.0, "filt_hdr": False})}
    abl = evaluate(docs, abl_pipes)
    abl.groupby(["scenario", "pipeline"]).agg(f1=("f1", "mean"), struct_err=("struct_err", "mean"),
                                             time_ms=("time_ms", "mean")).reset_index() \
       .to_csv(f"{OUT}/ablation_results.csv", index=False)
    sens_pipes = {f"tau={t:g}": (proposed, {"tau": float(t)}) for t in (1, 3, 5, 7)}
    sens = evaluate(docs, sens_pipes)
    sens.groupby(["scenario", "pipeline"]).agg(f1=("f1", "mean"), struct_err=("struct_err", "mean"),
                                              time_ms=("time_ms", "mean")).reset_index() \
        .to_csv(f"{OUT}/sensitivity_analysis_results.csv", index=False)
    pd.set_option("display.width", 200)
    print("=== SUMMARY ===")
    print(summ.round(2).to_string(index=False))
    print("\n=== ABLATION (tau=7) ===")
    print(pd.read_csv(f"{OUT}/ablation_results.csv").round(2).to_string(index=False))
    print("\n=== SENSITIVITY ===")
    print(pd.read_csv(f"{OUT}/sensitivity_analysis_results.csv").round(2).to_string(index=False))
