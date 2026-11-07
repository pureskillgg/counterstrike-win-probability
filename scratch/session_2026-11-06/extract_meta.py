import ast, os, re, sys, json

ROOT = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability"
TARGETS = [
    "scripts/model",
    "scripts/final",
    "scripts/sagemaker",
    "experiments/sprint",
    "experiments/positions",
    "analysis",
    "evaluation",
]

FAM = {
    "xgb": r"xgboost|XGBClassifier|XGBRegressor|xgb\.(train|DMatrix|Booster|cv)|\bxgb\b",
    "lgb": r"lightgbm|LGBM|lgb\.",
    "catboost": r"catboost|CatBoost",
    "gru": r"\bGRU\b|nn\.GRU",
    "lstm": r"\bLSTM\b|nn\.LSTM",
    "tcn": r"\bTCN\b|Conv1d.*dilation|dilat",
    "transformer": r"[Tt]ransformer",
    "logreg": r"LogisticRegression|logreg|logistic",
    "mlp": r"\bMLP\b|nn\.Linear|nn\.Sequential|mlp_",
    "gnn": r"\bGNN\b|GraphConv|GATConv|SAGEConv|graph model|train_graph|thgnn|THGNN",
    "w2v": r"[Ww]ord2[Vv]ec|gensim|skip-?gram",
    "rf_other": r"RandomForest|ExtraTrees|GradientBoosting|HistGradientBoosting|SVC|SVR|KNeighbors|Ridge|Lasso|ElasticNet",
}

TRAIN = r"\.fit\(|\bdef train|\btrain\(|optimizer|\.backward\(|xgb\.train|booster|\bepochs?\b|num_boost_round|lr=|learning_rate"
EVAL = r"log_loss|brier|\bBrier\b|\bauc\b|AUC|roc_auc|ECE|accuracy|metrics|score\(|predict_proba|\.predict\("
OUT = r"to_csv|to_parquet|to_json|json\.dump|savefig|write_text|\.write\(|np\.save|torch\.save|\.save_model|\.save\(|open\([^)]*['\"]w"


def docstring_or_comment(path):
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
    except Exception as e:
        return "", ""
    doc = ""
    try:
        tree = ast.parse(src)
        d = ast.get_docstring(tree)
        if d:
            doc = " ".join(d.strip().split())
    except Exception:
        pass
    comment = ""
    if not doc:
        lines = []
        for ln in src.splitlines()[:30]:
            s = ln.strip()
            if s.startswith("#"):
                lines.append(s.lstrip("#").strip())
            elif s == "" or s.startswith("#!") or s.startswith("# -*-"):
                continue
            else:
                break
        comment = " ".join(l for l in lines if l)
    return doc, comment


def scan(path):
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        src = ""
    fams = []
    counts = {}
    for k, pat in FAM.items():
        n = len(re.findall(pat, src))
        if n:
            counts[k] = n
            fams.append(k)
    train_n = len(re.findall(TRAIN, src))
    eval_n = len(re.findall(EVAL, src))
    out_hits = []
    for m in re.finditer(r"['\"]([^'\"\n]*?(?:results|output|outputs|reports|artifacts|models|out|tables)[^'\"\n]*?)['\"]", src):
        s = m.group(1)
        if "/" in s or "\\" in s or "." in s:
            if len(s) < 160:
                out_hits.append(s)
    out_hits = list(dict.fromkeys(out_hits))[:6]
    outw = len(re.findall(OUT, src))
    return {
        "lines": src.count("\n"),
        "fams": counts,
        "train_n": train_n,
        "eval_n": eval_n,
        "out_writes": outw,
        "out_paths": out_hits,
        "uses_sklearn": bool(re.search(r"sklearn", src)),
        "uses_torch": bool(re.search(r"\btorch\b", src)),
        "uses_xgb": bool(re.search(r"xgboost|xgb", src)),
    }


def main():
    rows = []
    for t in TARGETS:
        base = os.path.join(ROOT, t)
        if not os.path.isdir(base):
            continue
        for dp, dn, fn in os.walk(base):
            dn[:] = [d for d in dn if d != "__pycache__"]
            for f in sorted(fn):
                if not (f.endswith(".py") or f.endswith(".sh") or f.endswith(".ipynb")):
                    continue
                p = os.path.join(dp, f)
                rel = os.path.relpath(p, ROOT).replace("\\", "/")
                doc, com = docstring_or_comment(p) if f.endswith(".py") else ("", "")
                meta = scan(p)
                rows.append({"path": rel, "doc": doc[:400], "comment": com[:400], **meta})
    json.dump(rows, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print(len(rows))


if __name__ == "__main__":
    main()
