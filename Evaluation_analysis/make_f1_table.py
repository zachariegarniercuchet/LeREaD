import csv
from pathlib import Path

systems = [
    ("gpt5.2", "Evaluation_analysis/results_aafinal"),
    ("gemma4", "Evaluation_analysis/results_aafinal"),
    ("qwen38", "Evaluation_analysis/results_aafinal"),
    ("muse", "Evaluation_analysis/results_aafinal"),
    ("saullm", "Evaluation_analysis/results_aafinal"),
    ("baseline-rules", "Evaluation_analysis/results_baseline"),
]
data = {}
labels = []
for name, d in systems:
    f = next(Path(d).glob(name + "_label_errors.csv"))
    for r in csv.DictReader(open(f, encoding="utf-8")):
        data.setdefault(r["label"], {})[name] = float(r["F1"]) * 100
        if r["label"] not in labels:
            labels.append(r["label"])

order = ["decision", "legislation", "secondary sources", "title",
         "citation", "fragment", "authors", "source", "unable to classify"]
cols = ["gpt5.2", "gemma4", "qwen38", "muse", "saullm", "baseline-rules"]
print(f"{'label':<20}" + "".join(f"{c:>10}" for c in cols))
for lab in order:
    row = data.get(lab, {})
    print(f"{lab:<20}" + "".join(f"{row.get(m, float('nan')):>10.1f}" for m in cols))

with open("Evaluation_analysis/results_aafinal/comparison_all6_F1_by_label.csv",
          "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["label"] + cols)
    for lab in order:
        w.writerow([lab] + [f"{data.get(lab, {}).get(m, 0):.4f}" for m in cols])
print("saved comparison_all6_F1_by_label.csv")
