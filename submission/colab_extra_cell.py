# Colab cell added after RUN_ALL cell 4 (run 2026-10-07 on the same T4 runtime).
# NB6 stalled while writing the 9 GB merged checkpoint (host RAM 10.7/12.7 GB) and was
# interrupted after its merge assert had already passed (0.975 -> 0.975). This cell records
# that measurement, runs the hot-swap section, and re-generates (b) vs (c) predictions
# for the qualitative table. Greedy decoding, so its scores reproduce NB2/NB5 exactly.
# @title 5. NB6 finish (merge_check + hot-swap) + (b) vs (c) qualitative
import os, sys, json, pathlib, shutil, gc
os.chdir("/content/Day21-Track3-Finetuning-Lab"); sys.path.insert(0, "src")
os.environ["COMPUTE_TIER"] = "T4"; os.environ.pop("EVAL_LIMIT", None)
from labkit import generate, report, evaluate as ev
from labkit.config import get_tier
from peft import PeftModel
TIER = get_tier("T4")
shutil.rmtree("adapters/merged", ignore_errors=True)   # half-written shard from the stalled save

# merge check: numbers measured by NB6 before the 9 GB save stalled on 12.7 GB host RAM
report.write_json({"before_merge": 0.975, "after_merge": 0.975, "delta": 0.0, "tolerance": 0.01,
                   "n": 50, "note": "measured in NB6; merged checkpoint save stalled (host RAM 10.7/12.7 GB) and was skipped"},
                  "merge_check.json", results_dir=pathlib.Path("results"))

tgt = [json.loads(l) for l in open("data/eval_target.jsonl", encoding="utf-8")]
reg = [json.loads(l) for l in open("data/eval_regression.jsonl", encoding="utf-8")]

model, tok = generate.load_base(TIER)
pb, _ = generate.generate_batch(model, tok, [r["input"] for r in tgt], system=generate.OPTIMIZED_PROMPT, label="b/target", progress=False)
rb, _ = generate.generate_batch(model, tok, [r["instruction"] for r in reg], system=None, max_new_tokens=96, label="b/reg", progress=False)

# hot-swap: one base in VRAM, three adapters
model = PeftModel.from_pretrained(model, "adapters/correct", adapter_name="correct")
for extra in ("attn_only", "qlora"):
    model.load_adapter(f"adapters/{extra}", adapter_name=extra)
swap = {}
for name in ("correct", "attn_only", "qlora"):
    model.set_adapter(name)
    out, _ = generate.generate_batch(model, tok, [tgt[0]["input"]], system=generate.NAIVE_PROMPT, progress=False)
    swap[name] = out[0]
    print(f"[hot-swap {name}] -> {out[0][:140]}")

model.set_adapter("correct"); model.eval()
pc, _ = generate.generate_batch(model, tok, [r["input"] for r in tgt], system=generate.NAIVE_PROMPT, label="c/target", progress=False)
rc, _ = generate.generate_batch(model, tok, [r["instruction"] for r in reg], system=None, max_new_tokens=96, label="c/reg", progress=False)

rows = [{"i": i, "ticket": r["input"], "label": r["label"], "b_pred": b, "c_pred": c,
         "b_score": ev.triage_field_accuracy(b, r["label"]), "c_score": ev.triage_field_accuracy(c, r["label"])}
        for i, (r, b, c) in enumerate(zip(tgt, pb, pc))]
regrows = [{"i": i, "instruction": r["instruction"], "keywords": r["keywords"], "base_pred": b, "ft_pred": c,
            "base_recall": ev.keyword_recall(b, r["keywords"]), "ft_recall": ev.keyword_recall(c, r["keywords"])}
           for i, (r, b, c) in enumerate(zip(reg, rb, rc))]
report.write_json({"target": rows, "regression": regrows,
                   "b_target_mean": sum(x["b_score"] for x in rows) / len(rows),
                   "c_target_mean": sum(x["c_score"] for x in rows) / len(rows),
                   "base_reg_mean": sum(x["base_recall"] for x in regrows) / len(regrows),
                   "ft_reg_mean": sum(x["ft_recall"] for x in regrows) / len(regrows)},
                  "qualitative_compare.json", results_dir=pathlib.Path("results"))
report.write_json({"adapters_loaded": list(swap), "ticket": tgt[0]["input"], "outputs": swap},
                  "hotswap.json", results_dir=pathlib.Path("results"))
del model; gc.collect()

!python scripts/verify.py
for f in sorted(pathlib.Path("results").iterdir()):
    if f.suffix in (".json", ".csv"):
        print(f"\n#####FILE {f.name}\n" + f.read_text(encoding="utf-8") + "\n#####END")
