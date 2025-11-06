import json
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "data" / "cghd" / "classes.json"
if not p.exists():
    print(f"classes.json not found at: {p}")
    raise SystemExit(1)

classes = json.loads(p.read_text())
all_labels = list(classes.keys())
# stoplist: labels that are clearly not physical electronic components
stoplist = set([
    "__background__", "text", "junction", "crossover", "terminal",
    "explanatory", "unknown", "mechanical", "magnetic", "optical", "block"
])

components = [lbl for lbl in all_labels if lbl not in stoplist]

print(f"总标签数（含背景）: {len(all_labels)}")
print(f"去掉背景/显式非器件后的标签数: {len(components)}")
print("\n全部标签列表:")
for lbl in all_labels:
    print("  ", lbl)

print("\n按过滤规则判断为『电元器件/相关类别』的标签: (若需调整规则请告知)")
for lbl in components:
    print("  ", lbl)

# Also save result to a file for later reference
out = {
    "total_labels": len(all_labels),
    "filtered_count": len(components),
    "components": components
}
out_path = Path(__file__).resolve().parents[2] / "data" / "cghd" / "components_filtered.json"
out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"\n结果已保存到: {out_path}")
