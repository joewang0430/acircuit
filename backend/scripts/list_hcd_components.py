import json
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "data" / "hcd" / "Component Symbol and Text Label Data" / "component_annotations.json"
if not p.exists():
    print("component_annotations.json not found at: {}\nPlease check the path.".format(p))
    raise SystemExit(1)


d = json.loads(p.read_text())
cats = d.get("categories", [])
names = []
# categories can be list of dicts with 'name' key
for c in cats:
    if isinstance(c, dict):
        name = c.get("name")
        if name:
            names.append(name)
    else:
        names.append(str(c))

print(f"总类别数量 (categories field length): {len(cats)}")
print("\n类别清单:")
for n in names:
    print("  ", n)

# Save to file for reference
out = {"count": len(cats), "names": names}
out_path = p.parent / "components_hcd_list.json"
out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"\n结果已保存到: {out_path}")
