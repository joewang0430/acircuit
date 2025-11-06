import json
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "data"

cghd_p = root / "cghd" / "classes.json"
hcd_p = root / "hcd" / "Component Symbol and Text Label Data" / "components_hcd_list.json"

if not cghd_p.exists():
    print(f"CGHD classes.json not found at: {cghd_p}")
    raise SystemExit(1)
if not hcd_p.exists():
    print(f"HCD components file not found at: {hcd_p}")
    raise SystemExit(1)

cghd = json.loads(cghd_p.read_text())
# cghd is dict name->id; convert keys
cghd_labels = set(cghd.keys())

hcd = json.loads(hcd_p.read_text())
hcd_labels = set(hcd.get("names", []))

# Normalize labels: lower-case and replace spaces/dots/hyphens for fuzzy matching? We'll do exact match first, then a lower-normalized intersection.
exact_common = sorted(list(cghd_labels & hcd_labels))

# normalized: lowercase, remove punctuation and spaces
import re

def norm(s):
    return re.sub(r"[^0-9a-z]+", "", s.lower())

cghd_norm = {norm(s): s for s in cghd_labels}
hcd_norm = {norm(s): s for s in hcd_labels}

norm_common_keys = set(cghd_norm.keys()) & set(hcd_norm.keys())
norm_common = sorted([(cghd_norm[k], hcd_norm[k]) for k in norm_common_keys])

result = {
    "cghd_count": len(cghd_labels),
    "hcd_count": len(hcd_labels),
    "exact_common_count": len(exact_common),
    "exact_common": exact_common,
    "norm_common_count": len(norm_common),
    "norm_common_pairs": norm_common,
    "cghd_only": sorted(list(cghd_labels - hcd_labels)),
    "hcd_only": sorted(list(hcd_labels - cghd_labels)),
}

out_path = root / "labels_compare_report.json"
out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

print(json.dumps(result, indent=2, ensure_ascii=False))
print(f"\n结果已保存到: {out_path}")
