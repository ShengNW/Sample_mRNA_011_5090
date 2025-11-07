
# -*- coding: utf-8 -*-
import os, json, argparse

def load_manifest(dataset_dir):
    with open(os.path.join(dataset_dir, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def audit(dataset_dir):
    man = load_manifest(dataset_dir)
    counts = man.get("counts", {})
    total = counts.get("total", None)
    ok = True
    for split in ("train","val","test"):
        p = os.path.join(dataset_dir, "index", f"{split}.json")
        if not os.path.exists(p): 
            print(f"[{split}] index missing -> skip")
            continue
        rows = json.load(open(p, "r", encoding="utf-8"))
        # Check global_idx monotonic and within range
        last = -1
        for i, r in enumerate(rows):
            gi = int(r.get("global_idx", -1))
            if gi < 0:
                print(f"[{split}] row#{i} missing/invalid global_idx")
                ok=False; break
            if gi <= last:
                print(f"[{split}] non-monotonic global_idx at row#{i}: {gi} <= {last}")
                ok=False; break
            last = gi
        print(f"[{split}] {len(rows)} rows, monotonic={'OK' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    args = ap.parse_args()
    ok = audit(args.dataset_dir)
    print("ALL_OK" if ok else "HAS_ISSUE")
