import os, json, random

root = r"D:\BaiduNetdiskDownload\data\Aqua_60度"
src = os.path.join(root, "annotations_pose.json")

with open(src, "r", encoding="utf-8") as f:
    samples = json.load(f)

random.seed(42)
random.shuffle(samples)

n = len(samples)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)

train = samples[:n_train]
val   = samples[n_train:n_train+n_val]
test  = samples[n_train+n_val:]

for name, data in [("train_pose.json", train), ("val_pose.json", val), ("test_pose.json", test)]:
    out = os.path.join(root, name)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("saved:", out, "num:", len(data))
