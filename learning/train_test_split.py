from collections import defaultdict
import sys
import pathlib
import random

random.seed(123859)

dataset_path = pathlib.Path(sys.argv[1])

img_path = dataset_path / "images"

# Each subject has an image and a label associated with it
subjects_by_id = defaultdict(list)
for p in img_path.iterdir():
    this_id = int(p.stem.split("_")[1])
    subjects_by_id[this_id].append(p.stem)

train = 0.85
val = 0.1
test = 1 - (train + val)

allocations = random.choices(
    (0, 1, 2), weights=(train, val, test), k=len(subjects_by_id.keys())
)

output_txt_opts = ("train.txt", "val.txt", "test.txt")

# Delete the old files
for txt_fn in output_txt_opts:
    with open(dataset_path / txt_fn, 'w') as f:
        pass

for idx, subj_id in zip(allocations, subjects_by_id.keys()):
    with open(dataset_path / output_txt_opts[idx], "a") as f:
        for file_stem in subjects_by_id[subj_id]:
            print(f"./images/{file_stem}.png", file=f)
