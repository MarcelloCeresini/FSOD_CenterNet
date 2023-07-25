import json

annotation_path = "/Users/marcelloceresini/github/FSOD_CenterNet/data/fsod/annotations/fsod_train.json"

with open(annotation_path, "r") as f:
    annotations = json.load(f)

supercats = set()
cats = set()
for category in annotations["categories"]:
    supercats.add(category["supercategory"])
    cats.add(category["name"])

print(len(supercats))
print(len(cats))
