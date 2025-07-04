import datasets
import json

ds = datasets.load_dataset("NovaSky-AI/Sky-T1_data_17k")
print(ds)
print(ds["train"][0])

with open("Sky-T1_data_17k.jsonl", "w") as f:
    for each in ds["train"]:
        f.write(json.dumps(each) + "\n")
