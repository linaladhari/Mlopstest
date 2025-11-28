import json

# normally evaluation happens here
metrics = {"val_accuracy": 0.85}

with open(sys.argv[2], "w") as f:
    json.dump(metrics, f)
