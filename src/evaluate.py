import sys
import json

# Arguments
# sys.argv[1] -> input data
# sys.argv[2] -> model path
# sys.argv[3] -> output metrics path
metrics_output_path = "src/metrics.txt"

# Normally evaluation happens here
metrics = {"val_accuracy": 0.85}  # dummy metric

# Save metrics to file
with open(metrics_output_path, "w") as f:
    json.dump(metrics, f)

