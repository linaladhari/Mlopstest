import pickle

# Load the trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Example input for IRIS (4 features)
example = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(example)

print("Prediction:", prediction)
