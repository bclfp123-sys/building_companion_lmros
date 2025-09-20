import pickle

# replace 'meta.pkl' with your actual filename
filename = "meta.pkl"

with open(filename, "rb") as f:
    data = pickle.load(f)


for r in data:
    print("-------------------------------")
    print(r["summary"])
    