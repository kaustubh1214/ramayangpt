import json
from pymongo import MongoClient

# Replace with your actual password
MONGO_URI = "mongodb+srv://kaustubhshinde24:bas7.f.i6iDX8VK@cluster0.1bntd.mongodb.net/"

client = MongoClient(MONGO_URI)

# Create/use the database and collection
db = client["RamayanaDB"]
collection = db["VersesCollection"]

# List of JSON files
json_files = [
    "BalaKanda.json",
    "AyodhyaKanda.json",
    "AranyaKanda.json",
    "KishkindhaKanda.json",
    "SundaraKanda.json",
    "YuddhaKanda.json"
]

# Insert verses from each Kanda
for file_name in json_files:
    kanda = file_name.replace(".json", "")
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Add kanda name to each verse
        for verse in data:
            verse["kanda"] = kanda
        # Insert into MongoDB
        collection.insert_many(data)
        print(f"✅ Inserted {len(data)} verses from {file_name}")
