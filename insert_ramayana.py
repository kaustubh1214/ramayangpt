import os
import json
import mysql.connector

# Database connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="valmiki_ramayana"
)
cursor = conn.cursor()

# List of Kanda JSON files
json_files = [
    "BalaKanda.json",
    "AyodhyaKanda.json",
    "AranyaKanda.json",
    "KishkindhaKanda.json",
    "SundaraKanda.json",
    "YuddhaKanda.json"
]

for file_name in json_files:
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data:
            cursor.execute("""
                INSERT INTO verses (
                    book, chapter, verse, wordDictionary, translation
                ) VALUES (%s, %s, %s, %s, %s)
            """, (
                entry.get("book"),
                entry.get("chapter"),
                entry.get("verse"),
                entry.get("wordDictionary"),
                entry.get("translation")
            ))

conn.commit()
cursor.close()
conn.close()
print("✅ All JSON files inserted successfully into `verses` table.")
