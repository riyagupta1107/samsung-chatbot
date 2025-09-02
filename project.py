import os
import json
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # Make sure PINECONE_API_KEY is set

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to your existing index (all lowercase, no underscores)
index_name = "samsung-wm"  # Replace with your actual index name
index = pinecone_client.Index(index_name)

# Define categories and dataset paths
categories = {
    "washingmachine": "/Users/kashviaggarwal/Downloads/samsung_wm_issues.jsonl",
    "fridge": "/Users/kashviaggarwal/Downloads/samsung_fridge.jsonl"
}

# Loop through categories and upsert data into separate namespaces
for product_category, dataset_path in categories.items():
    if not os.path.exists(dataset_path):
        print(f"Dataset not found for {product_category}: {dataset_path}")
        continue

    with open(dataset_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    pinecone_records = []
    for r in records:
        issue_text = str(r.get("issue", "")).strip()
        solution_text = str(r.get("solution", "")).strip()
        if not issue_text:
            continue
        pinecone_records.append({
            "id": str(r["id"]),
            "issue": issue_text,
            "metadata": solution_text
        })

    # Upsert into Pinecone using namespace = product category
    index.upsert_records(
        namespace=product_category,
        records=pinecone_records
    )

    # Print stats per namespace
    stats = index.describe_index_stats()
    print(f"\nIndex stats for {product_category} namespace:")
    print(json.dumps(stats.to_dict(), indent=2))

print("\n✅ All datasets uploaded successfully to separate namespaces.")
