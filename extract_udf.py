import re, json

def extract_udfs_from_query(query: str, all_udf_names: list) -> list:
    query_lower = query.lower()
    return [name for name in all_udf_names if name in query_lower]

# Generar el mapping
all_names = [u['name'].lower() for u in udfs] + [f"aggregate_{u['name'].lower()}" for u in aggregate_ids]

mapping = {}
for q_id, gold_query in gold_queries.items():
    mapping[q_id] = extract_udfs_from_query(gold_query, all_names)

with open("benchmark_udf_mapping.json", "w") as f:
    json.dump(mapping, f, indent=2)