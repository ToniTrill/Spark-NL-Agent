import json

UDF_CATALOG_PATH = "db/udfbench/udf_catalog.json"
UDFDEV_PATH = "db/udfbench/udfdev.json"
OUTPUT_PATH = "db/udfbench/udf_mapping.json"

with open(UDF_CATALOG_PATH) as f:
    udfs = json.load(f).get("udfs", [])

all_udf_names = []
for u in udfs:
    name = u['name'].lower()
    u_id = u.get("id", "")
    if 25 <= int(u_id[1:]) <= 28:
        all_udf_names.append(f"aggregate_{name}")
    else:
        all_udf_names.append(name)

with open(UDFDEV_PATH) as f:
    benchmark = json.load(f)

mapping = {}
for entry in benchmark:
    q_id = str(entry["question_id"])
    sql = entry["SQL"].lower()
    found = [name for name in all_udf_names if name in sql]
    mapping[q_id] = found
    print(f"Q{q_id}: {found}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(mapping, f, indent=2)