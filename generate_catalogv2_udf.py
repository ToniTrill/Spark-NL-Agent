import os
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import traceback

load_dotenv()

UDFS_BASE_PATH = "db/udfbench/engines/pyspark/udfs"
OUTPUT_CATALOG = "db/udfbench/udf_catalog_v2.json"
CATEGORIES = ["scalar", "aggregate", "table"]

def clean_json_response(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def scan_and_describe_udfs():
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")
    catalog = {"udfs": []}
    counter = 1

    for category in CATEGORIES:
        folder_path = os.path.join(UDFS_BASE_PATH, category)
        print(f"\nProcessant categoria: {category} -> {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"  NO TROBADA: {folder_path}")
            continue

        files = os.listdir(folder_path)
        print(f"  Fitxers trobats: {files}")

        for filename in files:
            # Ignorar carpetes i fitxers que no son .py
            file_path = os.path.join(folder_path, filename)
            if not os.path.isfile(file_path):
                print(f"  Saltant carpeta: {filename}")
                continue
            if not filename.endswith(".py") or filename.startswith("__"):
                continue

            udf_name = filename.replace(".py", "")
            print(f"  Analitzant: {udf_name}...")

            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()

            prompt = f"""Analyze the following PySpark UDF code.
CATEGORY: {category}
NAME: {udf_name}

CODE:
{code_content}

Return ONLY a valid JSON object with no markdown, no backticks, no extra text:
{{
    "description": "One sentence explaining what this function does",
    "input_type": "type of input arguments",
    "output_type": "return type (use 'rset' if category is table)"
}}"""

            try:
                response = llm.invoke([HumanMessage(content=prompt)])

                if isinstance(response.content, list):
                    raw_content = " ".join([
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in response.content
                    ])
                else:
                    raw_content = str(response.content)

                print(f"    RAW response: {raw_content[:150]}")

                json_str = clean_json_response(raw_content)
                info = json.loads(json_str)

                catalog["udfs"].append({
                    "id": f"U{counter}",
                    "name": udf_name,
                    "category": category,
                    "description": info["description"],
                    "input_type": info["input_type"],
                    "output_type": info["output_type"]
                })
                print(f"    OK: {udf_name}")
                counter += 1

            except Exception as e:
                print(f"    ERROR en {udf_name}: {e}")
                traceback.print_exc()

    os.makedirs(os.path.dirname(OUTPUT_CATALOG), exist_ok=True)
    with open(OUTPUT_CATALOG, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=4)

    print(f"\nTotal UDFs processades: {counter - 1}")
    print(f"Catàleg guardat a: {OUTPUT_CATALOG}")

if __name__ == "__main__":
    scan_and_describe_udfs()