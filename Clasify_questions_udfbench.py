import os
import sys
import json
import re
import statistics
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from spark_toolkit.spark_sql import SparkSQL 
from generate_catalogv2_udf import clean_json_response
from pyspark.sql import SparkSession 

load_dotenv()

INPUT_FILE = "db/udfbench/udfdev.json"
OUTPUT_FILE = "db/udfbench/udfdev_v2.json"

BIRD_RULES = """
Detailed Rules for Difficulty Rating:
1. Question Understanding (1-3): 1=Straightforward, 2=Requires thought, 3=Ambiguous.
2. Knowledge Reasoning (1-3): 1=No external knowledge, 2=Requires some external evidence/UDFs, 3=Extensive knowledge/UDF logic needed.
3. Data Complexity (1-3): 1=Simple schema, 2=Complex values or links, 3=Highly complex values/schema.
4. SQL Complexity (1-3): 1=Simple SQL, 2=More keywords/functions, 3=Highly complex with many functions/subqueries.
The final difficulty is the average of these 4 scores.
"""
spark = SparkSession.builder \
    .appName("DifficultyClassifier") \
    .config("spark.driver.host", "localhost") \
    .getOrCreate()
db = SparkSQL(spark)

try:
    schema_info = db.get_table_info(db.get_usable_table_names())
except Exception as e:
    print(f"Error retrieving schema information: {str(e)}")
    schema_info = "Unable to retrieve schema information."


def clean_sql_query(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else text

def classify_difficulty():
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)

    classified_queries = []
    n = 3 # iterem per cada pregunta 3 vegades per veure si la classificacio es consistent o hi ha variabilitat de l'IA

    for query in queries:
        q_id = query.get("question_id")

        print(f"\nClassificant pregunta Q{q_id}...")

        iteration_averages = []
        all_justifications = []

        for i in range(n):
            prompt = f"""
                Act as an expert SQL annotator. Use the following BIRD criteria to evaluate a Text-to-SQL example.
                {BIRD_RULES}

                DATABASE SCHEMA:
                {schema_info}

                EXAMPLE TO EVALUATE:
                    Question: {query['question']}
                    Evidence: {query['evidence']}
                    Target SQL: {query['SQL']}

                Return ONLY a JSON:
                {{
                    "q_understanding": 1-3,
                    "k_reasoning": 1-3,
                    "data_complexity": 1-3,
                    "sql_complexity": 1-3,
                    "justification": "..."
                }}
            """
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                
                raw_content = response.content
                if isinstance(raw_content, list):
                    # Si és una llista, unim tots els fragments de text
                    raw_content = " ".join([
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in raw_content
                    ])
                else:
                    raw_content = str(raw_content)

                json_str = clean_json_response(raw_content)
                result = json.loads(json_str)

                #Calcular la puntuació total com la mitjana de les 4 categories
                total = [
                    result.get("q_understanding", 1),
                    result.get("k_reasoning", 1),
                    result.get("data_complexity", 1),
                    result.get("sql_complexity", 1)
                ]
                iter_avg = sum(total) / len(total)
                iteration_averages.append(iter_avg)
                all_justifications.append(result.get("justification", "No justification provided."))

            except Exception as e:
                print(f"Error classifying Q{q_id} on iteration {i+1}: {str(e)}")

        #calcular mitjana arodonida
        if iteration_averages:
            final_score = statistics.mean(iteration_averages)
            rounded_score = round(final_score)

            difficulty_map = {1: "SIMPLE", 2: "MODERATE", 3: "CHALLENGING"}
            
            new_item = query.copy()
            new_item["difficulty"] = difficulty_map.get(rounded_score, "ERROR")
            new_item["avg_score"] = round(final_score, 2)
            new_item["justifications"] = all_justifications

            classified_queries.append(new_item)
            print(f"Q{q_id} classified as {new_item['difficulty']} with average score {new_item['avg_score']}.")
        else:
            print(f"No scores obtained for Q{q_id}. Skipping classification.")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(classified_queries, f, indent=4)
    print(f"\nClassificació completada. Resultats guardats a {OUTPUT_FILE}")

if __name__ == "__main__":
    classify_difficulty()

