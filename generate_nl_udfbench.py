from langchain_google_genai import ChatGoogleGenerativeAI
import os, re
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

OUTPUT_PATH = "db/udfbench/udfdev.json"
UDF_PATH = "db/udfbench/engines/pyspark/queries"
CATALOG_PATH="db/udfbench/udf_catalog.json"

load_dotenv()
#Generar les preguntes en llenguatge naural a partir de la weuy en spark sql utiltizan udf
def clean_sql_query(sql):
    """
    Elimina salts de línia, tabulacions i espais múltiples 
    per deixar una query neta en una sola línia.
    """
    # Substitueix qualsevol sequencia de caracters blancs (espais, \n, \t) per un sol espai
    cleaned = re.sub(r'\s+', ' ', sql)
    return cleaned.strip()

def udf_context():
    if not os.path.exists(CATALOG_PATH):
        print(f"Error catalogue not found {CATALOG_PATH}")
        return ""
    
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    context = "TECHNICAL REFERENCE FOR CUSTOM FUNCTIONS (UDFs):\n"
    for udf in data.get("udfs", []):
        context += f"- {udf['name']}: {udf['description']} (Input: {udf['input_type']}, Output: {udf['output_type']})\n"
    return context
def clean_response(content):
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and 'text' in part:
                return part['text']
            elif isinstance(part, str):
                return part
    if isinstance(content, dict):
        return content.get('text', str(content))
    
    return str(content)

def generate_question():
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")
    queries = [1, 2, 3, 4, 7, 8, 9, 12, 13, 14, 15, 16, 18]
    udf = udf_context()
    dataset = []

    for q_id in queries:
        sql_path = os.path.join(UDF_PATH, f"q{q_id}.sql")

        if os.path.exists(sql_path):
            with open(sql_path, "r", encoding="utf-8") as f:
                raw_sql = f.read()

            sql_query = clean_sql_query(raw_sql)
            
            prompt = f"""
            You are a database expert. I will give you a Spark SQL query that uses custom functions (UDFs).
            Your task is to translate this SQL into a natural language question in ENGLISH.

            {udf}

            RULES:
            1. The question must sound like something a human researcher would ask.
            2. Do NOT mention UDF names (like 'Addnoise' or 'U1') or table names in the question.
            3. Focus on the BUSINESS LOGIC (e.g., instead of "extractyear", say "In which year...").
            4. Return ONLY the question. No quotes, no intro.

            SQL Query:
            {sql_query}
            """

            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            raw_text = clean_response(content)

            nl_question = raw_text.strip().replace('"', '')

            dataset.append({
                "question_id": q_id,
                "db_id": "udfbench",
                "question": nl_question,
                "evidence": "No evidence",
                "SQL": sql_query,
                "difficulty": "challenging",
            })
            print(f"query {q_id}trannslated to nl : {nl_question}")
        else:
            print("error")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f :
        json.dump(dataset, f, indent=4)
    print("finished translation")
if __name__ == "__main__":
    generate_question()
