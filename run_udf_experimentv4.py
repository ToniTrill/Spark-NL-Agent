#!/usr/bin/env python3
"""
Test script for natural language to SparkSQL conversion. With few shot

"""

import sys
import os,io,re
import json
import csv
import shutil 
import datetime
import time
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

OUTPUT_DIR = "results/csv"

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from google import genai
from google.genai import types
from langchain_core.documents import Document

import config
from config import Provider
from llm import get_llm
from load_db import load_tables
from spark_nl import (
    get_spark_session, get_spark_sql, get_spark_agent,
    run_nl_query, process_result, print_results
)

from utils import ensure_sqlite_jdbc_driver, pretty_print_result

from few_shot import load_vector
from validation import validate


# Configuration
QUERY_INDICES = [ 1, 2, 3, 4, 7, 8, 12, 13, 14, 15, 16, 18 ]#num preguntes que testeijar , 8, 9, 12, 13, 14, 15, 16, 18  1, 2, 3, 4, 7, 8,    
REPETITIONS = 1 #fer vaires repeticions i agafar la mitjana per fer estudis per variabilitat de la IA
FAISS_PATH = "db/faiss_index_udv" #a on es guarda la cache faiss
JSON_PATH='db/udfbench/udfdev_v2.json' #on estan les pregutnes i respostes de la db
OUTPUT_RESULTS_PATH = f'results/udf_results_experiment_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_v4.json'
PROGRESS_LOG_PATH = f'results/experiment_progress_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

def log_progress(message):
    """Escriu un missatge al fitxer de log i al terminal per traçar l'execució."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(f"\033[94m{full_message}\033[0m") # Blau al terminal
    with open(PROGRESS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")

def extract_udfs_from_sql(sql_query):
    """Extreu noms de funcions del SQL ignorant paraules clau estàndard."""
    if not sql_query: return []
    # Netegem backticks i busquem paraules abans d'un parèntesi
    found = re.findall(r'([a-zA-Z0-9_]+)\s*\(', sql_query.lower().replace('`', ''))
    standard_sql = {
        'select', 'from', 'where', 'group', 'by', 'order', 'count', 'sum', 
        'avg', 'min', 'max', 'table', 'and', 'or', 'not', 'distinct', 'as', 'join', 'on'
    }
    return list(set([f for f in found if f not in standard_sql]))

def main():

    # Setup
    jdbc_jar = ensure_sqlite_jdbc_driver()
    udf_base = "db/udfbench"
    java_jar = os.path.abspath(os.path.join(udf_base, "engines/pyspark/udfs/scalar/extractmonth_java/target/JavaUDFjarfile.jar"))
    scala_jar = os.path.abspath(os.path.join(udf_base, "engines/pyspark/udfs/scalar/extractday_scala/target/ScalaUDFjarfile.jar"))
    spark_jars = f"{jdbc_jar},{java_jar},{scala_jar}"

    spark = get_spark_session(extra_configs={
        "spark.jars": spark_jars,
        "spark.sql.execution.arrow.pyspark.enabled": "false", # Q14
        "spark.sql.execution.pythonUDF.arrow.enabled": "false",
        "spark.driver.extraClassPath": jdbc_jar,
        "spark.driver.memory": "4g",
        "spark.executorEnv.PYTHONIOENCODING": "utf-8",
        "spark.executorEnv.PYTHONUTF8": "1",
        "spark.yarn.appMasterEnv.PYTHONIOENCODING": "utf-8"
    })
    log_progress("Iniciant Setup de Spark i Jars...")
    #Paths necesaris en preguntes del benchmark
    files_to_add = ["db/udfbench/dataset/files/tiny/arxiv.csv", "db/udfbench/dataset/files/tiny/crossref.txt"]
    import shutil
    for f_path in files_to_add:
        target = os.path.basename(f_path)
        if not os.path.exists(target):
            shutil.copy(f_path, target)
            print(f"Fitxer copiat a l'arrel per a UDFs: {target}")

    #Filtrar les N priemres preguntes dificultat challenging
    with open(JSON_PATH, "r", encoding='utf-8') as f:
        all_data = json.load(f)


    load_tables(spark, "udfbench", benchmark_type="udfbench")
    spark_sql_instance = get_spark_sql()
    llm = get_llm(provider=Provider.GOOGLE.value)
    # Filtrar preguntes segons QUERY_INDICES
    test_questions = []
    for i in QUERY_INDICES:
        test_questions.append(all_data[i-1])

    results_log = []
    correct_count = 0
    total_in_tokens = 0
    total_out_tokens = 0
    log_progress(f"Iniciant bucle d'execució: {len(test_questions)} preguntes a processar.")
    print(f"Iniciant experiment amb {len(test_questions)} preguntes, Repeticions={REPETITIONS}")
    for i, test_item in enumerate(test_questions):
        config.metrics.clear()

        q_id = test_item["question_id"]
        difficulty = test_item.get("difficulty", "unknown")
        print(f'\n Pregunta nuemro Q{test_item["question_id"]} amb dificultat {difficulty}')
        log_progress(f"Processant Q{q_id} ({i+1}/{len(test_questions)}) | Dificultat: {difficulty}")
        agent = get_spark_agent(spark_sql_instance, llm, use_udf=True, allowed_udfs=None)  # o posar allowed si nomes UDF de la golden query
        
        query = test_item["question"]
        log_progress(f"Cridant a l'agent per a Q{q_id}...")
        run_nl_query(agent, query, llm)
        log_progress(f"CridaAD a l'agent per a Q{q_id}...")
        model_output = process_result()
        predicted_sql = model_output.get('sparksql_query', "")
        gold_udfs = extract_udfs_from_sql(test_item["SQL"])
        predicted_udfs = extract_udfs_from_sql(predicted_sql)

        is_correct = validate(spark, test_item["question"], model_output.get('query_result'), json_path=JSON_PATH)

        #guardar metadades
        log_entry = {
                "question_id": q_id,
                "difficulty": difficulty,
                "question": test_item["question"],
                "gold_sql": test_item["SQL"],
                "predicted_sql": predicted_sql,
                "gold_udfs": gold_udfs,
                "predicted_udfs": predicted_udfs,
                "is_functional_correct": is_correct,
                "tokens": {
                    "input": model_output.get('input_tokens', 0),
                    "output": model_output.get('output_tokens', 0)
                },
            }
        results_log.append(log_entry)
        print(f"   UDFs Gold: {gold_udfs}")
        print(f"   UDFs Pred: {predicted_udfs}")
        print(f"   Correcte: {is_correct}")
        correct_count += int(is_correct)
        total_in_tokens += model_output.get('input_tokens', 0)

        total_out_tokens += model_output.get('output_tokens', 0)
        print(f"   Tokens -> Input: {model_output.get('input_tokens')} | Output: {model_output.get('output_tokens')}")
    avg_in = total_in_tokens / len(test_questions) if test_questions else 0
    avg_out = total_out_tokens / len(test_questions) if test_questions else 0
    accuracy = (correct_count/len(test_questions)) * 100 if test_questions else 0
    print(f"Resultats de la itracio {accuracy} % d'encerts")

    with open(OUTPUT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=4)
    print(f"\n========================================")
    print(f"EXPERIMENT FINALITZAT")
    print(f"Accuracy Total: {accuracy:.2f}%")
    print(f"Mitjana Tokens Input: {avg_in:.1f}")
    print(f"========================================")
    print(f"\nExperiment finalitzat. Resultats guardats a: {OUTPUT_RESULTS_PATH}")
if __name__ == "__main__":
    main()
