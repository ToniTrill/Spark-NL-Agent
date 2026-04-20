#!/usr/bin/env python3
"""
Test script for natural language to SparkSQL conversion. With few shot

"""

import sys
import os,io
import json
import csv
import shutil 
import datetime
import time
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"


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

client = genai.Client()

# Configuration
ALL_AVAILABLE_DATABASES = ['udfbench']
TEST_DATABASES = ['udfbench'] #De quiens Db es treuen les preguntes
QUERY_INDICES = [ 1, 2, 3, 4, 7, 8, 12, 13, 14, 15, 16, 18 ]#num preguntes que testeijar , 8, 9, 12, 13, 14, 15, 16, 18  1, 2, 3, 4, 7, 8,    
REPETITIONS = 1 #fer vaires repeticions i agafar la mitjana per fer estudis per variabilitat de la IA
K =  [3] #probar amb varios valors de K 0,1,2,3,4,5,10
FAISS_PATH = "db/faiss_index_udv" #a on es guarda la cache faiss
JSON_PATH='db/udfbench/udfdev.json' #on estan les pregutnes i respostes de la db
CSV_PATH_OUTPUT = 'results/udf_results_experiment.csv'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(OUTPUT_DIR, f'udf_results_experiment_{now}.csv')

    #inicialitza CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Database', 'K', 'Repetition', 'Accuracy', 'Avg_Input_Tokens', 'Avg_Output_Tokens'])

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
    #Paths necesaris en preguntes del benchmark
    files_to_add = ["db/udfbench/dataset/files/tiny/arxiv.csv", "db/udfbench/dataset/files/tiny/crossref.txt"]
    import shutil
    for f_path in files_to_add:
        target = os.path.basename(f_path)
        if not os.path.exists(target):
            shutil.copy(f_path, target)
            print(f"Fitxer copiat a l'arrel per a UDFs: {target}")

    #Filtrar les N priemres preguntes dificultat challenging
    with open(JSON_PATH, "r") as f:
        all_data = json.load(f)

    llm = get_llm(provider=Provider.GOOGLE.value)
    #toolkit = SparkSQLToolkit(db=get_spark_sql(), llm=llm)
    #agent = get_spark_agent(get_spark_sql(), llm)

    #proba experimetnal
    with open("db/udfbench/udf_mapping.json") as f:
        udf_mapping = json.load(f)

    for db_name in TEST_DATABASES:
        load_tables(spark, db_name, benchmark_type="udfbench")
        #probisional
        spark_sql_instance = get_spark_sql()

        test_questions = []
        for i in QUERY_INDICES:
            test_questions.append(all_data[i-1])

        for k in K:
            for rep in range(1, REPETITIONS +1):
                print(f"\n Executant DB {db_name} K: {k} iteracio numero: {rep}")

                #db que formaran part del few.shot totes excpete la actual
                #fais_db = [db for db in ALL_AVAILABLE_DATABASES if db != db_name]
                fais_db = ALL_AVAILABLE_DATABASES

                #Esborrar faiss anterior
                if os.path.exists("db/faiss_index_udv"):
                    shutil.rmtree("db/faiss_index_udv")
                    print("FAISS cleaned")
                
                vector_store = load_vector(fais_db)
                #Probisional
                #agent = get_spark_agent(get_spark_sql(), llm, use_udf=True)

                correct_count = 0
                total_in_tokens = 0
                total_out_tokens = 0
                for i, test_item in enumerate(test_questions):

                    print(f'\n Pregunta nuemro Q{test_item["question_id"]}')

                    similar=[]

                    #proba exepriemtnal nomes udf de la pregunta actual
                    q_id = str(test_item["question_id"])
                    allowed = udf_mapping.get(q_id)

                    agent = get_spark_agent(spark_sql_instance, llm, use_udf=True, allowed_udfs=None)  # o posar allowed si nomes UDF de la golden query
                    
                    query = test_item["question"]

                    #FI

                    if k == 0 or vector_store is None:
                        raw_similar = None
                    else:
                        raw_similar = vector_store.similarity_search(query, k=k+1) #sempre trobara primera la pregunta actual que descaratem
                            
                    if raw_similar is not None:        
                        similar = [doc for doc in raw_similar if doc.page_content.strip().lower() != query.strip().lower()]
                    run_nl_query(agent, query, llm, similar)
                    time.sleep(2)
                    model_output = process_result()
                    total_in_tokens += model_output.get('input_tokens', 0)
                    total_out_tokens += model_output.get('output_tokens', 0)
                    print(f"   Tokens -> Input: {model_output.get('input_tokens')} | Output: {model_output.get('output_tokens')}")
                    model_data = model_output.get('query_result')

                    is_correct = validate(spark, query, model_data, json_path=JSON_PATH)

                    if is_correct:
                        correct_count += 1
                        print(" Resultat CORRECTE")
                    else:
                        print(f"Resultat INCORRECTE")
                        print(f"DEBUG: Model result: {str(model_data)[:500] if model_data else 'None'}")
                avg_in = total_in_tokens / len(test_questions) if test_questions else 0
                avg_out = total_out_tokens / len(test_questions) if test_questions else 0
                accuracy = (correct_count/len(test_questions)) * 100 if test_questions else 0
                print(f"Resultats de la itracio {accuracy} % d'encerts")

                with open(csv_file, mode='a', newline='') as file: #mode 'a' append
                    writer = csv.writer(file)
                    writer.writerow([db_name, k, rep, accuracy, avg_in, avg_out])

if __name__ == "__main__":
    main()
