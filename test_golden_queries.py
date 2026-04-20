#!/usr/bin/env python3
"""
Script CORREGIT per validar la infraestructura SparkSQL executant les Golden Queries.
"""

import sys
import os
import io
import json
import time
from dotenv import load_dotenv

# --- CONFIGURACIÓ D'ENTORN CRÍTICA PER A WINDOWS / PYSPARK ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# Això és el que faltava: dir-li a Spark que usi el mateix Python que aquest script
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# -----------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
load_dotenv()

from load_db import load_tables
from spark_nl import get_spark_session
from utils import ensure_sqlite_jdbc_driver
from validation import validate

JSON_PATH = 'db/udfbench/udfdev.json'
QUERY_INDICES = [1, 2, 3, 4, 7, 8, 12, 13, 14, 15, 16, 18]
DB_NAME = 'udfbench'

def main():
    print("--- Iniciant Validació de Golden Queries (Entorn Corregit) ---")

    # setup de Spark
    jdbc_jar = ensure_sqlite_jdbc_driver()
    udf_base = "db/udfbench"
    java_jar = os.path.abspath(os.path.join(udf_base, "engines/pyspark/udfs/scalar/extractmonth_java/target/JavaUDFjarfile.jar"))
    scala_jar = os.path.abspath(os.path.join(udf_base, "engines/pyspark/udfs/scalar/extractday_scala/target/ScalaUDFjarfile.jar"))
    spark_jars = f"{jdbc_jar},{java_jar},{scala_jar}"

    spark = get_spark_session(extra_configs={
        "spark.jars": spark_jars,
        "spark.sql.execution.arrow.pyspark.enabled": "false", 
        "spark.sql.execution.pythonUDF.arrow.enabled": "false",
        "spark.driver.extraClassPath": jdbc_jar,
        "spark.driver.memory": "4g",
        "spark.executorEnv.PYTHONIOENCODING": "utf-8",
        "spark.executorEnv.PYTHONUTF8": "1"
    })

    files_to_add = ["db/udfbench/dataset/files/tiny/arxiv.csv", "db/udfbench/dataset/files/tiny/crossref.txt"]
    import shutil
    for f_path in files_to_add:
        target = os.path.basename(f_path)
        if not os.path.exists(target):
            try:
                shutil.copy(f_path, target)
                print(f"Fitxer preparat: {target}")
            except: pass

    # carregar dades
    with open(JSON_PATH, "r") as f:
        all_data = json.load(f)

    load_tables(spark, DB_NAME, benchmark_type="udfbench")

    # executar
    correct_count = 0
    print(f"\nAnalitzant {len(QUERY_INDICES)} preguntes...\n")

    for idx in QUERY_INDICES:
        item = all_data[idx-1]
        q_id = item["question_id"]
        question = item["question"]
        golden_sql = item["SQL"]

        print(f"Q{q_id}...", end=" ", flush=True)
        
        try:
            # executar el sql
            df_result = spark.sql(golden_sql)
            data_result = df_result.limit(50001).collect()
            
            formatted_data = []
            for row in data_result:
                formatted_data.append(tuple(map(str, row.asDict().values())))

            # validacio
            is_correct = validate(spark, question, formatted_data, json_path=JSON_PATH)

            if is_correct:
                print("[OK]")
                correct_count += 1
            else:
                print("[FALLA VALIDACIÓ]")

        except Exception as e:
            print(f"[ERROR EXECUTANT]: {str(e)[:100]}")

    accuracy = (correct_count / len(QUERY_INDICES)) * 100
    print("\n" + "="*40)
    print(f"RESULTAT FINAL GOLDEN: {accuracy:.2f}% d'encerts")
    print("="*40)

if __name__ == "__main__":
    main()