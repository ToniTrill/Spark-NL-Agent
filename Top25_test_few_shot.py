#!/usr/bin/env python3
"""
Test script for natural language to SparkSQL conversion. With few shot

"""

import sys
import os
import json
from dotenv import load_dotenv

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
DB_NAME = "superhero" #sobre quina db es fan les preguntes
FAISS_PATH = "db/faiss_index" #a on es guarda la cache faiss
FAISS_DB=['california_schools', 'student_club', 'thrombosis_prediction', 'toxicology'] #DB per cache faiss['superhero']
JSON_PATH='db/bird-1/dev.json' #on estan les pregutnes i respostes de la db
N_QUESTIONS=25 #num preguntes que testeijar


def main():

    # Setup
    jdbc_jar_path = ensure_sqlite_jdbc_driver()
    spark = get_spark_session(extra_configs={
        "spark.jars": jdbc_jar_path,
        "spark.driver.extraClassPath": jdbc_jar_path,
    })
    load_tables(spark, DB_NAME)

    #faiss carregar consultes i respots de superheores
    vector_store = load_vector(FAISS_DB)

    #Filtrar les N priemres preguntes dificultat challenging
    with open(JSON_PATH, "r") as f:
        all_data = json.load(f)

    test_questions = [
        item for item in all_data
        if item["db_id"] == DB_NAME and item.get("difficulty") == "challenging"
    ][:N_QUESTIONS]

    llm = get_llm(provider=Provider.GOOGLE.value)
    agent = get_spark_agent(get_spark_sql(), llm)

    results = []

    #test de les N preguntes
    for i, test_item in enumerate(test_questions):
        query = test_item["question"]
        print(f"\n[{i+1}/{N_QUESTIONS}] Executant: {query}")

        similar = vector_store.similarity_search(query, k=5)
        config.metrics = {}

        run_nl_query(agent, query, llm, similar)    #fer la consulta a la IA
        #validar
        model_output = process_result()

        print_results(model_output, print_result=False)

        model_data = model_output.get('query_result')
        correct = validate(spark, query, model_data)

        if correct:
            print("<<<< CORRECTE >>>>>")
        else:
            print("++++ INCORRECTE +++++")
            golden_sql = next(item["SQL"] for item in all_data if item["question"] == query)
            try:
                expected_data = spark.sql(golden_sql).collect()
                print(f"   Esperat: {expected_data[:1]}")
            except:
                print(" ERROR al Golden SQL")

            # Si model_data és None, posem un missatge en lloc de tallar la llista
            if model_data is not None:
                print(f"   Obtingut: {model_data[:1]}")
            else:
                print("   Obtingut: ERROR (La IA ha generat un SQL invàlid)")
                    
        results.append({
            "question": query,
            "correct": correct,
            "sql_ia": model_output.get('sparksql_query')
        })
    
    #print(results)
    print("\n"+ "//// Resultats ///////////" + "\n")
    total_correct = sum(1 for r in results if r["correct"])
    print(f"\n Encerts: {total_correct} de {len(results)}")
    print(f"\n Accuracy: {(total_correct/len(results))*100}%")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
