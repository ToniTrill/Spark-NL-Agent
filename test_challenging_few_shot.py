#!/usr/bin/env python3
"""
Test script for natural language to SparkSQL conversion. With few shot

"""

import sys
import os

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Provider
from llm import get_llm
from load_db import load_tables
from spark_nl import (
    get_spark_session, get_spark_sql, get_spark_agent,
    run_nl_query, process_result, print_results
)
from utils import ensure_sqlite_jdbc_driver, pretty_print_result

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from google import genai
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import json
from langchain_core.documents import Document
from few_shot import load_vector
from validation import validate

client = genai.Client()

# Configuration
DB_NAME = "superhero"
NL_QUERY = "Which superhero has the most durability published by Dark Horse Comics?"

FAISS_PATH = "db/faiss_index"
FAISS_DB=['california_schools', 'student_club', 'thrombosis_prediction', 'toxicology']
JSON_PATH='db/bird-1/dev.json'

def main():
    print("=" * 60)
    print(" NATURAL LANGUAGE TO SPARKSQL TEST")
    print("=" * 60)
    print(f"\nDatabase: {DB_NAME}")
    print(f"Query: {NL_QUERY}")
    print("=" * 60)
    
    # Setup
    jdbc_jar_path = ensure_sqlite_jdbc_driver()
    spark = get_spark_session(extra_configs={
        "spark.jars": jdbc_jar_path,
        "spark.driver.extraClassPath": jdbc_jar_path,
    })
    load_tables(spark, DB_NAME)

    #faiss carregar consultes i respots de superheores
    vector_store = load_vector(FAISS_DB)

    #Top 5 consultes mes similars.
    if vector_store:
        similar = vector_store.similarity_search(NL_QUERY, k=5)
        #for sim in similar: #print per debug
         #   print()
    else:
        print("no s'ha pogut trobar el el vector sotre")
    # Run query
    llm = get_llm(provider=Provider.GOOGLE.value)
    agent = get_spark_agent(get_spark_sql(), llm)
    run_nl_query(agent, NL_QUERY, llm, similar)
    
    # Display results
    json_result = process_result()

    model_data = json_result.get('query_result')

    validate(spark, NL_QUERY, model_data)

    print("\nINFERRED SPARKSQL QUERY:")
    print("-" * 60)
    sql_query = json_result.get('sparksql_query')
    print(f"\033[94m{sql_query}\033[0m" if sql_query else "\033[91mNo SQL query generated.\033[0m")
    
    print("\nEXECUTION STATUS:")
    print("-" * 40)
    print_results(json_result, print_result=False)
    
    print(f"\nQUERY RESULTS: {NL_QUERY}")
    print("-" * 40)
    if json_result.get('execution_status') == "VALID":
        pretty_print_result(json_result.get('query_result'))
    elif json_result.get('spark_error'):
        print(f"\033[91mError: {json_result.get('spark_error')}\033[0m")
    else:
        print("No results available.")
    
    spark.stop()
    print("\nDone!")
    return json_result


if __name__ == "__main__":
    main()
