#!/usr/bin/env python3
"""
Test script for natural language to SparkSQL conversion.

Usage:
    python test_nl_query.py

Requirements:
    - GOOGLE_API_KEY environment variable (or .env file)
    - Database files in db/bird-1/
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

import json
from validation import validate

# Configuration
DB_NAME = "superhero"
NL_QUERY = "Which superhero has the most durability published by Dark Horse Comics?"


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
    
    # Run query
    llm = get_llm(provider=Provider.GOOGLE.value)
    agent = get_spark_agent(get_spark_sql(), llm)
    run_nl_query(agent, NL_QUERY, llm)
    
    # Display results
    json_result = process_result()

    model_data = json_result.get('query_result')

    validate(spark, NL_QUERY, model_data)

    print("\nINFERRED SPARKSQL QUERY:")
    print("-" * 40)
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
