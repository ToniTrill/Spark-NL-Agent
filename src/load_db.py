from config import (
    DB_PATH,
    BENCHMARK_FILE
)
import os
import sys
import json
import importlib.util
import sqlite3

from pyspark.sql.types import *
from pyspark.sql.functions import udf
from config import DB_PATH, BENCHMARK_FILE

def load_modules_from_folder(folder_path):
    modules = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            file_path = os.path.join(folder_path, filename)
            module_name = filename[:-3]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                modules[module_name] = module
    return modules

def register_udfs(spark, udfs_dir):
    sys.path.insert(0, udfs_dir + "/aggregate")
    sys.path.insert(0, udfs_dir + "/scalar")
    sys.path.insert(0, udfs_dir + "/table")

    UDFs = load_modules_from_folder(f"{udfs_dir}/scalar")
    UDAFs = load_modules_from_folder(f"{udfs_dir}/aggregate")
    UDTFs = load_modules_from_folder(f"{udfs_dir}/table")

    # Scalar UDFs
    spark.udf.register("addnoise", udf(UDFs["addnoise"].addnoise, DoubleType()))
    spark.udf.register("clean", udf(UDFs["clean"].clean, StringType()))
    spark.udf.register("cleandate", udf(UDFs["cleandate"].cleandate, StringType()))
    spark.udf.register("converttoeuro", udf(UDFs["converttoeuro"].converttoeuro, DoubleType()))
    spark.udf.register("extractclass", udf(UDFs["extractclass"].extractclass, StringType()))
    spark.udf.register("extractcode", udf(UDFs["extractcode"].extractcode, StringType()))
    spark.udf.register("extractday", udf(UDFs["extractday"].extractday, IntegerType()))
    spark.udf.register("extractfunder", udf(UDFs["extractfunder"].extractfunder, StringType()))
    spark.udf.register("extractid", udf(UDFs["extractid"].extractid, StringType()))
    spark.udf.register("extractmonth", udf(UDFs["extractmonth"].extractmonth, IntegerType()))
    spark.udf.register("extractprojectid", udf(UDFs["extractprojectid"].extractprojectid, StringType()))
    spark.udf.register("extractyear", udf(UDFs["extractyear"].extractyear, IntegerType()))
    spark.udf.register("filterstopwords", udf(UDFs["filterstopwords"].filterstopwords, StringType()))
    spark.udf.register("frequentterms", udf(UDFs["frequentterms"].frequentterms, StringType()))
    spark.udf.register("jaccard", udf(UDFs["jaccard"].jaccard, DoubleType()))
    spark.udf.register("jpack", udf(UDFs["jpack"].jpack, StringType()))
    spark.udf.register("jsoncount", udf(UDFs["jsoncount"].jsoncount, LongType()))
    spark.udf.register("jsonparse_q14", udf(UDFs["jsonparse"].jsonparse, StringType()))
    spark.udf.register("jsort", udf(UDFs["jsort"].jsort, StringType()))
    spark.udf.register("jsortvalues", udf(UDFs["jsortvalues"].jsortvalues, StringType()))
    spark.udf.register("keywords", udf(UDFs["keywords"].keywords, StringType()))
    spark.udf.register("log_10", udf(UDFs["log_10"].log_10, DoubleType()))
    spark.udf.register("lowerize", udf(UDFs["lowerize"].lowerize, StringType()))
    spark.udf.register("removeshortterms", udf(UDFs["removeshortterms"].removeshortterms, StringType()))
    spark.udf.register("stem", udf(UDFs["stem"].stem, StringType()))

    # Aggregate UDFs
    spark.udf.register("aggregate_avg", UDAFs["aggregate_avg"].aggregate_avg)
    spark.udf.register("aggregate_count", UDAFs["aggregate_count"].aggregate_count)
    spark.udf.register("aggregate_max", UDAFs["aggregate_max"].aggregate_max)
    spark.udf.register("aggregate_median", UDAFs["aggregate_median"].aggregate_median)

    # Table UDFs
    spark.udtf.register("extractfromdate", UDTFs["extractfromdate"].ExtractFromDate)
    spark.udtf.register("jsonparse", UDTFs["jsonparse"].JsonParse)
    spark.udtf.register("combinations", UDTFs["combinations"].Combinations)
    spark.udtf.register("combinations_q16", UDTFs["combinations_q16"].Combinations_q16)
    spark.udtf.register("extractkeys", UDTFs["extractkeys"].Extractkeys)
    spark.udtf.register("xmlparser", UDTFs["xmlparser"].Xmlparser)
    spark.udtf.register("aggregate_top", UDTFs["aggregate_top"].AggregateTop)
    spark.udtf.register("file_q7", UDTFs["file_q7"].File_q7)
    spark.udtf.register("file_q13", UDTFs["file_q13"].File_q13)
    spark.udtf.register("file_q18", UDTFs["file_q18"].File_q18)

    #UDF repetides en diferets llenguatges

    spark.udf.register("extractday_scala", udf(UDFs["extractday"].extractday, IntegerType()))
    spark.udf.register("extractmonth_java", udf(UDFs["extractmonth"].extractmonth, IntegerType()))


def get_db_path(db_name, benchmark_type):
    if benchmark_type == "bird-1":
        return os.path.join(DB_PATH, "bird-1", db_name, f"{db_name}.sqlite")
    else:
        return os.path.join(DB_PATH, "udfbench")

def load_tables(spark_session, db_name, benchmark_type):
    if benchmark_type == "bird-1":
        load_bird_tables(spark_session, db_name)
    elif benchmark_type =="udfbench":
        load_udfbench_environment(spark_session)

def load_udf_tables(spark_session, db_name):
    db_path = get_db_path(db_name, "udfbench")
    print(f"--- Scanning database: {db_path} ---")
    abs_db_path = os.path.abspath(db_path)
    jdbc_url = f"jdbc:sqlite:{abs_db_path}"
    db_connection = sqlite3.connect(db_path)
    cursor = db_connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    db_connection.close()

    if not tables:
        print("Warning: No tables found in the database!")
        return

    for table in tables:
        df = spark_session.read \
            .format("jdbc") \
            .option("url", jdbc_url) \
            .option("dbtable", table) \
            .option("driver", "org.sqlite.JDBC") \
            .load()

        df.createOrReplaceTempView(table)
        print(f" -> Registered table: '{table}'")

def load_udfbench_environment(spark_session):
    """Càrrega per a UDFBench (Parquet + UDFs)."""
    base_dir = os.path.join(DB_PATH, "udfbench")
    parquet_path = os.path.join(base_dir, "dataset", "parquet", "tiny")
    udfs_dir = os.path.join(base_dir, "engines", "pyspark", "udfs")
    scripts_dir = os.path.join(base_dir, "engines", "pyspark", "scripts")

    print(f"--- Loading UDFBench Parquet environment from: {parquet_path} ---")

    sys.path.insert(0, scripts_dir)
    try:
        import pyspark_schema
        import pyspark_load
        pyspark_load.load_parquet_files(spark_session, parquet_path, pyspark_schema.schemas)
        print(" Parquet tables registered.")
    except ImportError as e:
        print(f" Error: No s'han trobat els scripts a {scripts_dir}: {e}")

    register_udfs(spark_session, udfs_dir)
    
def load_bird_tables(spark_session, db_name):
    db_path = get_db_path(db_name, "bird-1")
    print(f"--- Scanning database: {db_path} ---")
    abs_db_path = os.path.abspath(db_path)
    jdbc_url = f"jdbc:sqlite:{abs_db_path}"
    db_connection = sqlite3.connect(db_path)
    cursor = db_connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    db_connection.close()

    if not tables:
        print("Warning: No tables found in the database!")
        return

    for table in tables:
        df = spark_session.read \
            .format("jdbc") \
            .option("url", jdbc_url) \
            .option("dbtable", table) \
            .option("driver", "org.sqlite.JDBC") \
            .load()

        df.createOrReplaceTempView(table)
        print(f" -> Registered table: '{table}'")


def load_query_info(query_id: int):

    query_data_file = os.path.join(DB_PATH, "bird-1", BENCHMARK_FILE)

    with open(query_data_file, 'r') as f:
        all_queries = json.load(f)

    query_info = None
    for query_entry in all_queries:
        if query_entry['question_id'] == query_id:
            query_info = query_entry
            break

    if query_info is None:
        raise ValueError(f"Query ID {query_id} not found")

    question = " ".join([
        query_info["question"],
        query_info.get("evidence", "")
    ])
    golden_query = query_info["SQL"]
    difficulty = query_info.get("difficulty", "unknown")

    return question, golden_query, difficulty
