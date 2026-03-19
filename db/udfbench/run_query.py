#!/usr/bin/env python3

import argparse
import os
import sys
import importlib.util

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR

DEFAULT_QUERIES_DIR = os.path.join(BASE_DIR, "engines", "pyspark", "queries")
DEFAULT_UDFS_DIR = os.path.join(BASE_DIR, "engines", "pyspark", "udfs")
DEFAULT_SCHEMA_PATH = os.path.join(BASE_DIR, "engines", "pyspark", "scripts", "pyspark_schema.py")
DEFAULT_LOADS_PATH = os.path.join(BASE_DIR, "engines", "pyspark", "scripts", "pyspark_load.py")
DEFAULT_PARQUET_PATH = os.path.join(BASE_DIR, "dataset", "parquet", "tiny")


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


def load_parquet_files(spark, folder_path, schemas):
    parquet_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))

    if not parquet_files:
        print("No parquet files in this folder")
        return

    list_names = ["artifacts", "artifact_citations", "projects", "projects_artifacts",
                  "artifact_authorlists", "artifact_abstracts", "artifact_authors", "views_stats"]

    for file in parquet_files:
        table_name = os.path.basename(file).replace(".parquet", "")

        if table_name in list_names:
            _schema = schemas.get(f"{table_name}_schema")
            df = spark.read.parquet(file, schema=_schema)

            if table_name == "artifacts":
                df = df.toDF(*["id", "title", "publisher", "journal", "date", "year", "access_mode",
                               "embargo_end_date", "delayed", "authors", "source", "abstract", "type",
                               "peer_reviewed", "green", "gold"])
            elif table_name == "projects":
                df = df.toDF(*["id", "acronym", "title", "funder", "fundingstring", "funding_lvl0",
                               "funding_lvl1", "funding_lvl2", "ec39", "type", "startdate", "enddate",
                               "start_year", "end_year", "duration", "haspubs", "numpubs", "daysforlastpub",
                               "delayedpubs", "callidentifier", "code", "totalcost", "fundedamount", "currency"])
            elif table_name == "projects_artifacts":
                df = df.toDF(*["projectid", "artifactid", "provenance"])
            elif table_name == "artifact_authorlists":
                df = df.toDF(*["artifactid", "authorlist"])
            elif table_name == "artifact_citations":
                df = df.toDF(*["artifactid", "target", "citcount"])
            elif table_name == "artifact_abstracts":
                df = df.toDF(*["artifactid", "abstract"])
            elif table_name == "artifact_authors":
                df = df.toDF(*["artifactid", "affiliation", "fullname", "name", "surname", "rank", "authorid"])
            elif table_name == "views_stats":
                df = df.toDF(*["date", "artifactid", "source", "repository_id", "count"])
            else:
                continue

            df.createOrReplaceTempView(table_name)


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


def run_query(query_id, queries_dir, parquet_path, udfs_dir, schema_path, loads_path, print_results=False):
    query_file = os.path.join(queries_dir, f"{query_id}.sql")
    if not os.path.exists(query_file):
        print(f"Error: Query file not found: {query_file}")
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(schema_path))
    from pyspark_schema import schemas

    sys.path.insert(0, os.path.dirname(loads_path))
    from pyspark_load import load_parquet_files as load_parquet

    # JAR paths for Java/Scala UDFs
    # (Necessary for Query 3)
    java_jar = os.path.join(udfs_dir, "scalar", "extractmonth_java", "target", "JavaUDFjarfile.jar")
    scala_jar = os.path.join(udfs_dir, "scalar", "extractday_scala", "target", "ScalaUDFjarfile.jar")
    jars = f"{java_jar},{scala_jar}" if os.path.exists(java_jar) and os.path.exists(scala_jar) else ""

    builder = SparkSession.builder.appName(f"UDFBench-{query_id}").config("spark.driver.memory", "4g")
    if jars:
        builder = builder.config("spark.jars", jars)
    spark = builder.getOrCreate()

    if os.path.exists(java_jar):
        spark.udf.registerJavaFunction("extractmonth_java", "com.example.udfs.ExtractmonthJava", IntegerType())
    if os.path.exists(scala_jar):
        spark.udf.registerJavaFunction("extractday_scala", "com.example.scalaudfs.ExtractdayScala", IntegerType())

    # Load database files
    load_parquet(spark, parquet_path, schemas)

    register_udfs(spark, udfs_dir)

    # Read and execute query
    with open(query_file, 'r') as f:
        query = f.read()

    print(f"Running query: {query_id}")
    result = spark.sql(query)

    if print_results:
        result.show()

    spark.stop()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a PySpark SQL query with UDFs")
    parser.add_argument("query_id", help="Query ID (e.g., q1, q2, q3)")
    parser.add_argument("--queries-dir", default=DEFAULT_QUERIES_DIR, help="Directory containing SQL queries")
    parser.add_argument("--parquet-path", default=DEFAULT_PARQUET_PATH, help="Path to parquet files")
    parser.add_argument("--udfs-dir", default=DEFAULT_UDFS_DIR, help="Path to UDFs directory")
    parser.add_argument("--schema-path", default=DEFAULT_SCHEMA_PATH, help="Path to schema Python file")
    parser.add_argument("--loads-path", default=DEFAULT_LOADS_PATH, help="Path to loads Python file")
    parser.add_argument("--print-results", action="store_true", help="Print query results")

    args = parser.parse_args()

    run_query(
        args.query_id,
        args.queries_dir,
        args.parquet_path,
        args.udfs_dir,
        args.schema_path,
        args.loads_path,
        args.print_results
    )
