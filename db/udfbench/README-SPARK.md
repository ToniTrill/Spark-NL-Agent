# Running Spark Queries

## Setup

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv ~/venvs/udf-venv
source ~/venvs/udf-venv/bin/activate
pip install -r requirements.txt
```

2. Requirements:
   - Python 3.10+
   - PySpark 3.5.5
   - pandas, numpy, nltk, pyarrow, setuptools

## Running Queries

From the project root directory:

```bash
~/venvs/udf-venv/bin/python run_query.py <query_id> [--print-results]
```

### Available Queries

| Query ID | Description |
|----------|-------------|
| q1 | Extract year, month, day from date |
| q2 | Extract funder from projects |
| q3 | Java/Scala UDFs (extractmonth_java, extractday_scala) |
| q4 | Extract code from projects |
| q7 | File table UDF |
| q8 | JSON operations |
| q9 | Keywords extraction |
| q12 | Aggregate operations |
| q13 | File table UDF |
| q14 | JSON parsing |
| q15 | Jaccard similarity |
| q16 | Combinations table UDF |
| q18 | File table UDF |

### Examples

```bash
~/venvs/udf-venv/bin/python run_query.py q1

~/venvs/udf-venv/bin/python run_query.py q1 --print-results
```

## What the Script Does

1. Loads all parquet data files from the configured path into the Spark Catalog (createOrReplaceTempView)
2. Registers all UDFs:
   - Scalar UDFs (Python)
   - Aggregate UDFs (pandas_udf)
   - Table UDFs (UDTFs)
   - Java/Scala UDFs (via JAR)
3. Executes the SQL query and returns results
