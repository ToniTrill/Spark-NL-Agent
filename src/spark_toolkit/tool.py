# flake8: noqa
"""Tools for interacting with Spark SQL."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator, ConfigDict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from spark_toolkit.prompt import QUERY_CHECKER
from spark_toolkit.spark_sql import SparkSQL

import json, os

UDF_PATH="db/udfbench/udf_catalog.json"

class BaseSparkSQLTool(BaseModel):
    """Base tool for interacting with Spark SQL."""

    db: SparkSQL = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class QuerySparkSQLInput(BaseModel):
    query: str = Field(
        ...,
        description="A fully formed Spark SQL query string.",
    )

    model_config = ConfigDict(extra="forbid")


class InfoSparkSQLInput(BaseModel):
    table_names: str = Field(
        ...,
        description="Comma-separated list of table names to describe.",
    )

    model_config = ConfigDict(extra="forbid")


class QueryCheckerInput(BaseModel):
    query: str = Field(
        ...,
        description="A fully formed Spark SQL query string to validate.",
    )

    model_config = ConfigDict(extra="forbid")


class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SubmitSparkSQLTool(BaseSparkSQLTool, BaseTool):
    """Tool for querying a Spark SQL."""

    name: str = "submit_final_query"
    args_schema: type[BaseModel] = QuerySparkSQLInput
    description: str = """
    MANDATORY: This is the ONLY tool to submit your final answer. 
    Use it ONLY when you have verified your query and are 100% certain it is correct.
    Calling this tool ends the session and sends the query for final evaluation.
    If you have any doubt about the data or column names, use 'investigate_sql_db' or 'schema_sql_db' first.
    Input must be a complete and valid Spark SQL query.
    """

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the results or an error message."""
        return self.db.run_no_throw(query)


class InfoSparkSQLTool(BaseSparkSQLTool, BaseTool):
    """Tool for getting metadata about a Spark SQL."""

    name: str = "schema_sql_db"
    args_schema: type[BaseModel] = InfoSparkSQLInput
    description: str = """
    Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.
    Be sure that the tables actually exist by calling list_tables_sql_db first!

    Example Input: "table1, table2, table3"
    """

    def _run(
        self,
        table_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for tables in a comma-separated list."""
        return self.db.get_table_info_no_throw(table_names.split(", "))


class ListSparkSQLTool(BaseSparkSQLTool, BaseTool):
    """Tool for getting tables names."""

    name: str = "list_tables_sql_db"
    args_schema: type[BaseModel] = EmptyInput
    description: str = "Input is an empty string, output is a comma separated list of tables in the Spark SQL."
    remind_udf: bool = False

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for a specific table."""
        tables = ", ".join(self.db.get_usable_table_names())
        if self.remind_udf:
            return tables + "\n\n NEXT REQUIRED STEP: Call `list_udf_sql_db` (empty input) before writing any query. Standard SQL functions are disabled in this environment."
        return tables


class QueryCheckerTool(BaseSparkSQLTool, BaseTool):
    """Use an LLM to check if a query is correct.
    Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/"""

    template: str = QUERY_CHECKER
    llm: BaseLanguageModel
    llm_chain: Any = Field(init=False)
    name: str = "query_checker_sql_db"
    args_schema: type[BaseModel] = QueryCheckerInput
    description: str = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with query_sql_db!
    """

    @model_validator(mode="before")
    @classmethod
    def initialize_llm_chain(cls, values: Dict[str, Any]) -> Any:
        if "llm_chain" not in values:
            from langchain_core.output_parsers import StrOutputParser
            template_to_use = values.get("template", QUERY_CHECKER)
            prompt = PromptTemplate(
                template=template_to_use, input_variables=["query"]
            )
            llm = values.get("llm")
            values["llm_chain"] = prompt | llm | StrOutputParser()

        return values

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the LLM to check the query."""
        return self.llm_chain.invoke(
            {"query": query}, config={"callbacks": run_manager.get_child() if run_manager else None}
        )

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return await self.llm_chain.ainvoke(
            {"query": query}, config={"callbacks": run_manager.get_child() if run_manager else None}
        )
class ListUDFSparkSQLTool(BaseSparkSQLTool, BaseTool):
    name: str = "list_udf_sql_db"
    args_schema: type[BaseModel] = EmptyInput
    allowed_udfs: Optional[list] = None 
    description: str = """MANDATORY: Call this tool BEFORE writing any SQL query.
    Returns all custom UDFs that REPLACE standard Spark functions in this environment.
    Standard functions (COUNT, AVG, YEAR, MONTH, DAY) are DISABLED — using them will cause errors.
    You MUST consult this catalog to know which custom functions to use instead."""
        
    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        catalog_path = "db/udfbench/udf_catalog_v2.json"
        if not os.path.exists(catalog_path):
            return "ERROR: UDF catalog not found. Run the generator script first."

        with open(catalog_path, "r") as f:
            data = json.load(f)
            udfs = data.get("udfs", [])
        print(f"\n[DEBUG] Carregades {len(udfs)} UDFs del catàleg.")

        # Filtrem segons les UDFs permeses per la Golden Query actual
        if self.allowed_udfs:
            allowed_lower = [x.lower() for x in self.allowed_udfs]
            udfs = [u for u in udfs if u['name'].lower() in allowed_lower or f"aggregate_{u['name'].lower()}" in allowed_lower]
            print(f"[DEBUG] Després del filtre queden: {len(udfs)} UDFs.")

        #  Organitzem per l'IA
        sections = {"scalar": [], "aggregate": [], "table": []}
        for u in udfs:
            name = u['name'].lower()
            category = u.get('category', 'scalar').lower()
            if u['category'] == 'aggregate' and not name.startswith('aggregate_'):
                name = f"aggregate_{name}"
            
            line = f"- {name}: {u['description']} (In: {u['input_type']}, Out: {u['output_type']})"
            if category in sections:
                sections[category].append(line)
            else:
                sections["scalar"].append(line)
           
        output = "AVAILABLE SPARK SQL UDFs (Use lowercase names):\n"

        output += "### 1. SCALAR FUNCTIONS (Use in SELECT or WHERE clauses):\n"
        output += ("\n".join(sections["scalar"]) if sections["scalar"] else "None available" + "\n\n")
        
        output += "### 2. AGGREGATE FUNCTIONS (Use for totals/averages. Use these INSTEAD of standard AVG/COUNT):\n"
        output += ("\n".join(sections["aggregate"]) if sections["aggregate"] else "None available" + "\n\n")
        
        output += "### 3. TABLE FUNCTIONS (UDTFs) (MANDATORY: Use ONLY in FROM clause with TABLE syntax):\n"
        output += "Syntax: SELECT ... FROM function_name(TABLE(SELECT ...), args)\n"
        output += ("\n".join(sections["table"] if sections["table"] else "None available" ))

        return output

"""
    except Exception as e:
        return f"Error processing UDF catalog: {str(e)}"
        output = "AVAILABLE SPARK SQL UDFs (Use lowercase names):\n\n"
        output += " @udtf(returnType=\"column1 STRING, column2 STRING,column3 STRING\") class File_q7: def eval(self, file_path: str, file_type:str):"
        output += "# U41.	File: parses an external file (csv, xml, json) and returns a table def file(file_path: str, file_type:str): "
        output += "@udtf(returnType=\"publicationdoi string, fundinginfo string\") class JsonParse: def eval(self, json_content: list, key1: str, key2: str):"
        output += "@udtf(returnType=\"record string\") class Xmlparser: def eval(self, xml_content: list, root_name: str):"
        output += "def eval(self, rows: Row, top_n: int, group_col: str,group_col2:str, value_col: str): @udtf(returnType=\"group_column1: string, group_column2: string, top_s: double\") class AggregateTop:"
        output += "@udtf(returnType=\"pubid string, pubdate string, projectstart string, projectend string, funder string, fclass string, projectid string,authorpair string\") class Combinations_q16: def eval(self, vals: Row, N:int):"
        output += "@udtf(returnType=\"publicationdoi string, fundinginfo string\") class Extractkeys: def eval(self,jvals:list,key1:str,key2:str):"
        output += "@udtf(returnType=\"column1 STRING, column2 STRING,column3 STRING\") class File_q7: def eval(self, file_path: str, file_type:str):"
        output += "# U41.	File: parses an external file (csv, xml, json) and returns a table  def file(file_path: str, file_type:str):    "

        output += "@pandas_udf(\"double\") def aggregate_median(values: pd.Series) -> float:" 
        output += "@pandas_udf(\"string\") def aggregate_max(values: pd.Series) -> float:"
        output += "@pandas_udf(\"long\") def aggregate_count(values: pd.Series) -> int:"
        output += "@pandas_udf(\"double\") def aggregate_avg(values: pd.Series) -> float:"

        output += "# U1. Add_noise : adds gaussian noise to a value and returns a float def addnoise(val:int)->float:"
        output += "# U2.	Clean: Performs a simple data cleaning task on the string tokens of a json list def clean(val: str)->str:"
        output += " # U3.	Cleandate: Reads a date and converts it to a common format if it is not, handles also dirty dates def cleandate(pubdate: str)->str:"
        output += " # U4.	Converttoeuro: : Converts currency to euro, returns a float def converttoeuro(x:float,y:str)->float:"
        output += "# U5.	Extractclass: extracts class from string with format funder::class::projectid  def extractclass(project:str)->str:"
        
        output += "# U6.	Extractcode: Processes a structured string containing the funder’s id, the funding class and the project id, and extracts the project id def extractcode(project: str)->str:"
        output += "# U7.	Extractday: Reads a date (as a string) and extracts an integer with the day def extractday(arg: str) -> int:"
        output += "# U8.	Extractfunder: extracts funder from string with format funder::class::projectid def extractfunder(project:str)->str:"
        output += "def extractid(project:str)->str:"
        output += "# U10. Extractmonth: Reads a date (as a string) and extracts an integer with the month def extractmonth(arg: str) -> int:"

        output += "# U11.	Extractprojectid: Processes a text snippet and extracts a 6 digit project identifier  def extractprojectid(input: str)->str:"
        output += "# U12.  Extractyear : Reads a date (as a string) and extracts an integer with the year def extractyear(arg: str) -> int:"
        output += "# U13.	Filterstopwords: It processes an input text and returns it after removing the stopwords, using a list of stopwords _stopwords=set([r\".\", r\"_\", r\"stessi\", r\"można\", r\"einseitiger\", r\"wären\", r\"fÛr\", r\"olette\","
        output += "# U14.	Frequentterms: Returns a space separated text containing the most N\% frequent tokens def frequentterms(input:str,N:int)->str:"
        output += "# U15.	Jaccard: Processes two json lists with tokens and calculated the jaccard distance def jaccard(arg1:str,arg2:str)->float:"

        output += "# U16.	Jpack: Converts a string to a json list with tokens def jpack(input:str)->str:"
        output += "# U17.	Jsoncount: Returns the length of a json list def jsoncount(jval: str) -> int:"
        output += "# U18.	Jsonparse: Parses a json dict per time and returns a string with the value def jsonparse(json_content: str,key1: str)->str:"
        output += "# U19.	Jsort: processes a json list and returns a sorted json list  def jsort(jval:str)->str:"
        output += "# U20.	Jsortvalues: processes a json list where each value contains more than one tokens, sorts the tokens in each value  def jsortvalues(jval:str)->str:"

        output += "# U21.	Keywords: Removes any punctuation from text and returns the keywords in one string  def keywords(input:str)->str:"
        output += "# U22.	Log_10: Calculates and returns the logarithm def log_10(x:float)->float:"
        output += "# U23. Lowerize: Converts to lower case the input text def lowerize(val: str)->str:"
        output += "# U24.	Removeshortterms:  processes a json list where each value contains more than one tokens and removes tokens with length less than 3 chars  def removeshortterms(jval:str)->str:"
        output += "# U25.	Stem: Stems the input text using Porter2 stemming algorithm. def stem(input:str)->str:"
        return output"""
""""

    return output
"""
class InvestigateSparkSQLTool(BaseSparkSQLTool, BaseTool):
    name: str = "investigate_sql_db"
    args_schema: type[BaseModel] = QuerySparkSQLInput
    description: str = """
    Use this tool to explore the database and verify your assumptions. 
    Returns the actual data from Spark SQL (use LIMIT 5 to avoid large outputs).
    Use it to check:
    - If a UDF or TVF returns the columns you expect.
    - The actual format of dates or string values.
    - If your joins are working before submitting.
    Results from this tool are for your reasoning only and DO NOT count as a final answer.
    """

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.db.run_no_throw(query, _no_early_exit=True)