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
    """
    """
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
    Always use this tool before executing a query with submit_final_query
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
            return "ERROR: UDF catalog not found."

        with open(catalog_path, "r", encoding="utf-8") as f:
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
            
            if category in sections:
                sections[category].append(f"- {name}")
            else:
                sections["scalar"].append(f"- {name}")
           
        output = "AVAILABLE SPARK SQL UDFs (Use lowercase names):\n"

        output += "### 1. SCALAR FUNCTIONS (Use in SELECT or WHERE clauses):\n"
        output += ("\n".join(sections["scalar"]) if sections["scalar"] else "None available" + "\n\n")
        
        output += "### 2. AGGREGATE FUNCTIONS (Use for totals/averages. Use these INSTEAD of standard AVG/COUNT):\n"
        output += ("\n".join(sections["aggregate"]) if sections["aggregate"] else "None available" + "\n\n")
        
        output += "### 3. TABLE FUNCTIONS (UDTFs) (MANDATORY: Use ONLY in FROM clause with TABLE syntax):\n"
        output += "Syntax: SELECT ... FROM function_name(TABLE(SELECT ...), args)\n"
        output += ("\n".join(sections["table"] if sections["table"] else "None available" ))

        return output

        
class GetUDFCodeTool(BaseSparkSQLTool, BaseTool):
    name: str = "get_udf_source_code"
    args_schema: type[BaseModel] = QuerySparkSQLInput
    description: str = "Retrieves the source code of a specified User Defined Function (UDF) registered in the Spark session. Input should be the name of the UDF."

    def _run(self, udf_name: str, run_manager: Optional[any] = None) -> str:
        base_path = "db/udfbench/engines/pyspark/udfs"
        categories = ["scalar", "aggregate", "table"]
        
        udf_name_clean = udf_name.lower()
        found_path = None

        possible_filenames = [
            f"{udf_name_clean}.py",
            f"aggregate_{udf_name_clean}.py",
            f"file_{udf_name_clean}.py"
        ]

        for category in categories:
            category_dir = os.path.join(base_path, category)
            if not os.path.isdir(category_dir):
                continue
                
            for fname in possible_filenames:
                p = os.path.join(category_dir, fname)
                if os.path.exists(p):
                    found_path = p
                    break
            
            if found_path: break

            for file in os.listdir(category_dir):
                if udf_name_clean in file.lower() and file.endswith(".py"):
                    found_path = os.path.join(category_dir, file)
                    break
            
            if found_path: break

        if found_path:
            try:
                with open(found_path, "r", encoding="utf-8") as f:
                    code = f.read()
                return f"Source code found in '{found_path}':\n\n```python\n{code}\n```"
            except Exception as e:
                return f"Error reading file {found_path}: {str(e)}"
        
        return f"UDF source code for '{udf_name}' not found in categories {categories}."
            
class ReadFileTool(BaseSparkSQLTool, BaseTool):
    name: str = "read_file"
    args_schema: type[BaseModel] = QuerySparkSQLInput
    description: str = "Reads the content of a specified file from the local filesystem. Input should be the relative path to the file."

    def _run(self, path: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            if not os.path.exists(path):
                return f"Error: File at {path} not found."
            
            with open(path, "r", encoding="utf-8") as f:
                content = f.read(2000) 
                return f"Content of {path} (first 2000 chars):\n\n{content}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
        
class UDFCanaryTool(BaseSparkSQLTool, BaseTool):
    name: str = "udf_canary"
    args_schema: type[BaseModel] = QuerySparkSQLInput
    description: str = "Executes a UDF with a single sample value to see the output. Use this to verify how a UDF behaves before writing the full SQL."

    def _run(self, udf_name: str, sample_input: str) -> str:
        try:
            result = self.db.spark.sql(f"SELECT {udf_name}('{sample_input}')").collect()
            return f"Test result for {udf_name}('{sample_input}'): {result[0][0]}"
        except Exception as e:
            return f"Canary test failed: {str(e)}"
        


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