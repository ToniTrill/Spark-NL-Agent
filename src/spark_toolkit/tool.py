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


class QuerySparkSQLTool(BaseSparkSQLTool, BaseTool):
    """Tool for querying a Spark SQL."""

    name: str = "query_sql_db"
    args_schema: type[BaseModel] = QuerySparkSQLInput
    description: str = """
    Input to this tool is a detailed and correct SQL query, output is a result from the Spark SQL.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
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

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for a specific table."""
        return ", ".join(self.db.get_usable_table_names())


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
    description: str = """Input is an empty string. 
    Output is a categorized list of custom User Defined Functions (UDFs) for Spark SQL.
    Use this to identify if a task requires a custom function instead of standard SQL."""
    
    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not os.path.exists(UDF_PATH):
            return "ERROR: UDF catalog not found."
        
        try:
            with open(UDF_PATH, "r") as f:
                data = json.load(f)
                udfs = data.get("udfs", [])
            
            if not udfs:
                return "No UDFs registered in the catalog."

            #Dividir udf per categories per ajudar al llm
            scalar_udfs = []
            aggregate_udfs = []
            table_udfs = []

            for u in udfs:
                #el nom sempre en minuscules per evitar el ROUTINE_NOT_FOUND
                u_info = f"- {u['name'].lower()}: {u['description']} (In: {u['input_type']}, Out: {u['output_type']})"
                
                # Decidir categoria
                u_id = u.get("id", "")
                if u['output_type'] in ['rset', 'P'] and int(u_id[1:]) > 28:
                    table_udfs.append(u_info)
                elif 25 <= int(u_id[1:]) <= 28:
                    #afegir 'aggregate_' per que al json esta malament
                    agg_name = f"aggregate_{u['name'].lower()}"
                    aggregate_udfs.append(f"- {agg_name}: {u['description']}")
                else:
                    scalar_udfs.append(u_info)

            output = "AVAILABLE SPARK SQL UDFs (Use lowercase names):\n\n"
            
            output += "### 1. SCALAR FUNCTIONS (Use in SELECT or WHERE clauses):\n"
            output += "\n".join(scalar_udfs) + "\n\n"
            
            output += "### 2. AGGREGATE FUNCTIONS (Use for totals/averages. Use these INSTEAD of standard AVG/COUNT):\n"
            output += "\n".join(aggregate_udfs) + "\n\n"
            
            output += "### 3. TABLE FUNCTIONS (UDTFs) (MANDATORY: Use ONLY in FROM clause with TABLE syntax):\n"
            output += "Syntax: SELECT ... FROM function_name(TABLE(SELECT ...), args)\n"
            output += "\n".join(table_udfs)

            return output
        except Exception as e:
            return f"Error processing UDF catalog: {str(e)}"