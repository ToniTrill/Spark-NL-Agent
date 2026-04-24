"""Toolkit for interacting with Spark SQL."""

from typing import List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import ConfigDict, Field

from spark_toolkit.prompt import QUERY_CHECKER, QUERY_CHECKER_UDFBENCH

from spark_toolkit.tool import (
    InfoSparkSQLTool,
    ListSparkSQLTool,
    QueryCheckerTool,
    SubmitSparkSQLTool,
    ListUDFSparkSQLTool,
    InvestigateSparkSQLTool
)
from spark_toolkit.spark_sql import SparkSQL


class SparkSQLToolkit(BaseToolkit):
    """Toolkit for interacting with Spark SQL.

    Parameters:
        db: SparkSQL. The Spark SQL database.
        llm: BaseLanguageModel. The language model.
    """

    db: SparkSQL = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)
    use_udf: bool = False
    allowed_udfs: Optional[list] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        checker_template = QUERY_CHECKER_UDFBENCH if self.use_udf else QUERY_CHECKER
        tools = []
        tools.append(ListSparkSQLTool(db=self.db, remind_udf=self.use_udf))
        if self.use_udf:
            tools.append(ListUDFSparkSQLTool(db=self.db, allowed_udfs=self.allowed_udfs))
        tools.append(InfoSparkSQLTool(db=self.db))
        tools.append(QueryCheckerTool(db=self.db, llm=self.llm, template=checker_template))
        #tools.append(InvestigateSparkSQLTool(db=self.db))
        tools.append(SubmitSparkSQLTool(db=self.db))

        return tools