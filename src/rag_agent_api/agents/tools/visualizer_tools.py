import json

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class TableInput(BaseModel):
    index: str = Field(description="names for table rows. Example: '1, 2, 3'")
    columns_names: str = Field(description="column names for table columns. Example: 'A, B, C'")
    data: str = Field(description="the values of each cell in the table. Example: '100, 200, 300'")


@tool("table creator tool", args_schema=TableInput, return_direct=True)
def table_creator(index: str, columns_names: str, data: str) -> str:
    """Создает таблицу из строковых данных.

        Аргументы:
            index: строка с индексами через запятую (например, "1, 2, 3, 4"). Индексы должны быть всегда. Если их нельзя выделить из текста, то индкексы должны идти по порядку
            columns_names: строка с названиями столбцов через запятую (например, "names, count").
            data: строка с данными, где строки разделены ';', а значения в строках — запятыми.
                  Количество значений в каждой строке должно совпадать с количеством столбцов!

        Возвращает:
            таблицу в формате json
        """

    index = [idx.strip() for idx in index.split(',')]
    columns = [col.strip() for col in columns_names.split(',')]

    # Разбиваем данные на строки и значения
    data_rows = [row.strip() for row in data.split(';')]
    data_values = [[val.strip() for val in row.split(',')] for row in data_rows]

    values_t = np.array(data_values)
    print("index", index, "columns ", columns, "VALUE T", values_t)
    df = pd.DataFrame(values_t, columns=columns, index=index)
    df.to_csv("table_test.csv", index=False)
    json_str = df.to_json(orient='records', force_ascii=False)
    print("json str", json_str)
    data = {
        "visualization": {
            "type": "table",
            "columns": df.columns.tolist(),
            "data": json.loads(json_str)
        }
    }

    print("data", data)
    return json.dumps(data)
