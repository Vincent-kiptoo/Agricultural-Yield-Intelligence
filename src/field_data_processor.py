"""
This is a field data processor module.
it ingests sql data into the pandas Dataframe, swaps the column names and merge weather data 
into a unified single pandas DataFrame
"""

import pandas as pd
from src.data_ingestion import create_db_engine, query_data, read_from_web_CSV
from src.logging_config import get_logger

class FieldDataProcessor:

    def __init__(self, config_params) -> None:
        self.db_path = config_params["db_path"]
        self.sql_query = config_params["sql_query"]
        self.columns_to_rename = config_params["columns_to_rename"]
        self.values_to_rename = config_params["values_to_rename"]
        self.weather_map_data = config_params["weather_mapping_csv"]
        self.df = None
        self.engine = None

        self.logger = get_logger(__name__)

        self.logger.info("FieldDataProcessor is initialized")

    def ingest_sql_data(self) ->pd.DataFrame:
        "Fetches Sql data into pandas DataFrame"
        self.engine = create_db_engine(self.db_path)
        self.df = query_data(self.engine, self.sql_query)
        self.logger.info("SQL data is successfully loaded into the pandas DataFrame")
        return self.df
    
    def rename_columns(self) -> None:
        "Swaps the column names that were mislabeled"
        if self.df is None:
            raise ValueError("The field data is empty, Check the SQL database")
        column1, column2 = list(self.columns_to_rename.keys())[0], list(self.columns_to_rename.keys())[1]
        temp_name = "temp_name_for_swap_"
        self.df = self.df.rename(columns={column1: temp_name, column2: column1})
        self.df = self.df.rename(columns={ temp_name: column2})
        self.logger.info(f"Swapped columns: {column1} with {column2}")

    def apply_correction(self, column_name = "Crop_type", abs_column = "Elevation") -> None:
        """
        Converts negative values of elevation feature to absulute intergers
        Edit crop names that were spelled incorrectly
        """

        if self.df is None or self.df.empty:
            raise ValueError("The field data is empty. Execute ingest_sql_data() method first")
        self.df[abs_column] = self.df[abs_column].abs()
        self.logger.info("Converted the negated elavation values to absulte figures.")
        self.df[column_name] = self.df[column_name].str.strip()
        self.df[column_name] = self.df[column_name].replace(self.values_to_rename)
        self.logger.info("Mispelled crop names and extra spaces were found and got fixed")

    def weather_station_mapping(self)-> pd.DataFrame | None:
        "Fetches the weather data from the wen and returns a pandas dataframe"
        return read_from_web_CSV(self.weather_map_data)

    def process(self) -> pd.DataFrame | None:
        "Processes the data into a clean format that is reather for downstrea analysis"
        self.ingest_sql_data()
        if self.df is None:
            raise ValueError("self.df is None")
        self.rename_columns()
        self.apply_correction()
        weather_df = self.weather_station_mapping()
        if weather_df is None:
            raise ValueError("Weather mapping failed")
        self.df = self.df.merge(weather_df, on = "Field_ID", how = "left")
        self.df = self.df.drop(columns = ["Unnamed: 0", "Weather_station", "Field_ID"])
        return self.df


    