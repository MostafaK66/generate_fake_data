from pathlib import Path
import pandas as pd
from typing import Union, List


class DataPreprocessor:
    def __init__(self):
        pass

    def read_data(self, filename):

        filename = filename if isinstance(filename, Path) else Path(filename)

        return pd.read_csv(filename, dtype={"PI": str})

    def split_and_sort(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Split the dataframe into three based on TicketProject values and sort by TicketCreatedDate.

        :param df: Input dataframe.
        :return: Three dataframes for ADA_Project_1, ADA_Project_2, and ADA_Project_3.
        """

        project_names = ['ADA_Project_1', 'ADA_Project_2', 'ADA_Project_3']
        result_dfs = []

        for project in project_names:
            project_df = df[df['TicketProject'] == project].sort_values(by='TicketCreatedDate')
            result_dfs.append(project_df)

        return tuple(result_dfs)

    def done_tickets_per_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the number of 'Done' tickets based on the TicketCreatedDate.

        :param df: Input dataframe with at least 'TicketStatus' and 'TicketCreatedDate' columns.
        :return: Dataframe with an additional 'DoneTicketsCount' column.
        """
        required_col = "TicketStatus"
        if required_col not in df.columns:
            print(f"Required column ('{required_col}') not found in the dataframe")
            return df


        done_tickets_per_date = (
            df[df["TicketStatus"] == "Done"]
            .groupby("TicketCreatedDate")
            .size()
            .reset_index(name="DoneTicketsCount")
        )


        df = df.merge(
            done_tickets_per_date, on="TicketCreatedDate", how="left"
        )


        df["DoneTicketsCount"] = df["DoneTicketsCount"].fillna(0).astype(int)

        return df

    def flow_per_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the flow based on the TicketCreatedDate for specific ticket statuses.

        :param df: Input dataframe with at least 'TicketStatus' and 'TicketCreatedDate' columns.
        :return: Dataframe with an additional 'FlowCount' column.
        """
        statuses = ["Refined", "In Progress", "To Do", "In Review"]

        required_col = "TicketStatus"
        if required_col not in df.columns:
            print(f"Required column ('{required_col}') not found in the dataframe")
            return df


        valid_tickets_per_date = (
            df[df["TicketStatus"].isin(statuses)]
            .groupby("TicketCreatedDate")
            .nunique()["TicketName"]
            .reset_index(name="FlowTicketsCount")
        )


        df = df.merge(
            valid_tickets_per_date, on="TicketCreatedDate", how="left"
        )


        df["FlowTicketsCount"] = df["FlowTicketsCount"].fillna(0).astype(int)

        return df

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the dataframe to retain only 'TicketCreatedDate', 'CumulativeDone', and 'CumulativeFlow' columns.
        Removes duplicate values based on 'TicketCreatedDate'.

        :param df: Input dataframe.
        :return: Filtered dataframe.
        """
        df_filtered = df[['TicketCreatedDate', "DoneTicketsCount", "FlowTicketsCount"]]

        df_filtered = df_filtered.drop_duplicates(subset='TicketCreatedDate')

        return df_filtered

    def fill_consecutive_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills in missing dates in the 'TicketCreatedDate' column with consecutive dates
        and forward-fills the 'CumulativeDone' and 'CumulativeFlow' columns for the missing dates.

        :param df: Input dataframe.
        :return: DataFrame with consecutive dates and forward-filled values.
        """

        df['TicketCreatedDate'] = pd.to_datetime(df['TicketCreatedDate'])


        df.set_index('TicketCreatedDate', inplace=True)
        all_dates = pd.date_range(start=df.index.min(), end=df.index.max())
        df = df.reindex(all_dates)


        df['DoneTicketsCount'] = df['DoneTicketsCount'].ffill().astype(int)
        df['FlowTicketsCount'] = df['FlowTicketsCount'].ffill().astype(int)

        return df

    @staticmethod
    def series_to_supervised(data: Union[List[float], pd.DataFrame],
                             n_in: int = 1,
                             n_out: int = 1,
                             dropnan: bool = True) -> pd.DataFrame:

        if isinstance(data, list):
            data = pd.DataFrame(data)
        elif isinstance(data, pd.Series):
            data = data.to_frame()

        cols = []
        for name, col in data.items():
            col_df = pd.DataFrame(col)
            for i in range(n_in, 0, -1):
                cols.append(col_df.shift(i).rename(columns={name: f'{name}(t-{i})'}))
            for i in range(0, n_out):
                cols.append(col_df.shift(-i).rename(columns={name: f'{name}(t+{i})'}))

        agg = pd.concat(cols, axis=1)
        if dropnan:
            agg.dropna(inplace=True)

        return agg





























