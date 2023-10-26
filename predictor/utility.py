from pathlib import Path
import pandas as pd


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

    def cumulative_done_per_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the cumulative number of 'Done' tickets based on the TicketCreatedDate.

        :param df: Input dataframe with at least 'TicketStatus' and 'TicketCreatedDate' columns.
        :return: Dataframe with an additional 'CumulativeDone' column.
        """
        required_col = "TicketStatus"
        if required_col not in df.columns:
            print(f"Required column ('{required_col}') not found in the dataframe")
            return df

        done_tickets_per_date = (
            df[df["TicketStatus"] == "Done"]
            .groupby("TicketCreatedDate")
            .size()
            .reset_index(name="count_done")
        )

        done_tickets_per_date["CumulativeDone"] = done_tickets_per_date["count_done"].cumsum()

        df = df.merge(
            done_tickets_per_date[["TicketCreatedDate", "CumulativeDone"]], on="TicketCreatedDate", how="left"
        )

        df["CumulativeDone"] = df["CumulativeDone"].ffill().fillna(0).astype(int)

        return df

    def cumulative_flow_per_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the cumulative flow based on the TicketCreatedDate for specific ticket statuses.

        :param df: Input dataframe with at least 'TicketStatus' and 'TicketCreatedDate' columns.
        :return: Dataframe with an additional 'CumulativeFlow' column.
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
            .reset_index(name="count_valid_tickets")
        )

        valid_tickets_per_date["CumulativeFlow"] = valid_tickets_per_date["count_valid_tickets"].cumsum()

        df = df.merge(
            valid_tickets_per_date[["TicketCreatedDate", "CumulativeFlow"]], on="TicketCreatedDate", how="left"
        )

        df["CumulativeFlow"] = df["CumulativeFlow"].ffill().fillna(0).astype(int)

        return df

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the dataframe to retain only 'TicketCreatedDate', 'CumulativeDone', and 'CumulativeFlow' columns.
        Removes duplicate values based on 'TicketCreatedDate'.

        :param df: Input dataframe.
        :return: Filtered dataframe.
        """
        df_filtered = df[['TicketCreatedDate', 'CumulativeDone', 'CumulativeFlow']]

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


        df['CumulativeDone'] = df['CumulativeDone'].ffill().astype(int)
        df['CumulativeFlow'] = df['CumulativeFlow'].ffill().astype(int)

        return df




























