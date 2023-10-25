from pathlib import Path
import pandas as pd


class TimeSeriesPreprocessor:
    def __init__(self):
        pass

    def read_data(self, filename):

        filename = filename if isinstance(filename, Path) else Path(filename)

        return pd.read_csv(filename, dtype={"PI": str})

    def process_and_sort(self, df):

        df = df.copy()

        if "PI" in df.columns:
            df["PI"] = pd.to_numeric(df["PI"], errors="coerce")
            df["PI"] = df["PI"].round(1)
            df = df.sort_values(by="PI")
        return df

    def split_and_process(self, df):

        ada_project_1 = self.process_and_sort(
            df[df["TicketProject"] == "ADA_Project_1"]
        )
        ada_project_2 = self.process_and_sort(
            df[df["TicketProject"] == "ADA_Project_2"]
        )
        ada_project_3 = self.process_and_sort(
            df[df["TicketProject"] == "ADA_Project_3"]
        )

        return ada_project_1, ada_project_2, ada_project_3

    def cumulative_done_per_pi(self, df):
        required_cols = ["PI", "TicketStatus"]
        if not all(col in df.columns for col in required_cols):
            print("Required columns ('PI' or 'TicketStatus') not found in the dataframe")
            return df

        done_tickets_per_pi = (
            df[df["TicketStatus"] == "Done"]
            .groupby("PI")
            .size()
            .reset_index(name="count_done")
        )

        done_tickets_per_pi["CumulativeDone"] = done_tickets_per_pi["count_done"].cumsum()

        df = df.merge(
            done_tickets_per_pi[["PI", "CumulativeDone"]], on="PI", how="left"
        )

        df["CumulativeDone"] = df["CumulativeDone"].fillna(method="ffill").fillna(0)

        return df

    def cumulative_flow_per_pi(self, df):

        statuses = ["Refined", "In Progress", "To Do", "In Review"]


        if not all(col in df.columns for col in ["PI", "TicketStatus"]):
            print("Required columns ('PI' or 'TicketStatus') not found in the dataframe")
            return df


        valid_tickets_per_pi = (
            df[df["TicketStatus"].isin(statuses)]
            .groupby("PI")
            .nunique()["TicketName"]
            .reset_index(name="count_valid_tickets")
        )


        valid_tickets_per_pi["CumulativeFlow"] = valid_tickets_per_pi["count_valid_tickets"].cumsum()


        df = df.merge(
            valid_tickets_per_pi[["PI", "CumulativeFlow"]], on="PI", how="left"
        )

        df["CumulativeFlow"] = df["CumulativeFlow"].fillna(method="ffill").fillna(0).astype(int)

        return df



