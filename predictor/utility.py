from pathlib import Path
import pandas as pd


class TimeSeriesPreprocessor:
    def __init__(self):
        pass

    def read_data(self, filename):

        filename = filename if isinstance(filename, Path) else Path(filename)

        return pd.read_csv(filename, dtype={"PI": str})

    def process_and_sort(self, df):
        """
        Process and sort dataframe based on the "PI" column.
        """
        df = (
            df.copy()
        )

        if "PI" in df.columns:
            df["PI"] = pd.to_numeric(df["PI"], errors="coerce")
            df["PI"] = df["PI"].round(1)
            df = df.sort_values(by="PI")
        return df

    def split_and_process(self, df):
        """
        Split dataframe based on 'TicketProject' and apply process_and_sort method on each subset.
        """
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
        if not all(col in df.columns for col in ["PI", "TicketStatus"]):
            print(
                "Required columns ('PI' or 'TicketStatus') not found in the dataframe"
            )
            return df

        done_tickets_per_pi = (
            df[df["TicketStatus"] == "Done"]
            .groupby("PI")
            .size()
            .reset_index(name="count_done")
        )

        done_tickets_per_pi["running_total_done"] = done_tickets_per_pi[
            "count_done"
        ].cumsum()

        df = df.merge(
            done_tickets_per_pi[["PI", "running_total_done"]], on="PI", how="left"
        )

        df["running_total_done"] = (
            df["running_total_done"].fillna(method="ffill").fillna(0)
        )

        return df
