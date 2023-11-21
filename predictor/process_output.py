import os
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class OutPutProcessor:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(os.getcwd(), "plotting_output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_projects(self, ada_projects, split_ratio, last_n_days=None):
        fig, axs = plt.subplots(len(ada_projects) * 2, 1, figsize=(10, 15), sharex=True)

        for idx, project in enumerate(ada_projects):
            ax_done = axs[idx * 2]
            ax_flow = axs[idx * 2 + 1]

            if last_n_days:
                end_date = pd.Timestamp.now()
                start_date = end_date - pd.Timedelta(days=last_n_days)
                project = project[
                    (project.index >= start_date) & (project.index <= end_date)
                ]

            split_idx = int(len(project) * split_ratio)

            ax_done.plot(
                project.index[:split_idx],
                project["DoneTicketsCount"][:split_idx],
                label="DoneTicketsCount (Train)",
                color="blue",
            )
            ax_done.plot(
                project.index[split_idx:],
                project["DoneTicketsCount"][split_idx:],
                label="DoneTicketsCount (Test)",
                color="green",
            )

            ax_flow.plot(
                project.index[:split_idx],
                project["FlowTicketsCount"][:split_idx],
                label="FlowTicketsCount (Train)",
                color="orange",
            )
            ax_flow.plot(
                project.index[split_idx:],
                project["FlowTicketsCount"][split_idx:],
                label="FlowTicketsCount (Test)",
                color="red",
            )

            ax_done.set_title(f"ada_project_{idx + 1} Done Tickets")
            ax_flow.set_title(f"ada_project_{idx + 1} Flow Tickets")

            ax_done.set_ylabel("Done Tickets Count")
            ax_flow.set_ylabel("Flow Tickets Count")

            ax_done.legend()
            ax_flow.legend()

            ax_done.tick_params(axis="x", rotation=45)
            ax_flow.tick_params(axis="x", rotation=45)

        axs[-1].set_xlabel("Date")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "projects_plot.png"))
        plt.close(fig)

    def plot_actual_vs_predicted(self, results_df, key):
        plt.figure(figsize=(10, 5))
        plt.plot(
            results_df.index,
            results_df["Actual"],
            label="Actual (Test)",
            color="blue",
            marker="o",
        )
        plt.plot(
            results_df.index,
            results_df["Predicted"],
            label="Predicted (Test)",
            color="red",
            linestyle="--",
            marker="x",
        )
        plt.title(f"Actual vs Predicted - {key}")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Ticket Count")
        plt.grid(True)
        plt.xticks(rotation=15)
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, f"{key}_actual_vs_predicted.png")
        plt.savefig(plot_path)
        plt.close()

    def plot_cumulative_actual_vs_predicted(self, results_df, key):
        plt.figure(figsize=(10, 5))
        plt.plot(
            results_df.index,
            results_df["Cumulative_Actual"],
            label="Cumulative Actual (Test)",
            color="blue",
            marker="o",
        )
        plt.plot(
            results_df.index,
            results_df["Cumulative_Predicted"],
            label="Cumulative Predicted (Test)",
            color="red",
            linestyle="--",
            marker="x",
        )
        plt.title(f"Cumulative Actual vs Predicted - {key}")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Cumulative Ticket Count")
        plt.grid(True)
        plt.xticks(rotation=15)
        plt.tight_layout()

        plot_path = os.path.join(
            self.output_dir, f"{key}_cumulative_actual_vs_cumulative_predicted.png"
        )
        plt.savefig(plot_path)
        plt.close()

    def process_and_plot(
        self,
        ada_projects,
        walk_forward_validation_results,
        split_ratio,
        n_in,
        last_n_days=None,
    ):
        output_dir = os.path.join(os.getcwd(), "prediction_output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        all_results = []
        for key, value in walk_forward_validation_results.items():
            results_df = value["ResultsDF"].copy()

            project_number = int(key.split("_")[-1].replace("df", "")) - 1
            project_df = ada_projects[project_number]

            split_idx = int(len(project_df) * split_ratio)

            if last_n_days is not None:
                end_date = (
                    pd.Timestamp.now() if last_n_days is None else project_df.index[-1]
                )
                start_date = end_date - pd.Timedelta(days=last_n_days)
                project_df = project_df[
                    (project_df.index >= start_date) & (project_df.index <= end_date)
                ]

            test_timestamps = project_df.index[split_idx + n_in :]

            min_length = min(len(test_timestamps), len(results_df))
            results_df = results_df.iloc[:min_length]
            test_timestamps = test_timestamps[:min_length]

            results_df.set_index(test_timestamps, inplace=True)

            self.plot_actual_vs_predicted(results_df, key)

            results_df["Cumulative_Actual"] = results_df["Actual"].cumsum()
            results_df["Cumulative_Predicted"] = results_df["Predicted"].cumsum()

            self.plot_cumulative_actual_vs_predicted(results_df, key)
            results_df["PredictionInfo"] = key
            all_results.append(results_df)
        combined_results_df = pd.concat(all_results)
        csv_file_path = os.path.join(output_dir, "combined_results.csv")
        combined_results_df.to_csv(csv_file_path)

        return combined_results_df

    def add_type_column(self, df, prediction_info_column="PredictionInfo"):
        df[prediction_info_column] = df[prediction_info_column].astype(str)

        df["Type"] = df[prediction_info_column]

        keywords = ["Done", "Flow"]

        df["Type"] = df["Type"].apply(
            lambda x: next((word for word in keywords if word in x), None)
        )

        output_dir = os.path.join(os.getcwd(), "prediction_output")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "categorized_predictions.csv")
        df.to_csv(output_path, index=False)

        print(f"File saved to {output_path}")

        return df
