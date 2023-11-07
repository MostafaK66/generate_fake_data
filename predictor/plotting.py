import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataPlotter:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(os.getcwd(), "plotting_output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_projects(self, ada_projects, last_n_days=None):
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

            ax_done.plot(
                project.index,
                project["DoneTicketsCount"],
                label="DoneTicketsCount",
                color="blue",
            )
            ax_flow.plot(
                project.index,
                project["FlowTicketsCount"],
                label="FlowTicketsCount",
                color="orange",
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
