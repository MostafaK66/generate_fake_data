from typing import Tuple

import pandas as pd

from mocked_up.ada_decorator_tools import DecoratorTools
from mocked_up.ada_df_generator import AdaBottleneckGenerator


@DecoratorTools.timer_decorator
@DecoratorTools.save_to_csv_decorator
def ada_df_generator() -> Tuple[pd.DataFrame, pd.DataFrame]:
    generator = AdaBottleneckGenerator(
        n_tickets=1000,
        seed=123,
        progress_rates={
            "ADA_Project_1": 1.0,
            "ADA_Project_2": 0.8,
            "ADA_Project_3": 0.7,
        },
        days_choices={
            "ADA_Project_1": {
                "In Review": list(range(1, 20, 1)),
                "Default": list(range(1, 3, 1)),
            },
            "ADA_Project_2": {
                "In Review": list(range(1, 10, 3)),
                "Default": list(range(1, 20, 4)),
            },
            "ADA_Project_3": {
                "In Review": list(range(1, 60, 10)),
                "Default": list(range(1, 35, 7)),
            },
        },
        project_capacities={
            "ADA_Project_1": int(90 * 0.95),
            "ADA_Project_2": int(90 * 0.8),
            "ADA_Project_3": int(90 * 0.5),
        },
        team_members_count={
            "ADA_Team_1": 5,
            "ADA_Team_2": 9,
            "ADA_Team_3": 6,
            "ADA_Team_4": 10,
            "ADA_Team_5": 4,
            "ADA_Team_6": 11,
        },
    )

    df = generator.get_dataframe()
    df = generator.assign_ticket_type(df=df)
    df = generator.assign_ticket_priority(df=df)
    df_pis = generator.generate_df_pis(df=df)
    df = generator.add_ticket_scope(df=df)
    df = generator.add_team_members_column(df=df)
    df = generator.add_story_points(df=df)
    return df, df_pis


if __name__ == "__main__":
    df, df_pis = ada_df_generator()
