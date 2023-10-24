import os
import pandas as pd
import time
import logging


script_dir = os.path.dirname(os.path.realpath(__file__))
PATH_FOR_OUTPUT = os.path.join(script_dir, "ada_output")
os.makedirs(PATH_FOR_OUTPUT, exist_ok=True)
logging.basicConfig(level=logging.INFO)


class DecoratorTools:
    @staticmethod
    def timer_decorator(func):
        """
        Decorator function to measure the execution time of a function.

        Args:
            func (function): The function whose execution time is to be measured.

        Returns:
            wrapper (function): The new function that includes timing functionality.
        """

        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            elapsed_time_seconds = end_time - start_time
            minutes = int(elapsed_time_seconds // 60)
            seconds = elapsed_time_seconds % 60

            logging.info(
                f"Function {func.__name__} took {minutes} minutes and {seconds:.2f} seconds to execute."
            )
            return result

        return wrapper

    @staticmethod
    def save_to_csv_decorator(func):
        """
        A staticmethod decorator that saves the result of the decorated function to CSV files.

        This decorator is designed to work with functions that return either a single DataFrame
        or a tuple of DataFrames. The output is saved in a specified directory with a filename
        based on the function's name and the type of output (either 'output' or 'pi').

        Parameters:
        - func (Callable): The function to be decorated.

        Returns:
        - Callable: The wrapped function that saves its results to CSV before returning.

        Note:
        - It assumes the existence of a global variable `PATH_FOR_OUTPUT` for the output directory path.
        - If the function returns a tuple of DataFrames, their names should be added in the `output_names` list.
        """

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            os.makedirs(PATH_FOR_OUTPUT, exist_ok=True)
            output_names = ["output", "pi"]  # Add output names here
            if isinstance(result, tuple):
                for name, df in zip(output_names, result):
                    if isinstance(df, pd.DataFrame):
                        df.to_csv(
                            os.path.join(
                                PATH_FOR_OUTPUT,
                                f"{func.__name__}_{name}.csv",
                            ),
                            index=False,
                        )
            else:
                if isinstance(result, pd.DataFrame):
                    result.to_csv(
                        os.path.join(
                            PATH_FOR_OUTPUT, f"{func.__name__}_output.csv"
                        ),
                        index=False,
                    )
            return result

        return wrapper

