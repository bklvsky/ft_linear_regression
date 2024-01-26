import csv
import sys
import pickle
from typing import Tuple, List


def get_parameters() -> Tuple[float, float]:
    """
    Fetches parameters saved with pickle module.
    """
    try:
        with open("regression_parameters.pkl", "rb") as f:
            weight, intercept = pickle.load(f)
            # print("parameters from pickle:")
            # print(f"weight: {weight}, intercept: {intercept}")
            return weight, intercept
    except OSError:
        return 0., 0.
    except EOFError:
        return 0., 0.
    except Exception as e:
        print(f"Pickle exception: {type(e)} : {e}")
        exit(1)


def save_parameters(weight, intercept):
    """
    Saves model parameters to a file "regression_parameters.pkl" for further reuse.
    """
    try:
        with open("regression_parameters.pkl", "wb") as f:
            pickle.dump((weight, intercept), f)
    except Exception as e:
        print(e)
        exit(1)


def db_error_exit(err: str):
    """
    Prints formatted database error message and exits with code 1.
    """
    print(
        f"Database is corrupted. {err}",
        file=sys.stderr,
    )
    exit(1)


def get_database(filename, input_name, output_name) -> Tuple[List[float], List[float]]:
    """
    Reads csv database with two columns and parses it into a list of training examples.
    Performs checks on validity of the db.

    Returns:
    List[List[input_parameter: float, output_parameter: float]].
    """
    try:
        with open(filename, "r") as f:
            reader = csv.reader(f)
            # if len(reader) == 0:
            #     db_error_exit("Empty database.")
            headers = next(reader, [])
            if (
                len(headers) != 2
                or headers[0] != input_name
                or headers[1] != output_name
            ):
                db_error_exit(f"Expected [{input_name}, {output_name}] fields.")
            try:
                data = [[float(x) for x in values] for values in reader]
            except ValueError:
                db_error_exit(f"Non-numerical value at line {reader.line_num}")
            input_data = [row[0] for row in data]
            output_data = [row[1] for row in data]
            return input_data, output_data
    except OSError as e:
        print(f"{filename} can't be read. Error: {e}", file=sys.stderr)
        exit(1)

