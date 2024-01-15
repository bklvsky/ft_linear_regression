# When you launch the program, it should prompt you for a mileage, and then give
# you back the estimated price for that mileage. The program will use the following
# hypothesis to predict the price :
# estimatePrice(mileage) = θ0 + (θ1 ∗ mileage)
# Before the run of the training program, theta0 and theta1 will be set to 0.

import pickle
import sys
from linear_regression import LinearRegression

# from linear_regression import model


# def get_parameters() -> Tuple[float, float]:
#     try:
#         with open("regression_parameters.pkl", "rb") as f:
#             return pickle.load(f)
#     except FileNotFoundError:
#         return 0, 0


def user_input_error_exit(in_str: str, err: str = ""):
    print(f"[{in_str}] is not a valid mileage for a car." + err, file=sys.stderr)
    exit(1)


def main():
    in_str = input("Enter the mileage of the car: ")
    try:
        in_value = float(in_str)
        if in_value < 0:
            user_input_error_exit(in_str, "Can't be less than 0.")
    except ValueError:
        user_input_error_exit(in_str)
    model = LinearRegression()
    print(
        f"Estimated price for the mileage of {in_value} km = {model.predict(in_value)}"
    )


if __name__ == "__main__":
    main()
