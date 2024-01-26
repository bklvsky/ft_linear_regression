import pickle
import sys
import utils
import utils_math
import config


def user_input_error_exit(in_str: str, err: str = ""):
    print(f"[{in_str}] is not a valid mileage for a car. " + err, file=sys.stderr)
    exit(1)


def main():
    """
    When you launch the program, it should prompt you for a mileage, and then give
    you back the estimated price for that mileage. The program will use the following
    hypothesis to predict the price :
    estimatePrice(mileage) = θ0 + (θ1 ∗ mileage)
    Before the run of the training program, theta0 and theta1 will be set to 0.
    """
    in_str = input("Enter the mileage of the car: ")
    try:
        in_value = float(in_str)
        if in_value < 0:
            user_input_error_exit(in_str, "Can't be less than 0.")
        if in_value > config.MAX_X:
            user_input_error_exit(in_str, "Too big input value for this model. Can't make an accurate prediction.")
    except ValueError:
        user_input_error_exit(in_str)
    weight, intercept = utils.get_parameters()
    prediction = utils_math.estimate(in_value, weight=weight, intercept=intercept)
    print(f"Estimated price for the mileage of {in_value} km = {prediction:.7}")


if __name__ == "__main__":
    main()
