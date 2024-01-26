from linear_regression import LinearRegression
import config
import sys


def main():
    args = ["-d", "--plot-debug"]
    debug = 0
    if len(sys.argv) >= 2:
        if len(sys.argv) > 2 or sys.argv[1] not in args:
            print(len(sys.argv), sys.argv)
            print(
                "Invalid arguments. Usage: python ./learn.py [-d | --plot-debug]",
                file=sys.stderr,
            )
            exit(1)
        else:
            print("Starting the model in debug mode...")
            debug = 1

    model = LinearRegression(debug=debug)
    model.load_data(config.DB_NAME, config.INPUT_PARAMETER, config.OUTPUT_PARAMETER)
    model.learn()
    return


if __name__ == "__main__":
    main()
