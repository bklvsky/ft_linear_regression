import utils
import utils_math
import numpy as np
import sys
import matplotlib.pyplot as plt
import config


def r2score(out_data, predictions, w, b) -> float:
    """
    Implements R-Squared Score metric to get precision of predictions in percentage.

    Formula:
    R2 = 1 - RSS / TSS, where:
    RSS - sum of squares of residuals
    TSS - total sum of squares

    RSS = sum(real_value - predicted_value)^2
    TSS = sum(real_value - mean)^2
    """

    num = out_data.shape[0]
    mean = np.sum(out_data) / num
    rss = 0
    tss = 0
    for i in range(num):
        rss += (out_data[i] - predictions[i]) ** 2
        tss += (out_data[i] - mean) ** 2

    r2score = 1 - rss / tss
    return r2score if r2score > 0. else 0.


def plot(predictions, in_data, out_data, plot_result=0):
    """
    Plots initial data to a matplotlib graph.
    Optionally plots estimation graph if optional argument [plot_result] is not 0.
    """

    plt.scatter(in_data, out_data, c="b")
    plt.xlabel("Mileage, kms")
    plt.ylabel("Price of the car, euros")
    if plot_result:
        plt.plot(in_data, predictions, c="r")
    plt.show()


def evaluate_model(plot_result=0, plot_data=0):
    """
    Evaluated precision of a model using R2 Score metric.
    Optionally visualizes data and graph of predictions.
    """
    w, b = utils.get_parameters()
    input_data, output_data = map(
        np.array,
        utils.get_database(
            config.DB_NAME, config.INPUT_PARAMETER, config.OUTPUT_PARAMETER
        ),
    )

    estimatedPrice = np.array(
        [utils_math.estimate(x, weight=w, intercept=b) for x in input_data]
    )
    precision = r2score(output_data, estimatedPrice, w, b)
    print(f"Estimated precision of the model: {100 * precision:.5}%.")

    if plot_result or plot_data:
        plot(estimatedPrice, input_data, output_data, plot_result)


def main():
    args = ["-r", "--plot-result", "-d", "--plot-data"]
    plot_result, plot_data = 0, 0

    if sys.argv:
        if not all(x in args for x in sys.argv[1:]):
            print(f"Invalid arguments. Valid arguments: [{args}]", file=sys.stderr)
            exit(1)
        if "-r" in sys.argv or "--plot-result" in sys.argv:
            plot_result = 1
        elif "-d" in sys.argv or "--plot-data" in sys.argv:
            plot_data = 1

    evaluate_model(plot_result=plot_result, plot_data=plot_data)


if __name__ == "__main__":
    main()
