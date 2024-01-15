# %%
import numpy as np
import utils
from typing import List
import matplotlib.pyplot as plt


# %%
class LinearRegression:
    def __init__(self, rate=0.005, num_iters=4000):
        self.in_data = np.zeros(1)
        self.out_data = np.zeros(1)

        self.in_data_max = 1  # to undo normalizing
        self.out_data_max = 1  # to undo normalizing

        self.weight, self.intercept = utils.get_parameters()
        self.learning_rate = rate
        self.num_iters = num_iters
        self.i = 0
        self.cost_history = []

    # gradient descent algorithm:
    # we iterate through every feature and predict some value
    # get cost function value
    # cost function will be quadratic cost function
    # then we count its derivative over weight and over intercept
    # simultaniuosly update these values
    #

    def __mean_abs_error__(self) -> float:
        num = self.in_data.shape[0]
        total_cost = 0
        # #
        # err_vector = predictions - out_data
        # cost = np.multiply(err_vector, err_vector) / num
        # return cost
        for i in range(num):
            prediction = self.in_data[i] * self.weight + self.intercept
            # J = (x - y)^2
            cost = (prediction - self.out_data[i]) ** 2
            # if i == 0:
            # print(f"prediction {prediction}, cost {cost}")
            total_cost += cost
        return total_cost / num

    def __get_mean__(data) -> float:
        return np.sum(data) / data.shape[0]

    def precision(self) -> float:
        """
        Implements R-Squared Score metric to get precision of predictions in percentage.

        Formula:
        R2 = 1 - RSS / TSS, where:
        RSS - sum of squares of residuals
        TSS - total sum of squares

        RSS = sum(predicted_value - real_value)^2
        TSS = sum(real_value - mean)^2
        """

        # R^2 = 1 - RSS / TSS
        # RSS = sum of squares of residuals
        num = self.out_data.shape[0]
        mean = np.sum(self.out_data) / self.out_data.shape[0]
        print(f"num = {num} mean = {mean}")
        rss = 0
        tss = 0
        print(f"f({self.in_data[0]})= {self.weight * self.in_data[0] + self.intercept}")
        for i in range(num):
            prediction = self.weight * self.in_data[i] + self.intercept
            rss += (self.out_data[i] - prediction) ** 2
            tss += (self.out_data[i] - mean) ** 2

        return 1 - rss / tss

    def get_predictions(self, input_data) -> np.ndarray:
        m = input_data.shape[0]
        predictions = np.ndarray(m, dtype=np.float64)
        for i in range(m):
            predictions[i] = self.weight * input_data[i] + self.intercept
        return predictions

    def compute_gradient(self):
        m = self.in_data.shape[0]
        dj_dw = 0
        dj_db = 0
        for i in range(m):
            f_wb = self.weight * self.in_data[i] + self.intercept
            dj_db_i = f_wb - self.out_data[i]
            dj_dw_i = (f_wb - self.out_data[i]) * self.in_data[i]
            dj_db += dj_db_i
            dj_dw += dj_dw_i
        dj_db /= m
        dj_dw /= m
        return dj_dw, dj_db

    def gradient_descent(self):
        m = self.in_data.shape[0]

        print(
            f"Initial weight: {self.weight: 0.5}, Initial intercept: {self.intercept: 0.5}"
        )
        cost = self.__mean_abs_error__()
        print(f"Initial Cost in {self.i} iteration: {cost: 0.5}")
        print()
        while self.i < self.num_iters:
            # derivative for weights
            # E(prediction) * weight / m
            gradient_weight, gradient_intercept = self.compute_gradient()

            # derivative for intercept
            # E(prediction) / m

            self.weight -= gradient_weight * self.learning_rate
            self.intercept -= gradient_intercept * self.learning_rate

            cost = self.__mean_abs_error__()
            self.cost_history.append(cost)
            if self.i % (self.num_iters // 10) == 0:
                print(
                    f"gradient_w {gradient_weight}, gradient_int {gradient_intercept}"
                )
                print(
                    f" weight: {self.weight: 0.5}, Initial intercept: {self.intercept: 0.5}"
                )
                print(f"Cost in {self.i} iteration: {cost: 0.5}")
                print()
            self.i += 1

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Implements feature scaling using mean normalization
        """
        print(f"data.max() = {data.max()}")
        m = data.shape[0]
        max = data.max()
        for i in range(m):
            data[i] = data[i] / max

        # return (data - data.mean()) / (data.max() - data.min())

    def load_data(self, database="data.csv", input_name="km", output_name="price"):
        in_data, out_data = utils.get_database(database, input_name, output_name)
        self.in_data = np.array(in_data, dtype=np.float64) / 10000  # in thousands kms
        # self.in_data_max = self.in_data.max()
        # self.normalize_data(self.in_data)

        self.out_data = (
            np.array(out_data, dtype=np.float64) / 10000
        )  # in thousands euros
        # self.out_data_max = self.out_data.max()
        # self.normalize_data(self.out_data)

        print("in:")
        print(self.in_data)
        print(self.in_data.shape)
        print("out:")
        print(self.out_data)
        print(self.out_data.shape)

    def learn(self, database="data.csv", input_name="km", output_name="price"):
        """
        Implements linear regression model training based on the database file.
        Database name and name of input and iutput fields can be modified.
        """

        prec = self.precision()
        print(f"precision before learning: {prec}")

        self.gradient_descent()

        # self.weight *= 10000
        self.intercept *= 10000
        self.in_data *= 10000
        self.out_data *= 10000

        prec = self.precision()
        print(f"precision after learning: {prec}")

        utils.save_parameters(self.weight, self.intercept)
        # for e in data:
        #     print(e)
        # print(f"{key}: {value}")

    def predict(self, in_feature: float) -> float:
        normalized_feature = in_feature / 10000
        print()
        # normalized_feature = in_feature / self.in_data_max
        # return (normalized_feature * self.weight + self.intercept) * self.out_data_max
        return in_feature * self.weight + self.intercept

    def plot_cost(self):
        plt.plot(self.cost_history, c="r")
        plt.title("Learning curve")
        plt.xlabel("Number of iterations")
        plt.ylabel("Square error cost")
        plt.show()

    def plot_result(self):
        unscaled_in_data = self.in_data * 10000
        unscaled_out_data = self.out_data * 10000
        # unscaled_in_data = self.in_data * self.in_data_max / 1000
        # unscaled_out_data = self.out_data * self.out_data_max / 1000
        predictions = np.array([self.weight * x + self.intercept for x in self.in_data])
        # predictions = (
        #     np.array([self.weight * x + self.intercept for x in self.in_data]) * 10000
        # )

        # plt.plot(self.in_data * 10000, predictions, c="r")
        plt.plot(self.in_data, predictions, c="r")
        # plt.scatter(self.in_data * 10000, self.out_data * 10000, c="b")
        plt.scatter(self.in_data, self.out_data, c="b")

        plt.xlabel("Mileage, 1000 kms")
        plt.ylabel("Price of the car, 1000 euros")


# %%
if __name__ == "__main__":
    model = LinearRegression()
    model.load_data()
    model.learn()

# %%
# model = LinearRegression()
# model.load_data()
# model.learn()
# model.plot_cost()

#
# %%
# model.plot_cost()
#

# %%
