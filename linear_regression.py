import numpy as np
import utils
import matplotlib.pyplot as plt
import utils_math
import config


class LinearRegression:
    def __init__(self, rate=0.005, num_iters=4000, debug=0):
        self.in_data = np.zeros(1)
        self.out_data = np.zeros(1)

        self.weight, self.intercept = utils.get_parameters()
        self.learning_rate = rate
        self.num_iters = num_iters

        self.debug = debug
        self.i = 0
        self.cost_history = np.zeros(num_iters)

    # gradient descent algorithm:
    # we iterate through every feature and predict some value
    # get cost function value
    # cost function will be quadratic cost function
    # then we count its derivative over weight and over intercept
    # simultaniuosly update these values
    #

    def __mean_square_error__(self) -> float:
        num = self.in_data.shape[0]
        total_cost = 0
        for i in range(num):
            prediction = self.in_data[i] * self.weight + self.intercept
            # J = (x - y)^2
            cost = (prediction - self.out_data[i]) ** 2
            # if i == 0:
            # print(f"prediction {prediction}, cost {cost}")
            total_cost += cost
        return total_cost / num

    def get_predictions(self) -> np.ndarray:
        predictions = np.zeros(self.in_data.shape[0])
        for i in range(predictions.shape[0]):
            predictions[i] = utils_math.estimate(
                self.in_data[i], self.weight, self.intercept
            )
        return predictions

    def compute_gradient(self):
        """
        Computes gradients for weight and intercept of the linear regression.
        Formulas used: derivatives for square error cost function.

        Cost:
        J = 1/2m * sum((y_pred_i - y_i)^2) = 1/2m * sum((w * x_i + b - y_i)^2)

        Gradient of weight - derivative of J with respect to weight:
        dw/dJ = 1/m * sum(w * x_i + b - y) * weight

        Gradient of intercept - derivative of J with respect to intercept:
        db/dJ = 1/m * sum(w * x_i + b - y)
        """

        predictions = self.get_predictions()
        m = self.in_data.shape[0]

        # starting points to aggregate gradients
        grad_weight = 0
        grad_intercept = 0

        for i in range(m):
            grad_intercept_i = predictions[i] - self.out_data[i]
            grad_weight_i = (predictions[i] - self.out_data[i]) * self.in_data[i]
            grad_intercept += grad_intercept_i
            grad_weight += grad_weight_i

        grad_intercept /= m
        grad_weight /= m
        return grad_weight, grad_intercept

    def gradient_descent(self):
        while self.i < self.num_iters:
            gradient_weight, gradient_intercept = self.compute_gradient()

            self.weight -= gradient_weight * self.learning_rate
            self.intercept -= gradient_intercept * self.learning_rate

            if self.debug:
                cost = utils_math.mean_square_error(
                    self.in_data,
                    self.out_data,
                    self.out_data.shape[0],
                    self.weight,
                    self.intercept,
                )
                self.cost_history[self.i] = cost
            self.i += 1


    def load_data(self, database, input_name, output_name):
        """
        Loads 2 parameters csv database into model's in_data and out_data fields.
        """
        in_data, out_data = utils.get_database(database, input_name, output_name)
        self.in_data = np.array(in_data, dtype=np.float64)

        self.out_data = np.array(out_data, dtype=np.float64)

    def learn(self):
        """
        Implements linear regression model training based on the database file.
        Database name and name of input and iutput fields can be modified.
        """

        self.in_data, self.out_data, self.intercept = utils_math.scale_parameters(
            self.in_data, self.out_data, self.intercept
        )

        self.gradient_descent()

        (
            self.intercept,
            self.in_data,
            self.out_data,
            self.cost_history,
        ) = utils_math.unscale_parameters(
            self.intercept, self.in_data, self.out_data, self.cost_history
        )

        if self.debug:
            self.plot_cost()

        utils.save_parameters(self.weight, self.intercept)

    def estimatePrice(self, in_feature: float) -> float:
        return in_feature * self.weight + self.intercept

    def plot_cost(self):
        plt.plot(self.cost_history, c="r")
        plt.title("Learning curve")
        plt.xlabel("Number of iterations")
        plt.ylabel("Square error cost")
        plt.show()
