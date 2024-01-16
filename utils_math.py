def scale_parameters(*args, scale=10000):
    new_args = [x / scale for x in args]
    return tuple(new_args)


def unscale_parameters(*args, scale=10000):
    new_args = [x * scale for x in args]

    return tuple(new_args)


def estimate(x, weight=0.0, intercept=0.0):
    return weight * x + intercept


def mean_square_error(
    in_data, out_data, m: int, weight: float, intercept: float
) -> float:
    total_cost = 0
    for i in range(m):
        prediction = in_data[i] * weight + intercept
        cost = (prediction - out_data[i]) ** 2
        total_cost += cost
    return total_cost / m
