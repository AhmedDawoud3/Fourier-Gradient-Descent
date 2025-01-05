import math
import pickle as pkl

import matplotlib.pyplot as plt
import torch

L = 5  # Width of function approximation


def function(x):
    # return x
    # return torch.sin(x)
    # return square_wave(x)
    return torch.exp(-(x**2))


FILE_NAME = "sqr.pkl"
GRAD_TOLERANCE = 1e-5
LEARNING_RATE = 0.01
MAX_ITERATIONS = 10_000
NUM_TERMS = 20


def main():
    coefficients = initialize_coefficients(NUM_TERMS)
    print("Initial Coefficients", coefficients)
    coefficients, history = train_model(coefficients, LEARNING_RATE, MAX_ITERATIONS)
    dumb_history(history, FILE_NAME)
    print(format_function(coefficients))
    plot_results(function, coefficients)


def initialize_coefficients(num_terms):
    coefficients = torch.randn((num_terms, 2), requires_grad=True)
    with torch.no_grad():
        coefficients *= 0.01
        coefficients[0, 0] = 0  # sin(0) = 0, grad will always be 0
    return coefficients


def model(x, coefficients):
    x = x.view(-1, 1)
    terms = torch.arange(coefficients.shape[0], dtype=torch.float32) * math.pi * x / L
    sin_terms = torch.sin(terms)
    cos_terms = torch.cos(terms)
    out = torch.matmul(sin_terms, coefficients[:, 0]) + torch.matmul(
        cos_terms, coefficients[:, 1]
    )
    return out


def square_wave(x):
    return torch.sign(torch.sin(x * math.pi / L))


def error(x, coefficients):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    e = ((model(x, coefficients) - function(x)) ** 2).sum() / len(x)
    return e


def train_model(coefficients, learning_rate, max_iterations):
    history = []
    for i in range(max_iterations):
        if (
            i % int(max_iterations * 0.01) == 0 and i < max_iterations * 0.1
        ) or i % int(max_iterations * 0.1) == 0:
            print(f"Iteration {i}")
        e = error(torch.randn(10) * L / 2, coefficients)
        e.backward()
        if torch.allclose(coefficients.grad, torch.tensor(0.0), atol=GRAD_TOLERANCE):  # type: ignore
            break
        with torch.no_grad():
            lrt = (
                learning_rate if i < (max_iterations * 0.75) else learning_rate / 10
            )  # Learning rate decay
            coefficients -= lrt * coefficients.grad
            coefficients[coefficients.abs() < 1e-5] = 0
            history.append(coefficients.detach().clone().numpy())
        # print(
        #     f"Coefficients: {coefficients.detach().numpy().flatten()}, Gradient: {coefficients.grad.numpy().flatten()}, Error: {e.detach().numpy()}"
        # )
        coefficients.grad = None  # Zero Grad
    print("-" * 50)
    print(f"Final Error: {e.detach().numpy()}")
    return coefficients, history


# Save training history
def dumb_history(history, filename="history.pkl"):
    with open(filename, "wb") as f:
        pkl.dump(history, f)


# Generate human-readable function
def format_function(coefficients):
    terms = []
    for i, coefficient in enumerate(coefficients):
        coefficient = coefficient.detach().numpy().round(decimals=4)
        if coefficient[0] != 0:
            terms.append(f"{coefficient[0]:.4}sin({i}πx)")
        if coefficient[1] != 0:
            terms.append(f"{coefficient[1]:.4}cos({i}πx)")
    return "f(x) = " + " + ".join(terms)


def plot_results(func, coefficients):
    x = torch.linspace(-math.pi * 2, math.pi * 2, 1000)
    y = model(x, coefficients)
    plt.plot(x, y.detach().numpy(), label="Model")
    plt.plot(x, func(x).detach().numpy(), label="Target Function")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
