from sympy import Symbol, solve

def loss_function(a, b, c, d, x):
    """Calculates the mean squared error loss for a cubic equation."""
    return (a*x**3 + b*x**2 + c*x + d) ** 2

def solve_cubic_sgd(a, b, c, d, learning_rate=0.01, num_iterations=1000):
    """
    Attempts to solve a cubic equation using SGD.

    Args:
        a: The coefficient of the x^3 term (int or float).
        b: The coefficient of the x^2 term (int or float).
        c: The coefficient of the x term (int or float).
        d: The constant term (int or float).
        learning_rate: The learning rate for SGD (default: 0.01).
        num_iterations: The number of iterations for SGD (default: 1000).

    Returns:
        The final guess for x (may not be the exact solution).
    """
    x = 0.0  # Initial guess
    for i in range(num_iterations):
        gradient = 3 * a * x**2 + 2 * b * x + c
        x -= learning_rate * gradient
    return x

def solve_cubic_sympy(a, b, c, d):
    """
    Solves a cubic equation using SymPy library.

    Args:
        a: The coefficient of the x^3 term (int or float).
        b: The coefficient of the x^2 term (int or float).
        c: The coefficient of the x term (int or float).
        d: The constant term (int or float).

    Returns:
        A list of symbolic solutions for x.
    """
    x = Symbol('x')
    equation = a*x**3 + b*x**2 + c*x + d
    solutions = solve(equation, x)
    return solutions

# Example usage (SGD approach might not find all solutions)
a = 1
b = -1
c = 2
d = -1

solution_sgd = solve_cubic_sgd(a, b, c, d)
print("SGD Solution (might not be exact):", solution_sgd)

# Example usage (SymPy approach)
solutions_sympy = solve_cubic_sympy(a, b, c, d)
print("SymPy Solutions:", solutions_sympy)
