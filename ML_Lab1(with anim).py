import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Градиентный спуск
def gradient_descent(gradient_func, initial_point, learning_rate, iterations):
    current_point = initial_point
    trajectory = [current_point]

    for i in range(iterations):
        gradient = gradient_func(current_point)
        current_point = current_point - learning_rate * gradient
        trajectory.append(current_point)

    return np.array(trajectory)

# Функции и их градиенты
def quadratic_function(x):
    return x ** 2

def quadratic_gradient(x):
    return x * 2

def test_gradient_descent(gradient, initial_point, learning_rate, iterations):
    trajectory = gradient_descent(gradient, initial_point, learning_rate, iterations)
    optimal_solution = trajectory[-1]
    error = np.linalg.norm(optimal_solution - np.array([1, 1]))
    return trajectory, optimal_solution, error

# Визуализация на двумерном графике с анимацией точки найденного решения
def visualize_2d_animation(trajectory, function, x_range):
    fig, ax = plt.subplots()
    x = np.linspace(x_range[0], x_range[1], 100)
    y = function(x)
    ax.plot(x, y, label='Function')
    line, = ax.plot([], [], 'ro-', label='Trajectory')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Optimization Function and Gradient Descent Trajectory')
    ax.legend()
    ax.grid(True)

    def update(frame):
        line.set_data(trajectory[:frame], function(np.array(trajectory[:frame])))
        return line,

    anim = FuncAnimation(fig, update, frames=len(trajectory), interval=100, blit=True)
    plt.show()

# Тестирование квадратичной функции
initial_point_quad = 10
learning_rate_quad = 0.1
iterations_quad = 100

trajectory_quad, optimal_solution_quad, error_quad = test_gradient_descent(quadratic_gradient,
                                                                           initial_point_quad,
                                                                           learning_rate_quad,
                                                                           iterations_quad)

visualize_2d_animation(trajectory_quad, quadratic_function, [-12, 12])
print(f"Optimal solution for quadratic function: {optimal_solution_quad}")
print(f"Error for quadratic function: {error_quad}")

