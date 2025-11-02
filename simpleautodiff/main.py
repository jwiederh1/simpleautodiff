from simpleautodiff import *

if __name__ == "__main__":
    import math
    Node.verbose = True

    # Simple test 1: y = sin(x1 * x2)
    x1 = Node(2.0)
    x2 = Node(3.0)
    y = sin(mul(x1, x2))  # v1 = x1 * x2    y = sin(v1)       # y = sin(v1)

    print(f"\nx1 = {x1.value}, x2 = {x2.value}, y = {y.value}\n")

    # Run backward
    backward(y)

    print("\nBackward Mode Results:")
    print(f"dz/dx1 = {x1.partial_derivative}")
    print(f"dz/dx2 = {x2.partial_derivative}")

    # Analytical expected
    expected_x1 = math.cos(x1.value * x2.value) * x2.value
    expected_x2 = math.cos(x1.value * x2.value) * x1.value
    print("\nExpected:")
    print(f"dz/dx1 = {expected_x1}")
    print(f"dz/dx2 = {expected_x2}")

    # Numerical gradient check (central difference)
    def eval_y():
        # rebuild graph to reflect current Node.value
        a = Node(x1.value)
        b = Node(x2.value)
        return sin(mul(a,b)).value

    eps = 1e-6
    # x1
    x1_val = x1.value
    x1.value = x1_val + eps
    f_plus = eval_y()
    x1.value = x1_val - eps
    f_minus = eval_y()
    x1.value = x1_val
    num_x1 = (f_plus - f_minus) / (2 * eps)

    # x2
    x2_val = x2.value
    x2.value = x2_val + eps
    f_plus = eval_y()
    x2.value = x2_val - eps
    f_minus = eval_y()
    x2.value = x2_val
    num_x2 = (f_plus - f_minus) / (2 * eps)

    print("\nNumerical (finite differences):")
    print(f"dz/dx1 ≈ {num_x1}")
    print(f"dz/dx2 ≈ {num_x2}")

