from simpleautodiff import *
import math

def build_function(func_name, x1, x2):
    """
    Build a computational graph based on the chosen function.
    Returns the output node y.
    """
    if func_name == "sin(x1 * x2)":
        return sin(mul(x1, x2))
    elif func_name == "log(x1) + x1*x2 - sin(x2)":
        v1 = log(x1)
        v2 = mul(x1, x2)
        v3 = sin(x2)
        v4 = add(v1, v2)
        return sub(v4, v3)
    elif func_name == "x1 + x2":
        return add(x1, x2)
    elif func_name == "x1 * x2 + log(x1)":
        return add(mul(x1, x2), log(x1))
    else:
        raise ValueError(f"Unknown function '{func_name}'")


def main():
    print("\n=== Automatic Differentiation Demo ===\n")

    # User chooses mode
    mode = input("Choose differentiation mode (forward/backward): ").strip().lower()
    if mode not in ["forward", "backward"]:
        print("Invalid mode. Please choose 'forward' or 'backward'.")
        return

    # Choose function
    print("\nAvailable functions:")
    functions = [
        "sin(x1 * x2)",
        "log(x1) + x1*x2 - sin(x2)",
        "x1 + x2",
        "x1 * x2 + log(x1)"
    ]
    for i, f in enumerate(functions):
        print(f"{i+1}. {f}")

    try:
        func_idx = int(input("\nSelect a function (1-4): ")) - 1
        func_name = functions[func_idx]
    except (IndexError, ValueError):
        print("Invalid selection.")
        return

    # Get input values
    try:
        x1_val = float(input("Enter value for x1: "))
        x2_val = float(input("Enter value for x2: "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Reset counters
    Node.input_count = 0
    Node.intermediate_count = 0

    # Build function graph
    x1 = Node(x1_val)
    x2 = Node(x2_val)
    y = build_function(func_name, x1, x2)

    print(f"\nSelected function: y = {func_name}")
    print(f"x1 = {x1.value}, x2 = {x2.value}, y = {y.value}")

    # === Print Topological Ordering ===
    if mode == "forward":
        ordering = topological_order(y)
        print("\nTopological Order (Forward Mode):")
    else:
        ordering = reverse_topological_order(y)
        print("\nReverse Topological Order (Backward Mode):")

    print(" -> ".join([node.name for node in ordering]))

    # === Compute derivatives ===
    if mode == "backward":
        print("\n=== BACKWARD MODE (Reverse) ===")
        backward(y)
        print(f"∂y/∂x1 = {x1.partial_derivative}")
        print(f"∂y/∂x2 = {x2.partial_derivative}")

    elif mode == "forward":
        print("\n=== FORWARD MODE ===")
        # Run forward mode for x1 (∂y/∂x1)
        forward(x1)
        print(f"∂y/∂x1 = {y.partial_derivative}")

        # Rebuild for x2 (since forward mode uses one seed at a time)
        Node.input_count = 0
        Node.intermediate_count = 0
        x1_f = Node(x1_val)
        x2_f = Node(x2_val)
        y2 = build_function(func_name, x1_f, x2_f)
        forward(x2_f)
        print(f"∂y/∂x2 = {y2.partial_derivative}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
