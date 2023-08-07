import random
import math
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def bezier_curve(t, A, B, C):
    return (
        (1 - t) ** 2 * A[0] + 2 * (1 - t) * t * C[0] + t**2 * B[0],
        (1 - t) ** 2 * A[1] + 2 * (1 - t) * t * C[1] + t**2 * B[1],
    )


def generate_points_around_line(a, b, c, num_points, max_distance):
    points = []
    for _ in range(num_points):
        t = random.uniform(0, 1)
        point_on_curve = bezier_curve(t, a, b, c)

        # Generate a random angle
        theta = random.uniform(0, 2 * math.pi)

        # Create a random vector with magnitude of max_distance
        dx = max_distance * math.cos(theta)
        dy = max_distance * math.sin(theta)

        # Add this vector to the random point on the curve
        new_point = (point_on_curve[0] + dx, point_on_curve[1] + dy)
        points.append(new_point)

    return points


def main():
    st.title("Interactive Bezier Curve")

    # Add sliders for A, B, C
    st.sidebar.markdown("## Control Points Coordinates")
    a_x = st.sidebar.slider("A X-coordinate", min_value=0.0, max_value=10.0, value=0.0)
    a_y = st.sidebar.slider("A Y-coordinate", min_value=0.0, max_value=10.0, value=0.0)

    b_x = st.sidebar.slider("B X-coordinate", min_value=0.0, max_value=10.0, value=10.0)
    b_y = st.sidebar.slider("B Y-coordinate", min_value=0.0, max_value=10.0, value=10.0)

    c_x = st.sidebar.slider("C X-coordinate", min_value=0.0, max_value=10.0, value=3.0)
    c_y = st.sidebar.slider("C Y-coordinate", min_value=0.0, max_value=10.0, value=8.0)

    # Add KDE sliders
    st.sidebar.markdown("## KDE Parameters")
    kernel = st.sidebar.selectbox("Kernel", options=["gau", "cos", "biw", "epa", "tri", "triw"])
    bandwidth = st.sidebar.slider("Bandwidth", min_value=0.1, max_value=2.0, value=0.5, step=0.1)


    # Add sliders for number of points and max distance
    st.sidebar.markdown("## Other Parameters")
    num_points = st.sidebar.slider("Number of Points", min_value=10, max_value=500, value=100)
    max_dist = st.sidebar.slider("Maximum Distance", min_value=0.1, max_value=5.0, value=2.0)

    a = (a_x, a_y)
    b = (b_x, b_y)
    c = (c_x, c_y)

    points = generate_points_around_line(a, b, c, num_points, max_dist)

    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(x_coords, y_coords, c="blue", marker="o", label="Generated Points")
    t_values = np.linspace(0, 1, 100)
    x_bezier = [bezier_curve(t, a, b, c)[0] for t in t_values]
    y_bezier = [bezier_curve(t, a, b, c)[1] for t in t_values]
    ax.plot(x_bezier, y_bezier, color="red", linewidth=2, label="Bent Line Segment")

    # KDE Plot
    sns.kdeplot(x=x_coords, y=y_coords, kernel=kernel, bw=bandwidth, ax=ax, fill=True)

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Points Around Bent Line Segment")
    ax.grid(True)

    st.pyplot(fig)


if __name__ == "__main__":
    main()
