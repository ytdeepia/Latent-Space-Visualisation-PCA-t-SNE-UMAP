from manim import *


class GaussianCurve(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.5, 0.1],
            axis_config={"include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="f(x)")

        # Gaussian function
        def gaussian(x):
            return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)

        # Plot the Gaussian curve
        graph = axes.plot(gaussian, color=BLUE)

        # Shade the area under the Gaussian curve
        area = axes.get_area(graph, x_range=[-4, 4], color=BLUE, opacity=0.3)

        # Add the elements to the scene
        self.add(axes, axes_labels, graph, area)
        self.wait()


# To render the scene, use:
# manim -pql script.py GaussianCurve
