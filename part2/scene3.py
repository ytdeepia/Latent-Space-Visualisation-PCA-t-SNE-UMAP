from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class Scene2_3(VoiceoverScene):
    def construct(self):

        self.wait(2)

        # Map distances to a Gaussian
        self.next_section(skip_animations=False)

        # Draw points
        central_dot = Dot(np.array([0, 0, 0]), color=RED, radius=0.15)
        pt1 = np.array([1.0, 1.6, 0])
        pt2 = np.array([-1.15, 0.9, 0])
        pt3 = np.array([-1.8, -1.2, 0])
        pt4 = np.array([0.9, -1.1, 0])
        pt5 = np.array([0.7, 2.2, 0])

        dot1 = Dot(pt1, color=BLUE, radius=0.15)
        dot2 = Dot(pt2, color=BLUE, radius=0.15)
        dot3 = Dot(pt3, color=BLUE, radius=0.15)
        dot4 = Dot(pt4, color=BLUE, radius=0.15)
        dot5 = Dot(pt5, color=BLUE, radius=0.15)

        labels = VGroup(
            MathTex("x_0").next_to(central_dot, RIGHT),
            MathTex("x_1").next_to(dot1, UP),
            MathTex("x_2").next_to(dot2, UP),
            MathTex("x_3").next_to(dot3, UP),
            MathTex("x_4").next_to(dot4, UP),
            MathTex("x_5").next_to(dot5, UP),
        )

        dots = VGroup(dot1, dot2, dot3, dot4, dot5)

        self.play(Create(central_dot))
        self.play(LaggedStartMap(Create, dots, lag_ratio=0.5), run_time=2)
        self.play(FadeIn(labels))

        arrow1 = DoubleArrow(
            central_dot.get_center(),
            dot1.get_center(),
            color=WHITE,
            buff=0.15,
            max_tip_length_to_length_ratio=0.15,
            stroke_width=2,
        )
        arrow2 = DoubleArrow(
            central_dot.get_center(),
            dot2.get_center(),
            color=WHITE,
            buff=0.15,
            max_tip_length_to_length_ratio=0.15,
            stroke_width=2,
        )
        arrow3 = DoubleArrow(
            central_dot.get_center(),
            dot3.get_center(),
            color=WHITE,
            buff=0.15,
            max_tip_length_to_length_ratio=0.15,
            stroke_width=2,
        )
        arrow4 = DoubleArrow(
            central_dot.get_center(),
            dot4.get_center(),
            color=WHITE,
            buff=0.15,
            max_tip_length_to_length_ratio=0.15,
            stroke_width=2,
        )
        arrow5 = DoubleArrow(
            central_dot.get_center(),
            dot5.get_center(),
            color=WHITE,
            buff=0.15,
            max_tip_length_to_length_ratio=0.15,
            stroke_width=2,
        )

        arrows = VGroup(arrow1, arrow2, arrow3, arrow4, arrow5)

        scatter_plot = VGroup(central_dot, dots, arrows, labels)

        self.play(LaggedStartMap(Create, arrows, lag_ratio=0.5), run_time=2)

        self.wait(0.8)

        self.play(scatter_plot.animate.scale(0.6).to_edge(LEFT, buff=1.0))

        # Draw the Gaussian
        axes = (
            Axes(
                x_range=[0, 3, 1],
                y_range=[0, 1.2, 0.2],
                axis_config={
                    "color": WHITE,
                    "include_numbers": True,
                    "include_tip": False,
                    "font_size": 24,
                },
            )
            .scale(0.7)
            .to_edge(RIGHT, buff=1.0)
        )

        # Define the Gaussian function
        def gaussian(x):
            return np.exp(-(x**2) / 2)

        # Create the plot of the Gaussian function
        gaussian_curve = axes.plot(lambda x: gaussian(x), color=WHITE)

        # Create the coordinates of the points
        points = [
            np.array([1.2, 1.6, 0]),
            np.array([-1.15, 0.9, 0]),
            np.array([-1.8, -1.2, 0]),
            np.array([0.9, -1.1, 0]),
            np.array([0.7, 2.2, 0]),
        ]

        distances = [np.linalg.norm(pt) for pt in points]
        point_dots = [
            Dot(axes.c2p(x, 0), color=BLUE, radius=0.1, fill_opacity=0.8)
            for x in distances
        ]

        # Calculate intersections and dashed lines
        dashed_lines = []
        intersection_points = []
        for x in distances:
            y = gaussian(x)
            intersection = axes.c2p(x, y)
            intersection_points.append(intersection)
            dashed_line = DashedLine(axes.c2p(x, 0), intersection, color=WHITE)
            dashed_lines.append(dashed_line)

        intersection_dots = [
            Dot(point, color=BLUE, radius=0.1, fill_opacity=0.8)
            for point in intersection_points
        ]
        central_point = Dot(axes.c2p(0, 0), color=RED, radius=0.2, fill_opacity=0.8)

        self.play(FadeIn(axes, central_point))

        self.play(Create(gaussian_curve))
        self.play(*[Create(dot) for dot in point_dots])

        self.wait(0.4)

        self.play(Indicate(point_dots[0], scale_factor=2.2, color=BLUE), run_time=1.8)
        self.play(Indicate(central_point, scale_factor=1.4, color=RED), run_time=1.8)

        self.wait(0.5)

        self.play(*[Create(line) for line in dashed_lines])
        self.play(*[Create(dot) for dot in intersection_dots])

        gaussian_formula = MathTex(
            r"p_{0, j} = e^{-\frac{||x_0 - x_j||^2}{2", "\sigma_{0}", "^2}}"
        ).to_corner(UR, buff=1.0)

        self.play(Write(gaussian_formula), run_time=2)

        self.wait(0.6)

        # Standard deviation influences the spread of the Gaussian
        self.next_section(skip_animations=False)

        self.wait()
        rect = SurroundingRectangle(gaussian_formula[1], buff=0.2, color=YELLOW)
        self.play(Create(rect))

        self.play(FadeOut(*intersection_dots, *dashed_lines))

        self.wait(0.6)

        def gaussian_wide(x):
            return np.exp(-(x**2) / 8)

        def gaussian_narrow(x):
            return np.exp(-(x**2) / 0.5)

        gaussian_curve_wide = axes.plot(lambda x: gaussian_wide(x), color=WHITE)
        gaussian_curve_narrow = axes.plot(lambda x: gaussian_narrow(x), color=WHITE)

        self.wait(1)
        self.play(Transform(gaussian_curve, gaussian_curve_wide), run_time=2)

        self.wait(0.3)

        self.play(Transform(gaussian_curve, gaussian_curve_narrow), run_time=2)

        self.play(FadeOut(rect), run_time=0.6)

        self.wait(0.7)

        # Normalize by the other Gaussians
        self.next_section(skip_animations=False)

        gaussian_generalized_formula = MathTex(
            r"p_{i, j} = e^{-\frac{||x_i - x_j||^2}{2\sigma_{i}^2}}",
        ).to_corner(UR, buff=1.0)

        self.play(Transform(gaussian_formula, gaussian_generalized_formula))

        rect = SurroundingRectangle(gaussian_formula, buff=0.1, color=YELLOW)
        self.play(Create(rect))

        self.wait(0.5)

        self.play(FadeOut(rect))
        gaussian_normalized_formula = MathTex(
            r"p_{i, j} = \frac{e^{-\frac{||x_i - x_j||^2}{2\sigma_{i}^2}}}{\sum_{k \neq i} e^{-\frac{||x_i - x_k||^2}{2\sigma_{i}^2}}}",
        ).to_corner(UR, buff=1.0)

        self.play(Transform(gaussian_formula, gaussian_normalized_formula))

        # Low dimension space with points at random
        # and computing the Gaussians there too
        self.next_section(skip_animations=False)

        self.play(
            FadeOut(
                gaussian_formula,
                scatter_plot,
                gaussian_curve,
                axes,
                central_point,
                *point_dots,
            )
        )

        nb_line = NumberLine(
            x_range=[-2, 2, 1],
            length=8,
            include_numbers=True,
            include_tip=False,
            color=WHITE,
            font_size=24,
        )

        central_point = Dot(nb_line.n2p(0), color=RED, radius=0.2, fill_opacity=0.8)
        pt1 = np.array([0.3])
        pt2 = np.array([-0.5])
        pt3 = np.array([1.2])
        pt4 = np.array([-0.9])
        pt5 = np.array([1.6])

        dot1 = Dot(nb_line.n2p(pt1[0]), color=BLUE, radius=0.1, fill_opacity=0.8)
        dot2 = Dot(nb_line.n2p(pt2[0]), color=BLUE, radius=0.1, fill_opacity=0.8)
        dot3 = Dot(nb_line.n2p(pt3[0]), color=BLUE, radius=0.1, fill_opacity=0.8)
        dot4 = Dot(nb_line.n2p(pt4[0]), color=BLUE, radius=0.1, fill_opacity=0.8)
        dot5 = Dot(nb_line.n2p(pt5[0]), color=BLUE, radius=0.1, fill_opacity=0.8)

        dots = VGroup(dot1, dot2, dot3, dot4, dot5)

        labels = VGroup(
            MathTex("y_0").next_to(central_point, UP),
            MathTex("y_1").next_to(dot1, UP),
            MathTex("y_2").next_to(dot2, UP),
            MathTex("y_3").next_to(dot3, UP),
            MathTex("y_4").next_to(dot4, UP),
            MathTex("y_5").next_to(dot5, UP),
        )

        self.play(FadeIn(nb_line, central_point))
        self.play(LaggedStartMap(Create, dots, lag_ratio=0.5), run_time=2)
        self.play(FadeIn(labels))

        self.wait(0.3)

        scatter_plot1d = VGroup(nb_line, central_point, dots, labels)

        self.play(scatter_plot1d.animate.scale(0.4).to_edge(LEFT, buff=0.6))

        axes = (
            Axes(
                x_range=[0, 3, 1],
                y_range=[0, 1.2, 0.2],
                axis_config={
                    "color": WHITE,
                    "include_numbers": True,
                    "include_tip": False,
                    "font_size": 24,
                },
            )
            .scale(0.7)
            .to_edge(RIGHT, buff=1.0)
        )

        gaussian_curve = axes.plot(lambda x: gaussian(x), color=WHITE)

        points = [pt1, pt2, pt3, pt4, pt5]
        distances = [np.linalg.norm(pt) for pt in points]

        point_dots = [
            Dot(axes.c2p(x, 0), color=BLUE, radius=0.1, fill_opacity=0.8)
            for x in distances
        ]

        dashed_lines = []
        intersection_points = []
        for x in distances:
            y = gaussian(x)
            intersection = axes.c2p(x, y)
            intersection_points.append(intersection)
            dashed_line = DashedLine(axes.c2p(x, 0), intersection, color=WHITE)
            dashed_lines.append(dashed_line)

        intersection_dots = [
            Dot(point, color=BLUE, radius=0.1, fill_opacity=0.8)
            for point in intersection_points
        ]
        central_point = Dot(axes.c2p(0, 0), color=RED, radius=0.2, fill_opacity=0.8)

        self.play(FadeIn(axes, central_point))

        self.play(Create(gaussian_curve), run_time=2)
        self.play(*[Create(dot) for dot in point_dots])
        self.wait(0.5)
        self.play(*[Create(line) for line in dashed_lines])
        self.play(*[Create(dot) for dot in intersection_dots])

        self.wait(1)

        gaussian_formula1d = MathTex(
            r"q_{i, j} = \frac{e^{-||y_i - y_j||^2}}{\sum_{k \neq i} e^{-||y_i - y_k||}}",
        ).to_corner(UR, buff=1.0)

        self.play(FadeIn(gaussian_formula1d))

        self.wait()

        # Display the two probability distributions
        self.next_section(skip_animations=False)

        self.play(
            FadeOut(
                gaussian_curve,
                *point_dots,
                *dashed_lines,
                *intersection_dots,
                axes,
                central_point,
            ),
            run_time=0.9,
        )

        self.play(
            scatter_plot1d.animate.scale(1.3).move_to(2 * DOWN + 3 * LEFT),
            gaussian_formula1d.animate.move_to(2 * UP + 3 * LEFT),
        )
        scatter_plot.remove(arrows)
        scatter_plot.move_to(DOWN + 3 * RIGHT)

        gaussian_formula2d = MathTex(
            r"p_{i, j} = \frac{e^{-\frac{||x_i - x_j||^2}{2\sigma_{i}^2}}}{\sum_{k \neq i} e^{-\frac{||x_i - x_k||^2}{2\sigma_{i}^2}}}",
        ).move_to(2 * UP + 3 * RIGHT)

        self.play(FadeIn(scatter_plot, gaussian_formula2d))

        self.play(
            FadeOut(
                gaussian_formula1d, scatter_plot1d, gaussian_formula2d, scatter_plot
            ),
            run_time=1,
        )

        self.wait(2)


if __name__ == "__main__":
    scene = Scene2_3()
    scene.render()
