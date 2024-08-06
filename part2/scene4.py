from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class Scene2_4(VoiceoverScene):
    def construct(self):
        self.wait(2)

        # KL formula and visualisation
        self.next_section(skip_animations=False)

        kltxt = Tex("Kullback-Leibler divergence")
        self.play(Write(kltxt))
        self.play(kltxt.animate.to_edge(UP))

        klformula = MathTex("D_{KL}(P||Q) = \\sum_{i} P(i) \\log \\frac{P(i)}{Q(i)}")
        self.play(Write(klformula))

        self.wait(0.5)

        self.play(FadeOut(klformula), run_time=0.6)

        # Display the distributions along with the KL
        self.next_section(skip_animations=False)

        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[0, 1, 0.2],
            axis_config={
                "color": WHITE,
                "include_numbers": True,
                "include_tip": False,
                "font_size": 24,
            },
        ).scale(0.7)

        axes_labels = axes.get_axis_labels(x_label="x", y_label="f(x)")
        mean1, std1 = 0, 1
        mean2, std2 = 2, 1

        gaussian1 = lambda x: np.exp(-((x - mean1) ** 2) / (2 * std1**2)) / (
            std1 * np.sqrt(2 * np.pi)
        )
        gaussian2 = lambda x: np.exp(-((x - mean2) ** 2) / (2 * std2**2)) / (
            std2 * np.sqrt(2 * np.pi)
        )

        legend = VGroup(
            Line(start=ORIGIN, end=0.3 * RIGHT, color=GREEN).shift(LEFT + UP),
            Tex("P").next_to(
                Line(start=ORIGIN, end=0.3 * RIGHT, color=GREEN).shift(LEFT + UP), RIGHT
            ),
            Line(start=ORIGIN, end=0.3 * RIGHT, color=RED).shift(LEFT + 0.5 * UP),
            Tex("Q").next_to(
                Line(start=ORIGIN, end=0.3 * RIGHT, color=RED).shift(LEFT + 0.5 * UP),
                RIGHT,
            ),
            Line(start=ORIGIN, end=0.3 * RIGHT, color=BLUE).shift(LEFT),
            Tex("KL divergence").next_to(
                Line(start=ORIGIN, end=0.3 * RIGHT, color=BLUE).shift(LEFT), RIGHT
            ),
        ).move_to(axes.get_corner(UL))

        self.play(FadeIn(axes, legend, axes_labels), run_time=1.5)

        graph1 = axes.plot(gaussian1, color=GREEN)
        graph2 = axes.plot(gaussian2, color=RED)

        self.play(Create(graph1), Create(graph2), run_time=2)

        kl_div = lambda x: gaussian1(x) * np.log(gaussian1(x) / gaussian2(x))

        graph3 = axes.plot(kl_div, color=BLUE)
        area = axes.get_area(graph3, x_range=[-5, 5], color=BLUE, opacity=0.3)

        self.play(Create(graph3), Create(area), run_time=2)

        self.play(FadeOut(area), run_time=0.6)

        mean2 = 3

        self.play(
            graph2.animate.become(axes.plot(gaussian2, color=RED)),
            graph3.animate.become(axes.plot(kl_div, color=BLUE)),
            run_time=2,
        )

        self.wait(2)

        mean2 = 1

        self.play(
            graph2.animate.become(axes.plot(gaussian2, color=RED)),
            graph3.animate.become(axes.plot(kl_div, color=BLUE)),
            run_time=2,
        )

        self.wait(0.6)

        mean1 = 0
        mean2 = 0

        self.play(
            graph1.animate.become(axes.plot(gaussian1, color=GREEN)),
            graph2.animate.become(axes.plot(gaussian2, color=RED)),
            graph3.animate.become(axes.plot(kl_div, color=BLUE)),
            run_time=3,
        )

        self.play(
            FadeOut(axes, legend, graph1, graph2, graph3, axes_labels, kltxt),
            run_time=1,
        )

        # Compute the derivative
        self.next_section(skip_animations=False)

        hd_txt = Tex("High Dimensional Distribution").scale(0.8)
        ld_txt = Tex("Low Dimensional Distribution").scale(0.8)
        hd_txt.to_edge(UP).shift(3 * LEFT)
        ld_txt.to_edge(UP).shift(3 * RIGHT)

        ul_hd = Underline(hd_txt, buff=0.1)
        ul_ld = Underline(ld_txt, buff=0.1)

        pijformula = MathTex(
            r"p_{i, j} = \frac{e^{-\frac{||x_i - x_j||^2}{2\sigma_{i}^2}}}{\sum_{k \neq i} e^{-\frac{||x_i - x_k||^2}{2\sigma_{i}^2}}}"
        ).scale(0.8)
        qijformula = MathTex(
            r"q_{i, j} = \frac{e^{-||y_i - y_j||^2}}{\sum_{k \neq i} e^{-||y_i - y_k||}}"
        ).scale(0.8)

        pijformula.next_to(hd_txt, DOWN, buff=1.0)
        qijformula.next_to(ld_txt, DOWN, buff=1.0)

        klformula = MathTex(
            r"D_{KL}(P||Q) = \sum_{i} \sum_{j} p_{i, j} \log \frac{p_{i, j}}{q_{i, j}}"
        ).move_to(DOWN)

        self.play(FadeIn(hd_txt, ul_hd), Write(pijformula))
        self.play(FadeIn(ld_txt, ul_ld), Write(qijformula))
        self.play(Write(klformula), run_time=2)

        self.wait(0.4)

        klformuladerivative = MathTex(
            r"\frac{\partial D_{KL}(P||Q)}{\partial y_i} = 2 \sum_{j} (p_{i, j} - q_{i, j} + p_{j, i} - q_{j, i})(y_i - y_j)"
        ).move_to(DOWN)

        self.play(Transform(klformula, klformuladerivative), run_time=2)

        self.wait(0.5)

        self.play(FadeOut(pijformula, qijformula, klformula), run_time=1)

        # Show the low-dimension representation evolving
        self.next_section(skip_animations=False)

        np.random.seed(0)
        points_2d_cluster1 = np.random.randn(10, 2) * 0.5 + np.array([-2, 2])
        points_2d_cluster2 = np.random.randn(10, 2) * 0.5 + np.array([2, 2])
        points_2d_cluster3 = np.random.randn(10, 2) * 0.5 + np.array([0, -2])
        points_2d_cluster4 = np.random.randn(10, 2) * 0.5 + np.array([2, -2])
        points_2d = np.vstack(
            [
                points_2d_cluster1,
                points_2d_cluster2,
                points_2d_cluster3,
                points_2d_cluster4,
            ]
        )

        # Create 1D points placed randomly
        points_1d = np.random.uniform(-3, 3, (40, 1))

        # Plot the 2D points with different colors for each cluster
        axes_2d = (
            Axes(
                x_range=[-5, 5, 1],
                y_range=[-5, 5, 1],
                axis_config={
                    "color": WHITE,
                    "include_numbers": True,
                    "include_tip": False,
                    "font_size": 24,
                },
            )
            .scale(0.5)
            .shift(3 * LEFT)
        )
        points_2d_plot = VGroup(
            *[
                Dot(
                    point=axes_2d.coords_to_point(x, y),
                    color=color,
                    radius=0.06,
                    fill_opacity=0.6,
                )
                for (x, y), color in zip(
                    points_2d, [BLUE] * 10 + [GREEN] * 10 + [RED] * 10 + [YELLOW] * 10
                )
            ]
        )

        # Plot the 1D points
        number_line = (
            NumberLine(x_range=[-5, 5, 1], include_numbers=True)
            .scale(0.5)
            .shift(3 * RIGHT)
        )
        points_1d_plot = VGroup(
            *[
                Dot(point=number_line.n2p(x), radius=0.06, fill_opacity=0.6)
                for x in points_1d
            ]
        )

        def compute_gradient(points_1d, points_2d):
            n_points = len(points_1d)
            gradients = np.zeros_like(points_1d)
            for i in range(n_points):
                for j in range(n_points):
                    if i != j:
                        y_i, y_j = points_2d[i], points_2d[j]
                        x_i, x_j = points_1d[i], points_1d[j]
                        p_ij = np.exp(-np.linalg.norm(x_i - x_j) ** 2)
                        q_ij = np.exp(-np.linalg.norm(y_i - y_j) ** 2)
                        p_ij /= np.sum(
                            [
                                np.exp(-np.linalg.norm(x_i - points_1d[k]) ** 2)
                                for k in range(n_points)
                            ]
                        )
                        q_ij /= np.sum(
                            [
                                np.exp(-np.linalg.norm(y_i - points_2d[k]) ** 2)
                                for k in range(n_points)
                            ]
                        )
                        gradient = 2 * (p_ij - q_ij) * (x_i - x_j)
                        gradients[i] += gradient
            return gradients

        # Number of iterations for the transformation
        num_iterations = 80

        self.play(FadeIn(axes_2d))
        self.play(Create(points_2d_plot))
        self.wait(1)
        self.play(FadeIn(number_line))
        self.play(Create(points_1d_plot))

        iteration_counter = Tex("Iteration: 0").next_to(number_line, UP, buff=2)

        for i in range(num_iterations):
            gradients = compute_gradient(points_1d, points_2d)
            points_1d += 0.1 * gradients

            points_1d_transformed_plot = VGroup(
                *[
                    Dot(point=number_line.n2p(x), radius=0.06, fill_opacity=0.6)
                    for x in points_1d
                ]
            )

            new_iteration_counter = Tex(f"Iteration: {i}").next_to(
                number_line, UP, buff=2
            )

            self.play(
                Transform(points_1d_plot, points_1d_transformed_plot),
                Transform(iteration_counter, new_iteration_counter),
                run_time=0.2,
            )

        self.play(
            FadeOut(
                axes_2d,
                points_2d_plot,
                number_line,
                points_1d_plot,
                iteration_counter,
                hd_txt,
                ld_txt,
                ul_hd,
                ul_ld,
            ),
            run_time=1,
        )

        txt = Tex("Perplexity")
        self.play(Write(txt))

        self.play(FadeOut(txt), run_time=1)

        self.wait(2)


if __name__ == "__main__":
    scene = Scene2_4()
    scene.render()
