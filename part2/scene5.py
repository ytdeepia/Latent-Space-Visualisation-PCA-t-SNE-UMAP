from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class Scene2_5(VoiceoverScene):
    def construct(self):

        self.wait(2)

        # Perplexity formula
        self.next_section(skip_animations=False)

        pijformula = MathTex(
            "" r"p_{i, j} = e^{-\frac{||x_i - x_j||^2}{2", "\sigma_{i}^2}}"
        ).scale(1.5)

        self.play(Write(pijformula))

        rect = SurroundingRectangle(pijformula[1], buff=0.1)
        self.play(Create(rect))

        self.play(FadeOut(rect), run_time=0.8)

        self.play(pijformula.animate.to_edge(UP))

        perplexity = MathTex(r"\text{Perplexity} =", r"2^{\text{H}(P_i)}")
        self.play(Write(perplexity))

        self.play(perplexity.animate.to_edge(LEFT))

        entropy = MathTex(
            r"\text{H}(P_i) = -\sum_{j \neq i} p_{i, j} \log_2 p_{i, j}"
        ).to_edge(RIGHT)

        self.play(Write(entropy))

        self.wait()
        self.play(Indicate(pijformula[1], color=RED, scale_factor=1.6), run_time=2)
        self.play(Indicate(perplexity[1], color=RED, scale_factor=1.2), run_time=2)

        self.wait(0.6)

        self.play(FadeOut(entropy, perplexity, pijformula), run_time=0.8)

        # Perplexity evolution animation
        self.next_section(skip_animations=False)

        axes = Axes(
            x_range=[0, 1, 1],
            y_range=[0, 1, 1],
            x_length=6,
            y_length=6,
            axis_config={
                "color": WHITE,
                "include_numbers": True,
                "include_tip": False,
                "font_size": 24,
            },
        ).shift(DOWN * 0.5)

        # Legend

        colors = [
            RED,
            GREEN,
            BLUE,
            YELLOW,
            PURPLE,
            ORANGE,
            PINK,
            TEAL,
            DARK_BROWN,
            GREY,
        ]

        colorbar = VGroup()
        clabels = VGroup()
        for i, color in enumerate(colors):
            rect = Rectangle(
                width=0.6,
                height=0.3,
                color=color,
                fill_opacity=1,
                stroke_width=0,
            )
            label = Tex(f"{i}", color=WHITE).scale(0.4)
            colorbar.add(rect)
            clabels.add(label)

        colorbar.next_to(axes, direction=RIGHT, buff=0.5)
        colorbar.arrange(DOWN, buff=0.2).to_edge(RIGHT, buff=1.5)

        for idx, clabel in enumerate(clabels):
            clabel.next_to(colorbar[idx], direction=RIGHT, buff=0.3)

        # First perplexity

        data = np.load("data/30.npy")
        coordinates = data[:, :2]
        labels = data[:, 2]

        normalized_coordinates = (coordinates - coordinates.min(axis=0)) / (
            coordinates.max(axis=0) - coordinates.min(axis=0)
        )

        dots = VGroup()
        for i, (x, y) in enumerate(normalized_coordinates):
            dot = Dot(
                axes.c2p(x, y),
                color=colors[int(labels[i])],
                radius=0.025,
                fill_opacity=0.6,
            )
            dots.add(dot)

        perp_counter = MathTex("Perplexity = 30").scale(0.7).next_to(axes, UP)

        self.play(FadeIn(axes, colorbar, clabels, dots, perp_counter), run_time=2)

        self.wait(0.5)

        # Perplexity evolution
        for perp in range(40, 71, 10):

            data = np.load(f"data/{perp}.npy")
            coordinates = data[:, :2]
            labels = data[:, 2]

            normalized_coordinates = (coordinates - coordinates.min(axis=0)) / (
                coordinates.max(axis=0) - coordinates.min(axis=0)
            )

            new_dots = VGroup()
            for i, (x, y) in enumerate(normalized_coordinates):
                dot = Dot(
                    axes.c2p(x, y),
                    color=colors[int(labels[i])],
                    radius=0.025,
                    fill_opacity=0.6,
                )
                new_dots.add(dot)
            new_perp_counter = (
                MathTex(f"Perplexity = {perp}").scale(0.7).next_to(axes, UP)
            )

            self.play(
                Transform(dots, new_dots),
                Transform(perp_counter, new_perp_counter),
                run_time=1.8,
            )

            self.wait(0.4)

        for perp in range(70, 101, 10):

            data = np.load(f"data/{perp}.npy")
            coordinates = data[:, :2]
            labels = data[:, 2]

            normalized_coordinates = (coordinates - coordinates.min(axis=0)) / (
                coordinates.max(axis=0) - coordinates.min(axis=0)
            )

            new_dots = VGroup()
            for i, (x, y) in enumerate(normalized_coordinates):
                dot = Dot(
                    axes.c2p(x, y),
                    color=colors[int(labels[i])],
                    radius=0.025,
                    fill_opacity=0.6,
                )
                new_dots.add(dot)
            new_perp_counter = (
                MathTex(f"Perplexity = {perp}").scale(0.7).next_to(axes, UP)
            )

            self.play(
                Transform(dots, new_dots),
                Transform(perp_counter, new_perp_counter),
                run_time=1.8,
            )

            self.wait(0.4)

        self.wait(0.6)

        self.play(FadeOut(axes, dots, perp_counter, colorbar, clabels), run_time=1)

        self.wait(2)


if __name__ == "__main__":
    scene = Scene2_5()
    scene.render()
