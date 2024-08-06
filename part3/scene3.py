from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


class Scene3_3(VoiceoverScene):
    def construct(self):

        self.wait(2)

        mhd = [
            [0, 0.2, 0, 0, 0.8],
            [0.2, 0, 0.5, 0, 0],
            [0, 0.5, 0, 0.7, 0],
            [0, 0, 0.7, 0, 0.6],
            [0.8, 0, 0, 0.6, 0],
        ]

        mld = [
            [0, 0.3, 0.4, 0, 0],
            [0.3, 0, 0.2, 0.5, 0],
            [0.4, 0.2, 0, 0, 0.9],
            [0, 0.5, 0, 0, 0.1],
            [0, 0, 0.9, 0.1, 0],
        ]

        mhd_obj = Matrix(mhd).scale(0.8).to_edge(LEFT, buff=1)
        mld_obj = Matrix(mld).scale(0.8).to_edge(RIGHT, buff=1)

        label1 = (
            Text("High-Dimensional Representation")
            .scale(0.5)
            .next_to(mhd_obj, UP, buff=1.5)
        )
        label2 = (
            Text("Low-Dimensional Representation")
            .scale(0.5)
            .next_to(mld_obj, UP, buff=1.5)
        )

        self.play(FadeIn(mhd_obj), FadeIn(mld_obj), FadeIn(label1), FadeIn(label2))

        self.wait(0.6)

        self.play(
            mhd_obj.animate.scale(0.6).shift(UP * 1.5),
            mld_obj.animate.scale(0.6).shift(UP * 1.5),
            run_time=1,
        )

        # Cross-entropy

        formula = (
            MathTex(
                r"\sum_{i, j} w_{hd} (i, j) \log \left( \frac{w_{hd} (i, j)}{w_{ld} (i, j)} \right) + (1 - w_{hd} (i, j)) \log \left( \frac{1 - w_{hd} (i, j)}{1 - w_{ld} (i, j)} \right)"
            )
            .scale(0.7)
            .shift(DOWN)
        )

        self.play(FadeIn(formula))

        rect = SurroundingRectangle(formula, buff=0.2, color=WHITE)

        self.play(ShowPassingFlash(rect, time_width=0.3), run_time=2.5)

        self.wait(0.4)

        self.play(FadeOut(formula, mhd_obj, mld_obj, label1, label2))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene3_3()
    scene.render()
