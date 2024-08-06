from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


class Scene3_1(VoiceoverScene):
    def construct(self):
        self.wait(2)

        date = Tex("2018").scale(1.8)

        self.play(FadeIn(date))
        self.play(Flash(date, color=YELLOW, line_length=0.8, flash_radius=1.0))
        self.play(FadeOut(date))

        umap_txt = Tex("UMAP").scale(1.2).shift(UP)
        umap_txt_target = (
            Tex("Uniform Manifold Approximation and Projection").scale(1.2).shift(UP)
        )

        self.play(Write(umap_txt))
        self.play(Transform(umap_txt, umap_txt_target))

        self.wait(0.8)

        self.play(FadeOut(umap_txt), run_time=0.9)

        paper_svg = SVGMobject("./images/paper.svg").shift(UP)
        arrow = Arrow(
            start=paper_svg.get_bottom(),
            end=paper_svg.get_bottom() + DOWN * 2,
            max_stroke_width_to_length_ratio=10,
            stroke_width=8,
            buff=0.2,
        )

        self.wait(2)
        self.play(FadeIn(paper_svg, arrow))
        self.play(arrow.animate.shift(0.5 * DOWN), run_time=0.5)
        self.play(arrow.animate.shift(0.5 * UP), run_time=0.5)
        self.play(arrow.animate.shift(0.5 * DOWN), run_time=0.5)
        self.play(arrow.animate.shift(0.5 * UP), run_time=0.5)

        self.wait(0.4)

        self.play(FadeOut(paper_svg, arrow), run_time=1.0)

        self.wait(2)


if __name__ == "__main__":
    scene = Scene3_1()
    scene.render()
