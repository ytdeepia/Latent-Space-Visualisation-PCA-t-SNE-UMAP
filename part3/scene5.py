from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


class Scene3_5(VoiceoverScene):
    def construct(self):
        self.wait(2)

        podium = SVGMobject("./images/podium.svg").scale(1.4).shift(DOWN)
        self.play(FadeIn(podium))

        umap = Tex("UMAP").scale(1.2).next_to(podium, UP, buff=0.4)
        self.play(GrowFromPoint(umap, podium.get_center()))

        tsne = (
            Tex("t-SNE").scale(1.2).move_to(umap.get_center() + 0.9 * DOWN + 1.7 * LEFT)
        )
        self.play(GrowFromPoint(tsne, podium.get_center()))

        pca = (
            Tex("PCA").scale(1.2).move_to(umap.get_center() + 1.2 * DOWN + 1.5 * RIGHT)
        )
        self.play(GrowFromPoint(pca, podium.get_center()))

        self.wait(0.4)

        self.play(
            FadeOut(podium),
            umap.animate.move_to(ORIGIN),
            tsne.animate.move_to(4 * LEFT),
            pca.animate.move_to(4 * RIGHT),
        )

        self.play(
            ShowPassingFlash(SurroundingRectangle(pca, buff=0.2), time_width=0.4),
            run_time=2,
        )

        self.wait(0.4)

        self.play(
            ShowPassingFlash(SurroundingRectangle(tsne, buff=0.2), time_width=0.4),
            run_time=2,
        )
        self.wait(1)
        self.play(Flash(umap, color=YELLOW, line_length=0.8, flash_radius=1.0))

        self.wait(0.5)

        self.play(FadeOut(pca, umap, tsne))
        trimap = Tex("TriMap").scale(1.2).move_to(2 * LEFT)
        pacmap = Tex("PaCMAP").scale(1.2).move_to(2 * RIGHT)

        self.play(Write(trimap))
        self.play(Write(pacmap))

        self.wait(0.5)
        paper = SVGMobject("./images/paper.svg").scale(1.2).shift(2 * UP)
        self.play(FadeIn(paper))

        self.play(FadeOut(paper, trimap, pacmap), run_time=1.0)
        self.wait(2)


if __name__ == "__main__":
    scene = Scene3_5()
    scene.render()
