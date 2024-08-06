from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class Scene2_2(VoiceoverScene):
    def construct(self):

        self.wait(2)

        # Points close to each other in high-dimensional space
        # also close in low-dimensional space
        self.next_section(skip_animations=False)

        axes = (
            Axes(
                x_range=[0, 1, 0.5],
                y_range=[0, 1, 0.5],
                x_length=4,
                y_length=4,
                axis_config={
                    "color": WHITE,
                    "include_numbers": True,
                    "include_tip": False,
                    "font_size": 18,
                },
            )
            .to_edge(LEFT, buff=1.0)
            .shift(UP)
        )

        nbline = (
            NumberLine(
                x_range=[0, 1, 0.5],
                include_numbers=True,
                length=4,
                color=WHITE,
                include_tip=False,
                font_size=18,
            )
            .to_edge(RIGHT, buff=1.0)
            .shift(UP)
        )

        hd_txt = Tex("High Dimensional Space").scale(0.8).next_to(axes, 2 * DOWN)
        ld_txt = Tex("Low Dimensional Space").scale(0.8).next_to(nbline, 2 * DOWN)
        ld_txt.move_to(np.array([ld_txt.get_center()[0], hd_txt.get_center()[1], 0]))
        ld_txt.add(Underline(ld_txt, buff=0.1))
        hd_txt.add(Underline(hd_txt, buff=0.1))

        mean = [0.5, 0.5]
        data_hd = [
            [0.49786874, 0.56421662],
            [0.56577013, 0.53735222],
            [0.52979158, 0.4653348],
            [0.29770518, 0.47921205],
            [0.46542031, 0.31758831],
        ]
        data_hd = np.concatenate((data_hd, [mean]))

        data_ld = [
            [0.41844153],
            [0.58673297],
            [0.76049533],
            [0.59953988],
            [0.63068882],
        ]
        data_ld = np.concatenate((data_ld, [[0.5]]))

        radius = 0.08
        points_hd = [
            Dot(axes.c2p(*point), color=GREEN, radius=radius) for point in data_hd
        ]

        points_ld = [
            Dot(nbline.n2p(point), color=GREEN, radius=radius) for point in data_ld
        ]

        self.play(FadeIn(axes, hd_txt))
        self.play(LaggedStartMap(Create, points_hd, lag_ratio=0.1))

        self.play(FadeIn(nbline, ld_txt))
        self.play(LaggedStartMap(Create, points_ld, lag_ratio=0.1))

        arrow1 = DoubleArrow(
            points_hd[0].get_center(),
            points_hd[4].get_center(),
            buff=0.08,
            stroke_width=2,
            max_stroke_width_to_length_ratio=10,
            max_tip_length_to_length_ratio=0.1,
        )
        arrow2 = DoubleArrow(
            points_ld[0].get_center(), points_ld[2].get_center(), buff=0
        )
        arrow2.shift(UP)

        line1 = DashedLine(arrow2.get_left(), points_ld[0].get_center())
        line2 = DashedLine(arrow2.get_right(), points_ld[2].get_center())

        self.play(Create(arrow1))
        self.play(
            LaggedStart(Create(arrow2), Create(line1), Create(line2), lag_ratio=0.3)
        )

        self.wait(0.5)

        outlier_hd_1 = np.array([0.1, 0.9])
        outlier_hd_2 = np.array([0.1, 0.2])
        dot_out_hd_1 = Dot(axes.c2p(*outlier_hd_1), color=RED, radius=radius)
        dot_out_hd_2 = Dot(axes.c2p(*outlier_hd_2), color=RED, radius=radius)

        outlier_ld_1 = np.array([0.1])
        outlier_ld_2 = np.array([0.9])
        dot_out_ld_1 = Dot(nbline.n2p(outlier_ld_1), color=RED, radius=radius)
        dot_out_ld_2 = Dot(nbline.n2p(outlier_ld_2), color=RED, radius=radius)

        self.play(Create(dot_out_hd_1), Create(dot_out_hd_2))
        self.play(Create(dot_out_ld_1), Create(dot_out_ld_2))

        arrow3 = DoubleArrow(
            dot_out_hd_1.get_center(),
            points_hd[0].get_center(),
            buff=0.08,
            stroke_width=2,
            max_stroke_width_to_length_ratio=10,
            max_tip_length_to_length_ratio=0.1,
        )
        arrow4 = DoubleArrow(
            dot_out_hd_2.get_center(),
            points_hd[0].get_center(),
            buff=0.08,
            stroke_width=2,
            max_stroke_width_to_length_ratio=10,
            max_tip_length_to_length_ratio=0.1,
        )
        arrow5 = DoubleArrow(
            points_ld[0].get_center(),
            dot_out_ld_1.get_center(),
            max_tip_length_to_length_ratio=0.2,
            buff=0.0,
        ).shift(1.5 * UP)
        arrow6 = DoubleArrow(
            points_ld[0].get_center(),
            dot_out_ld_2.get_center(),
            max_tip_length_to_length_ratio=0.2,
            buff=0.0,
        ).shift(1.5 * UP)

        line3 = DashedLine(arrow5.get_left(), dot_out_ld_1.get_center())
        line4 = DashedLine(arrow5.get_right(), points_ld[0].get_center())
        line5 = DashedLine(arrow6.get_right(), dot_out_ld_2.get_center())

        self.play(Create(arrow3), Create(arrow4))

        self.play(
            LaggedStart(Create(arrow5), Create(line3), Create(line4), lag_ratio=0.3)
        )
        self.play(LaggedStart(Create(arrow6), Create(line5), lag_ratio=0.3))

        self.play(
            FadeOut(
                arrow1,
                arrow2,
                line1,
                line2,
                arrow3,
                arrow4,
                line3,
                line4,
                line5,
                arrow5,
                arrow6,
            )
        )

        # Loss function
        self.next_section(skip_animations=False)

        self.play(
            FadeOut(
                axes,
                hd_txt,
                nbline,
                ld_txt,
                *points_hd,
                *points_ld,
                dot_out_hd_1,
                dot_out_hd_2,
                dot_out_ld_1,
                dot_out_ld_2,
            )
        )

        loss = MathTex(r"\text{Loss}~(HighDim,~LowDim)").scale(1.2)
        self.play(Write(loss))

        self.wait(0.4)

        derivative_loss = MathTex(
            r"\frac{\partial \text{Loss}~(HighDim,~LowDim)}{\partial LowDim}"
        ).scale(1.2)

        self.play(Transform(loss, derivative_loss))

        self.wait(0.6)

        self.play(FadeOut(loss))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene2_2()
    scene.render()
