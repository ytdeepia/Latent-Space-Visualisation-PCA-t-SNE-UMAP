from manim import *
from util import read_script, grayscale_to_hex
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.elevenlabs import ElevenLabsService

import numpy as np
from sklearn.decomposition import PCA


class Scene1_2(VoiceoverScene):
    def construct(self):

        self.wait(2)

        np.random.seed(42)

        axes = Axes(
            x_range=[0, 1, 0.5],
            y_range=[0, 1, 0.5],
            x_length=6,
            y_length=6,
            axis_config={
                "color": WHITE,
                "include_numbers": True,
                "include_tip": False,
                "font_size": 18,
            },
        ).move_to(UP * 0.5)

        x_label = (
            axes.get_x_axis_label("Blood Pressure")
            .set(font_size=18)
            .move_to(axes.get_x_axis().get_right() + 0.5 * DOWN)
        )
        y_label = (
            axes.get_y_axis_label("Cholesterol")
            .set(font_size=18)
            .move_to(axes.get_y_axis().get_top() + 0.2 * UP)
        )

        # Set random seed for reproducibility
        np.random.seed(42)

        # Draw the intial distribution
        self.next_section(skip_animations=False)

        mean1 = [0.2, 0.7]  # Upper left corner
        mean2 = [0.7, 0.7]  # Upper right corner
        mean3 = [0.7, 0.2]  # Lower right corner
        cov = [[0.005, 0], [0, 0.005]]  # Small covariance for tight clusters

        cluster1 = np.random.multivariate_normal(mean1, cov, 100)
        cluster2 = np.random.multivariate_normal(mean2, cov, 100)
        cluster3 = np.random.multivariate_normal(mean3, cov, 100)
        data = np.vstack((cluster1, cluster2, cluster3))

        cluster1_dots = VGroup()
        for point in cluster1:
            dot = Dot(
                axes.c2p(point[0], point[1]), color=BLUE, radius=0.03, fill_opacity=0.8
            )
            cluster1_dots.add(dot)
        cluster2_dots = VGroup()
        for point in cluster2:
            dot = Dot(
                axes.c2p(point[0], point[1]), color=RED, radius=0.03, fill_opacity=0.8
            )
            cluster2_dots.add(dot)
        cluster3_dots = VGroup()
        for point in cluster3:
            dot = Dot(
                axes.c2p(point[0], point[1]), color=GREEN, radius=0.03, fill_opacity=0.8
            )
            cluster3_dots.add(dot)

        self.play(FadeIn(axes), Write(x_label), Write(y_label), run_time=2)

        self.wait(0.5)

        self.play(LaggedStartMap(Create, cluster1_dots, lag_ratio=0.01), run_time=1)
        self.play(LaggedStartMap(Create, cluster2_dots, lag_ratio=0.01), run_time=1)
        self.play(LaggedStartMap(Create, cluster3_dots, lag_ratio=0.01), run_time=1)

        self.wait(0.8)

        # Project onto x-axis
        self.next_section(skip_animations=False)

        cluster1_original = cluster1_dots.copy()
        cluster2_original = cluster2_dots.copy()
        cluster3_original = cluster3_dots.copy()

        cluster1_x = VGroup(
            *[
                Dot(point=axes.c2p(x, 0), radius=0.05, color=BLUE, fill_opacity=0.8)
                for x, _ in cluster1
            ]
        )

        cluster2_x = VGroup(
            *[
                Dot(point=axes.c2p(x, 0), radius=0.05, color=RED, fill_opacity=0.8)
                for x, _ in cluster2
            ]
        )

        cluster3_x = VGroup(
            *[
                Dot(point=axes.c2p(x, 0), radius=0.05, color=GREEN, fill_opacity=0.8)
                for x, _ in cluster3
            ]
        )

        self.play(
            Transform(cluster1_dots, cluster1_x),
            Transform(cluster2_dots, cluster2_x),
            Transform(cluster3_dots, cluster3_x),
            run_time=3,
        )

        self.wait(3)

        self.play(
            Transform(cluster1_dots, cluster1_original),
            Transform(cluster2_dots, cluster2_original),
            Transform(cluster3_dots, cluster3_original),
            run_time=2,
        )

        self.wait(0.4)

        cluster1_y = VGroup(
            *[
                Dot(point=axes.c2p(0, y), radius=0.05, color=BLUE, fill_opacity=0.8)
                for _, y in cluster1
            ]
        )

        cluster2_y = VGroup(
            *[
                Dot(point=axes.c2p(0, y), radius=0.05, color=RED, fill_opacity=0.8)
                for _, y in cluster2
            ]
        )

        cluster3_y = VGroup(
            *[
                Dot(point=axes.c2p(0, y), radius=0.05, color=GREEN, fill_opacity=0.8)
                for _, y in cluster3
            ]
        )

        self.play(
            Transform(cluster1_dots, cluster1_y),
            Transform(cluster2_dots, cluster2_y),
            Transform(cluster3_dots, cluster3_y),
            run_time=2,
        )

        self.play(
            Transform(cluster1_dots, cluster1_original),
            Transform(cluster2_dots, cluster2_original),
            Transform(cluster3_dots, cluster3_original),
            run_time=1.5,
        )

        # Project onto principal component
        self.next_section(skip_animations=False)

        pca = PCA(n_components=1)
        data_1d = pca.fit_transform(data)

        eigenvector = pca.components_[0]
        mean = [0.5, 0.5]

        vector1 = Arrow(
            start=axes.c2p(*mean),
            end=axes.c2p(*(mean + eigenvector / 5)),
            buff=0,
            color=WHITE,
            stroke_width=8,
            max_stroke_width_to_length_ratio=10,
            z_index=1,
        )

        self.play(GrowArrow(vector1), run_time=1)

        self.wait(0.5)

        projections = [
            np.dot(eigenvector, np.array([x, y]) - mean) * eigenvector + mean
            for x, y in data
        ]

        cluster1_1d = projections[:100]
        cluster2_1d = projections[100:200]
        cluster3_1d = projections[200:]

        cluster1_proj = VGroup(
            *[
                Dot(
                    point=axes.c2p(point[0], point[1]),
                    radius=0.05,
                    color=BLUE,
                    fill_opacity=0.8,
                )
                for point in cluster1_1d
            ]
        )
        cluster2_proj = VGroup(
            *[
                Dot(
                    point=axes.c2p(point[0], point[1]),
                    radius=0.05,
                    color=RED,
                    fill_opacity=0.8,
                )
                for point in cluster2_1d
            ]
        )
        cluster3_proj = VGroup(
            *[
                Dot(
                    point=axes.c2p(point[0], point[1]),
                    radius=0.05,
                    color=GREEN,
                    fill_opacity=0.8,
                )
                for point in cluster3_1d
            ]
        )

        self.play(FadeOut(vector1), run_time=1)

        self.play(
            Transform(cluster1_dots, cluster1_proj),
            Transform(cluster2_dots, cluster2_proj),
            Transform(cluster3_dots, cluster3_proj),
            run_time=2,
        )

        self.wait(0.4)

        cluster1_proj_x = VGroup(
            *[
                Dot(point=axes.c2p(x, 0), radius=0.05, color=BLUE, fill_opacity=0.8)
                for x, _ in cluster1_1d
            ]
        )
        cluster2_proj_x = VGroup(
            *[
                Dot(point=axes.c2p(x, 0), radius=0.05, color=RED, fill_opacity=0.8)
                for x, _ in cluster2_1d
            ]
        )
        cluster3_proj_x = VGroup(
            *[
                Dot(point=axes.c2p(x, 0), radius=0.05, color=GREEN, fill_opacity=0.8)
                for x, _ in cluster3_1d
            ]
        )

        y_axis = axes.get_y_axis()
        x_axis = axes.get_x_axis()

        new_x_label = (
            axes.get_x_axis_label("Principal Component")
            .set(font_size=18)
            .move_to(x_label)
        )

        self.play(
            FadeOut(y_axis, y_label),
            Transform(x_label, new_x_label),
            Transform(cluster1_dots, cluster1_proj_x),
            Transform(cluster2_dots, cluster2_proj_x),
            Transform(cluster3_dots, cluster3_proj_x),
            run_time=2,
        )

        self.wait(0.7)

        txt = Tex("How do we compute Principal Components ?")
        self.play(
            FadeOut(x_axis, x_label, cluster1_dots, cluster2_dots, cluster3_dots),
            run_time=2,
        )
        self.play(Write(txt), run_time=1)

        self.wait(2)


if __name__ == "__main__":
    scene = Scene1_2()
    scene.render()
