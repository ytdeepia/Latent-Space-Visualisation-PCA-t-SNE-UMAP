from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler


def generate_spirals(n_points, radius=1, turns=3, noise=0.01):
    theta = np.linspace(0, turns * 2 * np.pi, n_points)
    z = np.linspace(0, 1, n_points)
    x1 = radius * np.cos(theta) + np.random.randn(n_points) * noise
    y1 = radius * np.sin(theta) + np.random.randn(n_points) * noise
    z1 = z + np.random.randn(n_points) * noise

    x2 = radius * np.cos(theta + np.pi) + np.random.randn(n_points) * noise
    y2 = radius * np.sin(theta + np.pi) + np.random.randn(n_points) * noise
    z2 = z + np.random.randn(n_points) * noise

    data = np.vstack((np.column_stack((x1, y1, z1)), np.column_stack((x2, y2, z2))))
    labels = np.hstack((np.zeros(n_points), np.ones(n_points)))

    return data, labels


class Scene2_6(VoiceoverScene):
    def construct(self):
        self.wait(2)

        # Non-linear dimensionality reduction
        self.next_section(skip_animations=False)

        txt = Tex("Non-linear data distributions").scale(1.5)

        self.play(Write(txt), run_time=2)

        self.play(FadeOut(txt), run_time=1)

        self.wait(2)

        txt_PCA = Tex("PCA").scale(1.2).shift(2.5 * UP)
        txt_SNE = Tex("SNE").scale(1.2)

        axes_pca = Axes(
            x_range=[0, 1, 0.5],
            y_range=[0, 1, 0.5],
            x_length=3.5,
            y_length=3.5,
            axis_config={
                "color": WHITE,
                "include_numbers": True,
                "include_tip": False,
                "font_size": 18,
            },
        ).next_to(txt_PCA, DOWN)

        axes_sne = Axes(
            x_range=[0, 1, 0.5],
            y_range=[0, 1, 0.5],
            x_length=3.5,
            y_length=3.5,
            axis_config={
                "color": WHITE,
                "include_numbers": True,
                "include_tip": False,
                "font_size": 18,
            },
        ).next_to(axes_pca, RIGHT)

        txt_SNE.next_to(axes_sne, UP)

        # Create the dataset
        n_points = 500
        data, labels = generate_spirals(n_points)

        colors = [BLUE, RED]

        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)
        data_pca = (data_pca - data_pca.min(axis=0)) / (
            data_pca.max(axis=0) - data_pca.min(axis=0)
        )

        dots_pca = VGroup()
        for i, (x, y) in enumerate(data_pca):
            dot = Dot(
                axes_pca.c2p(x, y),
                radius=0.04,
                color=colors[int(labels[i])],
                fill_opacity=0.8,
            )
            dots_pca.add(dot)

        tsne = TSNE(n_components=2, random_state=0, perplexity=100)
        data_tsne = tsne.fit_transform(data)
        data_tsne = (data_tsne - data_tsne.min(axis=0)) / (
            data_tsne.max(axis=0) - data_tsne.min(axis=0)
        )

        dots_sne = VGroup()
        for i, (x, y) in enumerate(data_tsne):
            dot = Dot(
                axes_sne.c2p(x, y),
                radius=0.04,
                color=colors[int(labels[i])],
                fill_opacity=0.8,
            )
            dots_sne.add(dot)

        ul_pca = Underline(txt_PCA, buff=0.1)
        ul_sne = Underline(txt_SNE, buff=0.1)

        self.play(FadeIn(txt_PCA, ul_pca, axes_pca, dots_pca))
        self.play(FadeIn(txt_SNE, ul_sne, axes_sne, dots_sne))

        self.wait(0.7)

        self.play(
            FadeOut(
                txt_PCA, ul_pca, axes_pca, dots_pca, txt_SNE, ul_sne, axes_sne, dots_sne
            ),
            run_time=1,
        )

        # t-sne changes
        self.next_section(skip_animations=False)

        txt = Tex("SNE").scale(1.5)
        txt_tsne = Tex("t-SNE").scale(1.2)

        self.play(Write(txt))
        self.play(Transform(txt, txt_tsne))
        self.play(Flash(txt, flash_radius=1.1, num_lines=16, line_length=0.6))

        self.play(FadeOut(txt))

        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[0, 0.5, 0.1],
            x_length=8,
            y_length=4,
            axis_config={
                "color": WHITE,
                "include_numbers": True,
                "include_tip": False,
                "font_size": 18,
            },
        ).scale(1.2)

        t_func = lambda x: stats.t.pdf(x, 1)
        gaussian = lambda x: stats.norm.pdf(x, 0, 1)
        graph_t = axes.plot(t_func, x_range=[-5, 5], color=RED)
        graph_gaussian = axes.plot(gaussian, x_range=[-5, 5], color=BLUE)

        label_1 = axes.get_graph_label(
            graph_t, r"t~distribution", x_val=-2, direction=UL
        )
        label_2 = axes.get_graph_label(
            graph_gaussian, r"Gaussian", x_val=1, direction=RIGHT
        )

        self.play(FadeIn(axes))
        self.play(Create(graph_gaussian), FadeIn(label_2))
        self.play(Create(graph_t), FadeIn(label_1))

        self.wait(0.6)

        self.play(FadeOut(axes, graph_t, label_1, graph_gaussian, label_2))

        # Complexity and time comparison
        self.next_section(skip_animations=False)

        pca_times = [
            0.03408479690551758,
            0.04994630813598633,
            0.09099149703979492,
            0.13801074028015137,
            0.15573668479919434,
            0.14432907104492188,
            0.1967780590057373,
            0.3617284297943115,
            0.3464477062225342,
            0.546889066696167,
        ]

        tsne_times = [
            5.03499174118042,
            12.883524656295776,
            20.816063404083252,
            29.46470046043396,
            39.37271237373352,
            48.95806956291199,
            59.08102488517761,
            68.41888761520386,
            77.59302854537964,
            87.58669137954712,
        ]

        samples = list(range(1000, 10001, 1000))

        axes = (
            Axes(
                x_range=[0, 10000, 1000],
                y_range=[0, 100, 10],
                x_length=6,
                y_length=6,
                axis_config={
                    "color": WHITE,
                    "include_numbers": True,
                    "include_tip": False,
                    "font_size": 22,
                },
            )
            .scale(0.95)
            .shift(0.7 * UP)
        )

        x_label = (
            axes.get_x_axis_label(r"Number~of~samples")
            .scale(0.6)
            .move_to(axes.x_axis.get_right() + 0.5 * DOWN)
        )
        y_label = (
            axes.get_y_axis_label(r"Time~(s)")
            .scale(0.6)
            .move_to(axes.y_axis.get_top() + 1.2 * LEFT)
        )

        legend = (
            VGroup(
                VGroup(
                    Dot(axes.c2p(0, 0), radius=0.1, color=BLUE), Tex("PCA", color=BLUE)
                ).arrange(RIGHT, buff=0.2),
                VGroup(
                    Dot(axes.c2p(0, 0), radius=0.1, color=RED), Tex("t-SNE", color=RED)
                ).arrange(RIGHT, buff=0.2),
            )
            .arrange(DOWN, buff=0.2)
            .next_to(axes, LEFT, buff=1.0)
        )

        dots = VGroup()
        for x, y in zip(samples, pca_times):
            dot = Dot(axes.c2p(x, y), radius=0.1, color=BLUE)
            dots.add(dot)

        for x, y in zip(samples, tsne_times):
            dot = Dot(axes.c2p(x, y), radius=0.1, color=RED)
            dots.add(dot)

        self.play(FadeIn(axes, x_label, y_label, legend))
        self.play(LaggedStartMap(Create, dots, run_time=2))
        self.wait(2)
        self.play(Indicate(dots[-1], color=RED, scale_factor=1.6), run_time=2)

        self.wait(0.8)

        self.play(FadeOut(axes, x_label, y_label, legend, dots))

        # Perplexity sensitivity
        self.next_section(skip_animations=False)

        axes = Axes(
            x_range=[0, 1, 0.1],
            y_range=[0, 1, 0.1],
            x_length=6,
            y_length=6,
            axis_config={
                "color": WHITE,
                "include_numbers": True,
                "include_tip": False,
                "font_size": 22,
            },
        )

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

        data30 = np.load("data/30.npy")
        tsne30 = data30[:, :2]
        labels = data30[:, 2]

        tsne30 = (tsne30 - tsne30.min(axis=0)) / (
            tsne30.max(axis=0) - tsne30.min(axis=0)
        )

        dots30 = VGroup()
        for i, (x, y) in enumerate(tsne30):
            dot = Dot(
                axes.c2p(x, y),
                color=colors[int(labels[i])],
                radius=0.025,
                fill_opacity=0.6,
            )
            dots30.add(dot)

        data31 = np.load("data/31.npy")
        tsne31 = data31[:, :2]
        labels = data31[:, 2]

        tsne31 = (tsne31 - tsne31.min(axis=0)) / (
            tsne31.max(axis=0) - tsne31.min(axis=0)
        )

        dots31 = VGroup()
        for i, (x, y) in enumerate(tsne31):
            dot = Dot(
                axes.c2p(x, y),
                color=colors[int(labels[i])],
                radius=0.025,
                fill_opacity=0.6,
            )
            dots31.add(dot)

        self.play(FadeIn(axes, colorbar, clabels))
        self.play(FadeIn(dots30))

        self.wait(0.6)

        self.play(Transform(dots30, dots31), run_time=3)

        self.wait(0.8)

        self.play(FadeOut(axes, colorbar, clabels, dots30))
        txt = Tex("Slow but better than PCA").scale(1.5)
        self.play(Write(txt))

        self.play(FadeOut(txt))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene2_6()
    scene.render()
