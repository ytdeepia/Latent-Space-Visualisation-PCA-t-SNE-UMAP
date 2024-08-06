from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


class Scene3_4(VoiceoverScene):
    def construct(self):

        self.wait(2)

        # MNIST visualisation
        self.next_section(skip_animations=False)

        axes = Axes(
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
        ).scale(1.4)

        title = Tex("MNIST projection with UMAP").scale(0.9).to_edge(UP)
        title_ul = Underline(title)

        axes.next_to(title, direction=DOWN, buff=0.5)

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

        data = np.load("data/mnist_umap.npy", allow_pickle=True)
        coords = data[:, :2]
        labels = data[:, 2]

        dots = VGroup()
        for i, (x, y) in enumerate(coords):
            dot = Dot(
                axes.c2p(x, y),
                color=colors[int(labels[i])],
                radius=0.025,
                fill_opacity=0.6,
            )
            dots.add(dot)

        self.play(
            Write(title),
            GrowFromEdge(title_ul, LEFT),
            FadeIn(axes, colorbar, clabels),
            run_time=2,
        )

        self.play(FadeIn(dots))

        self.wait(0.8)

        time = Tex("Only 5 seconds ! ").scale(1.8).set_z_index(1)
        rect_bg = BackgroundRectangle(
            time, fill_opacity=0.8, buff=0.25, color=BLACK, z_index=0
        )

        self.play(FadeIn(rect_bg, time))

        self.wait(0.4)

        # Time comparison of the different methods
        self.next_section(skip_animations=False)

        self.play(
            FadeOut(time, rect_bg, axes, title, title_ul, dots, colorbar, clabels)
        )
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

        umap_times = [
            1.6609816551208496,
            3.59382963180542,
            6.724499464035034,
            9.919588088989258,
            5.05280613899231,
            6.045594215393066,
            6.93833327293396,
            7.948130130767822,
            8.86552357673645,
            10.15117073059082,
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
                    Dot(axes.c2p(0, 0), radius=0.1, color=BLUE),
                    Tex("PCA", color=BLUE),
                ).arrange(RIGHT, buff=0.2),
                VGroup(
                    Dot(axes.c2p(0, 0), radius=0.1, color=RED),
                    Tex("t-SNE", color=RED),
                ).arrange(RIGHT, buff=0.2),
                VGroup(
                    Dot(axes.c2p(0, 0), radius=0.1, color=GREEN),
                    Tex("UMAP", color=GREEN),
                ).arrange(RIGHT, buff=0.2),
            )
            .arrange(DOWN, buff=0.4)
            .next_to(axes, LEFT, buff=1.0)
        )

        dots = VGroup()
        for x, y in zip(samples, pca_times):
            dot = Dot(axes.c2p(x, y), radius=0.1, color=BLUE)
            dots.add(dot)

        for x, y in zip(samples, tsne_times):
            dot = Dot(axes.c2p(x, y), radius=0.1, color=RED)
            dots.add(dot)

        for x, y in zip(samples, umap_times):
            dot = Dot(axes.c2p(x, y), radius=0.1, color=GREEN)
            dots.add(dot)

        self.play(FadeIn(axes, x_label, y_label, legend))
        self.play(LaggedStartMap(Create, dots, run_time=2))

        self.play(FadeOut(axes, x_label, y_label, legend, dots))

        # Evolution of the number of neighbors
        self.next_section(skip_animations=False)

        txt = Tex("Number of nearest neighbors").scale(1.2)
        self.play(Write(txt))

        self.play(FadeOut(txt))

        title_proj = Tex("UMAP projection").scale(0.9).to_edge(UP).shift(3 * LEFT)
        title_proj_ul = Underline(title_proj)
        title_3d = Tex("Original 3D data").scale(0.9).to_edge(UP).shift(4 * RIGHT)
        title_3d_ul = Underline(title_3d)

        axes = (
            Axes(
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
            .scale(0.9)
            .next_to(title_proj, direction=DOWN, buff=0.5)
        )

        data_3 = np.load("data/mammoth_umap_3.npy", allow_pickle=True)
        data_3 = (data_3 - np.min(data_3, axis=0)) / (
            np.max(data_3, axis=0) - np.min(data_3, axis=0)
        )
        data_5 = np.load("data/mammoth_umap_5.npy", allow_pickle=True)
        data_5 = (data_5 - np.min(data_5, axis=0)) / (
            np.max(data_5, axis=0) - np.min(data_5, axis=0)
        )
        data_10 = np.load("data/mammoth_umap_10.npy", allow_pickle=True)
        data_10 = (data_10 - np.min(data_10, axis=0)) / (
            np.max(data_10, axis=0) - np.min(data_10, axis=0)
        )
        data_15 = np.load("data/mammoth_umap_15.npy", allow_pickle=True)
        data_15 = (data_15 - np.min(data_15, axis=0)) / (
            np.max(data_15, axis=0) - np.min(data_15, axis=0)
        )
        data_20 = np.load("data/mammoth_umap_20.npy", allow_pickle=True)
        data_20 = (data_20 - np.min(data_20, axis=0)) / (
            np.max(data_20, axis=0) - np.min(data_20, axis=0)
        )
        data_50 = np.load("data/mammoth_umap_50.npy", allow_pickle=True)
        data_50 = (data_50 - np.min(data_50, axis=0)) / (
            np.max(data_50, axis=0) - np.min(data_50, axis=0)
        )
        data_100 = np.load("data/mammoth_umap_100.npy", allow_pickle=True)
        data_100 = (data_100 - np.min(data_100, axis=0)) / (
            np.max(data_100, axis=0) - np.min(data_100, axis=0)
        )

        indices = np.random.choice(data_3.shape[0], 10000, replace=False)
        data_3 = data_3[indices]
        data_5 = data_5[indices]
        data_10 = data_10[indices]
        data_15 = data_15[indices]
        data_20 = data_20[indices]
        data_50 = data_50[indices]
        data_100 = data_100[indices]

        dots = VGroup()
        for i, (x, y) in enumerate(data_3):
            dot = Dot(
                axes.c2p(x, y),
                color=BLUE,
                radius=0.025,
                fill_opacity=0.6,
            )
            dots.add(dot)

        dots_5 = VGroup()
        for i, (x, y) in enumerate(data_5):
            dot = Dot(
                axes.c2p(x, y),
                color=BLUE,
                radius=0.025,
                fill_opacity=0.6,
            )
            dots_5.add(dot)

        dots_10 = VGroup()
        for i, (x, y) in enumerate(data_10):
            dot = Dot(
                axes.c2p(x, y),
                color=BLUE,
                radius=0.025,
                fill_opacity=0.6,
            )
            dots_10.add(dot)

        dots_15 = VGroup()
        for i, (x, y) in enumerate(data_15):
            dot = Dot(
                axes.c2p(x, y),
                color=BLUE,
                radius=0.025,
                fill_opacity=0.6,
            )
            dots_15.add(dot)

        dots_20 = VGroup()
        for i, (x, y) in enumerate(data_20):
            dot = Dot(
                axes.c2p(x, y),
                color=BLUE,
                radius=0.025,
                fill_opacity=0.6,
            )
            dots_20.add(dot)

        dots_50 = VGroup()
        for i, (x, y) in enumerate(data_50):
            dot = Dot(
                axes.c2p(x, y),
                color=BLUE,
                radius=0.025,
                fill_opacity=0.6,
            )
            dots_50.add(dot)

        dots_100 = VGroup()
        for i, (x, y) in enumerate(data_100):
            dot = Dot(
                axes.c2p(x, y),
                color=BLUE,
                radius=0.025,
                fill_opacity=0.6,
            )
            dots_100.add(dot)

        self.play(FadeIn(axes, title_proj, title_proj_ul, title_3d, title_3d_ul))
        self.play(FadeIn(dots))

        self.wait(0.7)

        title_proj_target = Tex("UMAP with 5 neighbors").move_to(title_proj)
        title_proj_ul_target = Underline(title_proj_target)

        self.play(
            Transform(dots, dots_5),
            Transform(title_proj, title_proj_target),
            Transform(title_proj_ul, title_proj_ul_target),
        )
        self.wait(1)
        title_proj_target = Tex("UMAP with 10 neighbors").move_to(title_proj)
        title_proj_ul_target = Underline(title_proj_target)

        self.play(
            Transform(dots, dots_10),
            Transform(title_proj, title_proj_target),
            Transform(title_proj_ul, title_proj_ul_target),
        )
        self.wait(1)
        title_proj_target = Tex("UMAP with 15 neighbors").move_to(title_proj)
        title_proj_ul_target = Underline(title_proj_target)

        self.play(
            Transform(dots, dots_15),
            Transform(title_proj, title_proj_target),
            Transform(title_proj_ul, title_proj_ul_target),
        )
        self.wait(1)

        self.wait(0.4)

        title_proj_target = Tex("UMAP with 20 neighbors").move_to(title_proj)
        title_proj_ul_target = Underline(title_proj_target)

        self.play(
            Transform(dots, dots_20),
            Transform(title_proj, title_proj_target),
            Transform(title_proj_ul, title_proj_ul_target),
        )
        self.wait(1)
        title_proj_target = Tex("UMAP with 50 neighbors").move_to(title_proj)
        title_proj_ul_target = Underline(title_proj_target)

        self.play(
            Transform(dots, dots_50),
            Transform(title_proj, title_proj_target),
            Transform(title_proj_ul, title_proj_ul_target),
        )
        self.wait(1)
        title_proj_target = Tex("UMAP with 100 neighbors").move_to(title_proj)
        title_proj_ul_target = Underline(title_proj_target)

        self.play(
            Transform(dots, dots_100),
            Transform(title_proj, title_proj_target),
            Transform(title_proj_ul, title_proj_ul_target),
        )

        self.wait(0.6)

        self.play(FadeOut(axes, title_proj, title_proj_ul, title_3d, title_3d_ul, dots))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene3_4()
    scene.render()
