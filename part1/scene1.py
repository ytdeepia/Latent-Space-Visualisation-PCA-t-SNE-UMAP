from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


class Scene1_1(VoiceoverScene):
    def construct(self):

        self.wait(2)

        np.random.seed(42)

        # Draw the intial distribution and normalize it
        self.next_section(skip_animations=False)

        text = Tex("Principal Component Analysis", font_size=60)
        self.play(Write(text))

        self.wait(0.7)

        self.play(FadeOut(text))

        table_data = [
            ["Alice", "170 cm", "68 kg"],
            ["Bob", "185 cm", "82 kg"],
            ["Charlie", "178 cm", "77 kg"],
            ["...", "...", "..."],
        ]

        # Create the table
        table = Table(
            table_data,
            col_labels=[Text("Name"), Text("Height (cm)"), Text("Weight (kg)")],
            include_outer_lines=True,
        )

        # Set the table position
        table.scale(0.5)

        self.play(FadeIn(table), run_time=2)

        self.wait(0.5)

        axes = Axes(
            x_range=[120, 220, 10],
            y_range=[30, 110, 10],
            x_length=6,
            y_length=6,
            axis_config={
                "color": WHITE,
                "include_numbers": True,
                "include_tip": False,
                "font_size": 18,
            },
        ).move_to(UP * 0.5)

        axes_x_label = (
            axes.get_x_axis_label("Height (cm)")
            .set(font_size=18)
            .move_to(axes.get_x_axis().get_right() + 0.5 * DOWN)
        )
        axes_y_label = (
            axes.get_y_axis_label("Weight (kg)")
            .set(font_size=18)
            .move_to(axes.get_y_axis().get_top() + 0.2 * UP)
        )

        # Generate the data points
        mean = [170, 70]
        cov = [[50, 40], [40, 30]]  # diagonal covariance

        data = np.random.multivariate_normal(mean, cov, 1000)

        scatter_plot = VGroup()
        for point in data:
            dot = Dot(
                axes.c2p(point[0], point[1]), color=BLUE, radius=0.03, fill_opacity=0.8
            )
            scatter_plot.add(dot)

        self.play(FadeOut(table), run_time=1)
        self.play(FadeIn(axes), Write(axes_x_label), Write(axes_y_label), run_time=2)
        self.play(LaggedStartMap(Create, scatter_plot, lag_ratio=0.01), run_time=2)

        self.wait(0.4)

        # Transform to normalized axes
        normalized_axes = Axes(
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

        # Normalize data
        normalized_data = (data - np.min(data, axis=0)) / (
            np.max(data, axis=0) - np.min(data, axis=0)
        )

        normalized_scatter_plot = VGroup()

        for point in normalized_data:
            dot = Dot(
                normalized_axes.c2p(point[0], point[1]),
                color=BLUE,
                radius=0.03,
                fill_opacity=0.8,
            )
            normalized_scatter_plot.add(dot)

        self.play(
            Transform(axes, normalized_axes),
            Transform(scatter_plot, normalized_scatter_plot),
            run_time=2,
        )

        self.wait(0.5)

        mean = np.mean(normalized_data, axis=0)
        cov = np.cov(normalized_data, rowvar=False)

        # Displaying the eigenvectors
        self.next_section(skip_animations=False)

        # Calculate eigenvalues and eigenvectors of the covariance matrix
        eigvals, eigvecs = np.linalg.eig(cov)

        # Scale eigenvectors
        scaled_eigvecs = eigvecs * np.sqrt(eigvals)

        # Create vectors for the eigenvectors starting from the mean
        vector1 = Arrow(
            start=normalized_axes.c2p(*mean),
            end=normalized_axes.c2p(*(mean + scaled_eigvecs[:, 0])),
            buff=0,
            color=RED,
            stroke_width=8,
            max_stroke_width_to_length_ratio=10,
            max_tip_length_to_length_ratio=0.7,
            z_index=1,
        )

        vector2 = Arrow(
            start=normalized_axes.c2p(*mean),
            end=normalized_axes.c2p(*(mean + 4 * scaled_eigvecs[:, 1])),
            buff=0,
            color=GREEN,
            stroke_width=8,
            max_stroke_width_to_length_ratio=10,
            max_tip_length_to_length_ratio=0.7,
            z_index=1,
        )

        self.wait(1)
        self.play(GrowArrow(vector1), run_time=1)
        self.play(GrowArrow(vector2), run_time=1)
        self.wait(0.5)
        self.play(Indicate(vector1, color=RED), run_time=2)
        self.play(Indicate(vector1, color=RED), run_time=2)

        self.wait(0.6)

        # Projecting along the eigenvectors
        self.next_section(skip_animations=False)

        self.play(FadeOut(vector2), run_time=0.6)

        # Calculate the projection onto the eigenvector with the largest eigenvalue
        max_eigval_index = np.argmax(eigvals)
        principal_eigvec = eigvecs[:, max_eigval_index]

        projections = [
            np.dot(principal_eigvec, np.array([x, y]) - mean) * principal_eigvec + mean
            for x, y in normalized_data
        ]

        # Create dots for the projections
        projection_dots = VGroup(
            *[
                Dot(point=normalized_axes.c2p(x, y), radius=0.05, color=BLUE)
                for x, y in projections
            ]
        )

        scatter_plot_original = scatter_plot.copy()

        self.wait(1)
        self.play(Transform(scatter_plot, projection_dots), run_time=3)
        self.play(FadeOut(vector1))

        self.wait(0.4)

        projection_dots = VGroup(
            *[
                Dot(point=normalized_axes.c2p(x, 0), radius=0.05, color=BLUE)
                for x, _ in projections
            ]
        )

        y_axis = axes.get_y_axis()
        x_axis = axes.get_x_axis()

        new_axes_x_label = (
            axes.get_x_axis_label("Principal Component")
            .set(font_size=18)
            .move_to(axes_x_label)
        )

        original_axes_x_label = axes_x_label.copy()

        self.play(
            FadeOut(y_axis, axes_y_label),
            Transform(scatter_plot, projection_dots),
            Transform(axes_x_label, new_axes_x_label),
            run_time=3,
        )

        self.wait(0.4)

        # Show projection on the horizontal axis alone
        self.next_section(skip_animations=False)

        self.wait(0.4)

        self.play(
            FadeIn(y_axis, axes_y_label),
            Transform(scatter_plot, scatter_plot_original),
            Transform(axes_x_label, original_axes_x_label),
            run_time=3,
        )

        projection_dots = VGroup(
            *[
                Dot(point=normalized_axes.c2p(x, 0), radius=0.05, color=BLUE)
                for x, _ in normalized_data
            ]
        )

        self.wait(1)

        self.play(Transform(scatter_plot, projection_dots), run_time=2)

        self.play(FadeOut(axes, scatter_plot, axes_x_label, y_axis, axes_y_label))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene1_1()
    scene.render()
