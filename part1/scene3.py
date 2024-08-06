from manim import *
import numpy as np
from sklearn.decomposition import PCA
from manim_voiceover import VoiceoverScene


class Scene1_3(VoiceoverScene, LinearTransformationScene):
    def __init__(self, **kwargs):
        super().__init__(
            include_background_plane=True,
            show_coordinates=True,
            show_basis_vectors=False,
            background_plane_kwargs={
                "x_range": [-2, 2, 0.2],
                "y_range": [-2, 2, 0.2],
                "x_length": config.frame_width * 1.8,
                "y_length": config.frame_width * 1.8,
            },
            foreground_plane_kwargs={
                "x_range": [-2, 2, 0.2],
                "y_range": [-2, 2, 0.2],
                "x_length": config.frame_width * 1.8,
                "y_length": config.frame_width * 1.8,
            },
            **kwargs
        )

    def construct(self):
        np.random.seed(42)

        # Create the dataset matrix X
        self.next_section(skip_animations=False)
        FRAME_WIDTH = config["frame_width"]
        FRAME_HEIGHT = config["frame_height"]

        self.plane.shift(0.1 * RIGHT + 2.5 * DOWN)
        self.background_plane.shift(0.1 * RIGHT + 2.5 * DOWN)

        bg_rectangle = Rectangle(
            width=FRAME_WIDTH / 2 - 0.5,
            height=FRAME_HEIGHT - 0.5,
            fill_opacity=1,
            fill_color=BLACK,
            stroke_width=2,
            stroke_color=WHITE,
        ).to_edge(LEFT, buff=0.25)

        self.add(bg_rectangle)

        self.wait(2)

        matrix = MathTex(
            r"X = \begin{bmatrix} x_1 & y_1 \\ x_2 & y_2 \\ x_3 & y_3 \\ ... & ... \end{bmatrix}"
        ).move_to(bg_rectangle.get_center())

        matrix_cp = matrix.copy()

        title = Tex("Dataset Matrix", font_size=52).move_to(
            bg_rectangle.get_top() + DOWN * 0.5
        )

        self.wait(0.7)

        # Define points
        mean = [170, 70]
        cov = [[50, 40], [40, 30]]  # diagonal covariance

        points = np.random.multivariate_normal(mean, cov, 300)

        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        points = (points - min_vals) / (max_vals - min_vals)

        # Add points to the scene as dots and label them
        dots = VGroup(
            *[
                Dot(
                    point=self.plane.c2p(point[0], point[1], 0),
                    radius=0.09,
                    color=GREEN,
                    fill_opacity=0.8,
                )
                for point in points
            ]
        )

        self.play(LaggedStartMap(Create, dots, lag_ratio=0.01), run_time=2)

        self.wait(0.6)
        self.add_transformable_mobject(
            dots
        )  # Ensure dots are affected by transformations

        self.play(Write(title))
        self.play(Create(matrix))

        self.wait(0.4)

        mean_x = MathTex(r"\bar{x} = \frac{x_1 + x_2 + x_3 + ...}{n}")
        mean_y = MathTex(r"\bar{y} = \frac{y_1 + y_2 + y_3 + ...}{n}")

        mean_x.move_to(bg_rectangle.get_center() + DOWN)
        mean_y.next_to(mean_x, DOWN, buff=0.5)

        self.play(
            matrix.animate.shift(1.5 * UP),
            Transform(matrix_cp[0][0], mean_x[0][3:5]),  # x_1
            Transform(matrix_cp[0][1], mean_y[0][3:5]),  # y_1
            Transform(matrix_cp[0][2], mean_x[0][6:8]),  # x_2
            Transform(matrix_cp[0][3], mean_y[0][6:8]),  # y_2
            Transform(matrix_cp[0][4], mean_x[0][9:11]),  # x_3
            Transform(matrix_cp[0][5], mean_y[0][9:11]),  # y_3
            run_time=1.5,
        )
        self.play(Write(mean_x), Write(mean_y))

        self.wait(1)

        self.play(
            FadeOut(mean_x),
            FadeOut(mean_y),
            FadeOut(
                matrix_cp[0][0],
                matrix_cp[0][1],
                matrix_cp[0][2],
                matrix_cp[0][3],
                matrix_cp[0][4],
                matrix_cp[0][5],
            ),
        )

        matrix_centered = MathTex(
            r"X = \begin{bmatrix} x_1 - \bar{x} & y_1 - \bar{y} \\ x_2 - \bar{x} & y_2 - \bar{y}\\ x_3 - \bar{x} & y_3 - \bar{y} \\ ... & ... \end{bmatrix}"
        ).move_to(bg_rectangle.get_center())

        title_centered = Tex("Centered Data", font_size=52).move_to(title)
        self.play(Transform(title, title_centered), Transform(matrix, matrix_centered))

        self.wait(1)

        self.wait(0.8)

        # Compute the covariance matrix
        self.next_section(skip_animations=False)

        title_cov = Tex("Covariance Matrix", font_size=52).move_to(title)

        cov_matrix = MathTex(r"C = \frac{1}{n-1}X^TX").move_to(
            bg_rectangle.get_center()
        )

        cov_matrix_eq = MathTex(
            r"C = \begin{bmatrix} \sigma_{x}^2 & \sigma_{xy} \\ \sigma_{yx} & \sigma_{y}^2 \end{bmatrix}"
        ).move_to(cov_matrix)

        self.play(FadeOut(matrix))
        self.play(Transform(title, title_cov), Write(cov_matrix), run_time=2)

        self.wait(0.8)

        self.play(Transform(cov_matrix, cov_matrix_eq), run_time=2)
        self.play(
            ShowPassingFlash(
                SurroundingRectangle(cov_matrix_eq, buff=0.1), time_width=0.3
            ),
            run_time=2,
        )
        self.wait(0.8)

        # Calculate the covariance matrix of the centered points
        centered_coords = np.array([dot.get_center()[:2] for dot in dots])
        covariance_matrix = np.cov(centered_coords.T)

        self.apply_matrix(covariance_matrix, run_time=2)
        self.wait(4)
        self.apply_inverse(covariance_matrix, run_time=2)

        self.wait(0.6)

        title_eigen = Tex("Eigenvectors", font_size=52).move_to(title)
        eigenvectors = MathTex(r"Cv = \lambda v").move_to(cov_matrix)

        eigenvectors_eq = MathTex(
            r"\begin{bmatrix} \sigma_{x}^2 & \sigma_{xy} \\ \sigma_{yx} & \sigma_{y}^2 \end{bmatrix} \begin{bmatrix} v_x \\ v_y \end{bmatrix} = \lambda \begin{bmatrix} v_x \\ v_y \end{bmatrix}"
        ).move_to(cov_matrix)
        #
        self.play(Transform(title, title_eigen), Transform(cov_matrix, eigenvectors))

        self.wait(0.4)

        # Project the data onto the principal components
        self.next_section(skip_animations=False)

        self.play(Transform(cov_matrix, eigenvectors_eq))

        self.wait(0.4)

        title_proj = Tex("Projection", font_size=52).move_to(title)
        pca_matrix = MathTex(r"P = \begin{bmatrix} v_x & v_y \end{bmatrix}").move_to(
            cov_matrix
        )

        self.play(Transform(title, title_proj), Transform(cov_matrix, pca_matrix))

        # Calculate the first principal component
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        first_pc = eigenvectors[:, np.argmax(eigenvalues)]

        # Create a projection matrix to the first principal component
        projection_matrix = np.outer(first_pc, first_pc)
        self.apply_matrix(projection_matrix, run_time=3)

        self.wait(2)


if __name__ == "__main__":
    scene = Scene1_3()
    scene.render()
