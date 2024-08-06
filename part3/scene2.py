from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


class Scene3_2(VoiceoverScene):
    def construct(self):
        self.wait(2)

        # Show a high-dimensional graph and a low-dimensional graph
        self.next_section(skip_animations=False)

        central_dot = Dot(ORIGIN, color=WHITE, radius=0.1)

        dot1 = Dot(np.array([0.6, 0.8, 0]), color=BLUE, radius=0.1)
        dot2 = Dot(np.array([0.9, -0.4, 0]), color=GREEN, radius=0.1)
        dot3 = Dot(np.array([-0.8, 0.7, 0]), color=RED, radius=0.1)
        dot4 = Dot(np.array([-0.9, -0.6, 0]), color=PURPLE, radius=0.1)

        arrow1 = DoubleArrow(central_dot, dot1, color=dot1.color, buff=0)
        arrow2 = DoubleArrow(central_dot, dot2, color=dot2.color, buff=0)
        arrow3 = DoubleArrow(central_dot, dot3, color=dot3.color, buff=0)
        arrow4 = DoubleArrow(central_dot, dot4, color=dot4.color, buff=0)

        self.play(FadeIn(central_dot, dot1, dot2, dot3, dot4))
        self.play(
            LaggedStart(
                Create(arrow1),
                Create(arrow2),
                Create(arrow3),
                Create(arrow4),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        hd_txt = Tex("High Dimensional Space").to_edge(LEFT, buff=1).shift(2.5 * UP)
        ld_txt = Tex("Low Dimensional Space").to_edge(RIGHT, buff=1).shift(2.5 * UP)

        nbline = NumberLine(
            x_range=[0, 1, 0.5],
            include_numbers=True,
            length=4,
            color=WHITE,
            include_tip=False,
            font_size=18,
        ).next_to(ld_txt, DOWN, buff=3.0)

        central_dot_ld = Dot(nbline.n2p(0.5), color=WHITE, radius=0.1)
        dot1_ld = Dot(nbline.n2p(0.6), color=dot1.color, radius=0.1)
        dot2_ld = Dot(nbline.n2p(0.9), color=dot2.color, radius=0.1)
        dot3_ld = Dot(nbline.n2p(0.1), color=dot3.color, radius=0.1)
        dot4_ld = Dot(nbline.n2p(0.3), color=dot4.color, radius=0.1)

        self.play(
            VGroup(
                central_dot, dot1, dot2, dot3, dot4, arrow1, arrow2, arrow3, arrow4
            ).animate.to_edge(LEFT, buff=2)
        )
        self.play(FadeIn(nbline, central_dot_ld, dot1_ld, dot2_ld, dot3_ld, dot4_ld))

        self.play(Write(hd_txt))
        self.play(Write(ld_txt))

        self.wait(0.7)

        self.play(
            FadeOut(
                nbline,
                central_dot_ld,
                dot1_ld,
                dot2_ld,
                dot3_ld,
                dot4_ld,
                central_dot,
                dot1,
                dot2,
                dot3,
                dot4,
                arrow1,
                arrow2,
                arrow3,
                arrow4,
            )
        )

        vertex_colors_hd = {
            "1": RED,
            "2": GREEN,
            "3": BLUE,
            "4": YELLOW,
            "5": PURPLE,
            "6": ORANGE,
            "7": PINK,
        }

        vertex_colors_ld = {
            "1": RED,
            "2": GREEN,
            "3": BLUE,
            "4": YELLOW,
            "5": PURPLE,
        }

        # Create the second random graph (directed)
        graph_hd = (
            Graph(
                ["1", "2", "3", "4", "5", "6", "7"],
                [
                    ("1", "2"),
                    ("2", "3"),
                    ("3", "1"),
                    ("3", "4"),
                    ("4", "5"),
                    ("5", "6"),
                    ("6", "7"),
                    ("7", "5"),
                ],
                layout="spring",
                labels=False,
                vertex_config={
                    vertex: {"radius": 0.15, "color": color}
                    for vertex, color in vertex_colors_hd.items()
                },
                edge_config={"color": WHITE, "stroke_width": 4},
                edge_type=Line,
            )
            .scale(0.9)
            .next_to(hd_txt, DOWN, buff=1.0)
        )

        graph_ld = (
            Graph(
                ["1", "2", "3", "4", "5"],
                [("1", "2"), ("2", "3"), ("3", "1"), ("3", "5"), ("5", "4")],
                layout="spring",
                labels=False,
                vertex_config={
                    vertex: {"radius": 0.15, "color": color}
                    for vertex, color in vertex_colors_ld.items()
                },
                edge_config={"color": WHITE, "stroke_width": 4},
                edge_type=Line,
            )
            .scale(0.9)
            .next_to(ld_txt, DOWN, buff=1.0)
        )

        self.play(Create(graph_hd), run_time=2)
        self.play(Create(graph_ld), run_time=2)

        self.play(FadeOut(hd_txt, ld_txt, graph_hd, graph_ld), run_time=0.9)

        txt = Tex("How do we compute these graphs ?")
        self.play(Write(txt))

        self.play(FadeOut(txt), run_time=1)

        # Find the nearest neighbors of a point
        self.next_section(skip_animations=False)

        txt = Tex("k-nearest neighbors").scale(1.5).to_edge(UP, buff=0.5)
        self.play(Write(txt))

        center_point = Dot(color=WHITE, radius=0.2).move_to(ORIGIN)

        points = [
            Dot(color=GREEN, radius=0.2).move_to([-0.8, 1.5, 0]),
            Dot(color=RED, radius=0.2).move_to([2.5, 1.3, 0]),
            Dot(color=PURPLE, radius=0.2).move_to([1.5, -2.5, 0]),
            Dot(color=BLUE, radius=0.2).move_to([3, -1, 0]),
            Dot(color=YELLOW, radius=0.2).move_to([-2.5, -2, 0]),
        ]
        distances = sorted([np.linalg.norm(point.get_center()) for point in points])
        circle = Circle(radius=0.01, color=WHITE)

        self.play(FadeIn(center_point, *points))
        self.play(Create(circle))

        self.wait(0.6)

        txt_target = Tex("3-nearest neighbors").scale(1.5).to_edge(UP)
        self.play(Transform(txt, txt_target))

        lines = VGroup()
        for i, dist in enumerate(distances[:3]):
            self.play(circle.animate.scale_to_fit_width(2 * dist), run_time=0.8)
            p = points[i]
            self.play(Indicate(p, scale_factor=1.5, color=p.color))
            line = Line(
                center_point.get_center(), p.get_center(), z_index=-1, color=WHITE
            )
            lines.add(line)
            self.play(Create(line))

            self.wait(0.4)

        self.wait(0.7)

        self.play(FadeOut(txt, circle), run_time=1)

        # Binary graph to weighted graph
        self.next_section(skip_animations=False)

        binary_graph = VGroup(lines, center_point, *points)
        self.play(
            binary_graph.animate.scale(0.6).to_edge(LEFT, buff=1.0),
        )

        distance_formula = (
            MathTex(r"v_{i,j} = e^{-\frac{max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}}")
            .scale(1.3)
            .to_edge(RIGHT, buff=1.6)
            .shift(1.6 * UP)
        )
        dist_txt = Tex("Weight computation").next_to(distance_formula, UP, buff=1)
        dist_txt_ul = Underline(dist_txt)

        self.play(Write(dist_txt), GrowFromEdge(dist_txt_ul, LEFT))

        self.wait(0.8)

        rect = SurroundingRectangle(distance_formula, color=WHITE, buff=0.25)
        self.play(Write(distance_formula), Create(rect))

        self.wait(0.4)

        rho_formula = (
            MathTex(r"\rho_i = \text{distance~to~nearest~neighbor}")
            .scale(1.2)
            .next_to(distance_formula, DOWN, buff=1.2)
        )
        rect2 = SurroundingRectangle(rho_formula, color=WHITE, buff=0.25)

        self.play(Write(rho_formula), Create(rect2))

        self.wait(0.8)

        label1 = Tex("1").scale(0.6).move_to(lines[0].get_center() + LEFT * 0.35)
        label2 = Tex("0.8").scale(0.6).move_to(lines[1].get_center() + UP * 0.35)
        label3 = Tex("0.3").scale(0.6).move_to(lines[2].get_center() + RIGHT * 0.35)

        self.play(GrowFromCenter(label1))
        self.play(GrowFromCenter(label2))
        self.play(GrowFromCenter(label3))

        self.play(
            FadeOut(dist_txt, dist_txt_ul, rect, rect2, distance_formula, rho_formula)
        )

        # Repeat for every point
        self.next_section(skip_animations=False)

        points2 = (
            VGroup(*points, central_dot).copy().next_to(binary_graph, RIGHT, buff=1.0)
        )
        points3 = VGroup(*points, central_dot).copy().next_to(points2, RIGHT, buff=1.0)

        self.play(GrowFromPoint(points2, binary_graph.get_center()), run_time=0.8)
        self.play(GrowFromPoint(points3, binary_graph.get_center()), run_time=0.8)

        lines2_1 = Line(
            points2[0].get_center(),
            points2[-1].get_center(),
            color=GREEN,
            z_index=-1,
        )
        label2_1 = (
            Tex("1", color=GREEN)
            .scale(0.6)
            .move_to(lines2_1.get_center() + (LEFT + DOWN) * 0.2)
        )
        lines2_2 = Line(
            points2[0].get_center(),
            points2[1].get_center(),
            color=GREEN,
            z_index=-1,
        )
        label2_2 = (
            Tex("0.77", color=GREEN)
            .scale(0.6)
            .move_to(lines2_2.get_center() + UP * 0.35)
        )
        lines2_3 = Line(
            points2[0].get_center(),
            points2[4].get_center(),
            color=GREEN,
            z_index=-1,
        )
        label2_3 = (
            Tex("0.25", color=GREEN)
            .scale(0.6)
            .move_to(lines2_3.get_center() + LEFT * 0.35)
        )

        self.play(
            LaggedStart(
                Create(lines2_1),
                Create(lines2_2),
                Create(lines2_3),
                AnimationGroup([FadeIn(label2_1), FadeIn(label2_2), FadeIn(label2_3)]),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        lines3_1 = Line(
            points3[1].get_center(), points3[-1].get_center(), color=RED, z_index=-1
        )
        label3_1 = (
            Tex("1", color=RED).scale(0.6).move_to(lines3_1.get_center() + DOWN * 0.35)
        )
        lines3_2 = Line(
            points3[1].get_center(), points3[3].get_center(), color=RED, z_index=-1
        )
        label3_2 = (
            Tex("0.6", color=RED)
            .scale(0.6)
            .move_to(lines3_2.get_center() + RIGHT * 0.35)
        )
        lines3_3 = Line(
            points3[1].get_center(), points3[0].get_center(), color=RED, z_index=-1
        )
        label3_3 = (
            Tex("0.5", color=RED).scale(0.6).move_to(lines3_3.get_center() + UP * 0.35)
        )

        self.play(
            LaggedStart(
                Create(lines3_1),
                Create(lines3_2),
                Create(lines3_3),
                AnimationGroup(
                    [FadeIn(label3_1), FadeIn(label3_2), FadeIn(label3_3)],
                ),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(0.6)

        # Symmetrize the graph
        self.next_section(skip_animations=False)
        txt = Tex("How do we combine these graphs ?").to_edge(UP, buff=0.5)
        txt_ul = Underline(txt)
        self.play(Write(txt), GrowFromEdge(txt_ul, LEFT))

        self.play(
            FadeOut(
                points2, lines2_1, lines2_2, lines2_3, label2_1, label2_2, label2_3
            ),
            run_time=0.7,
        )

        self.wait()
        symm_formula = MathTex(r"w_{ij} = v_{i,j} + v_{j, i} - v_{i,j}.v_{j,i}").scale(
            0.9
        )
        rect = SurroundingRectangle(symm_formula, color=WHITE, buff=0.25)

        self.play(Write(symm_formula), Create(rect))

        self.wait(0.5)

        self.play(Indicate(VGroup(symm_formula, rect), color=WHITE, scale_factor=1.2))

        self.play(VGroup(symm_formula, rect).animate.shift(2 * UP), run_time=0.8)

        self.play(
            Wiggle(lines3_1, color=RED, scale_value=1.5),
            Wiggle(lines[1], color=WHITE, scale_value=1.5),
        )

        rect2 = SurroundingRectangle(label2, color=WHITE, buff=0.25)
        self.play(ShowPassingFlash(rect2, time_width=0.4, run_time=1.5))

        rect3_1 = SurroundingRectangle(label3_1, color=WHITE, buff=0.25)
        self.play(ShowPassingFlash(rect3_1, time_width=0.4, run_time=1.5))

        symm_formula_digits = MathTex(
            "w_{ij} = ", "1", "+", "0.8", "-", "1", r"\times", "0.8"
        ).scale(0.9)

        symm_formula_digits[1].set_color(RED)
        symm_formula_digits[5].set_color(RED)

        self.play(Write(symm_formula_digits))

        self.play(
            FadeOut(
                rect,
                symm_formula,
                symm_formula_digits,
                *points3,
                lines3_1,
                lines3_2,
                lines3_3,
                label3_1,
                label3_2,
                label3_3,
                label1,
                label2,
                label3,
                binary_graph,
                txt,
                txt_ul
            )
        )

        # final graph representations
        self.next_section(skip_animations=False)

        vertices = [
            np.array([0, 0, 0]),
            np.array([0.6, 0.8, 0]),
            np.array([0.9, -0.4, 0]),
            np.array([-0.8, 0.7, 0]),
            np.array([-0.9, -0.6, 0]),
        ]
        vertices = [2 * vertex for vertex in vertices]

        # Define the edges (pairs of vertex indices)
        edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4)]

        # Create the graph
        hd_graph = VGroup()

        # Create the vertices
        for vertex in vertices:
            dot = Dot(point=vertex, radius=0.1)
            hd_graph.add(dot)

        # Create the edges and labels
        lines = VGroup()
        for edge in edges:
            start, end = edge
            line = Line(vertices[start], vertices[end])
            hd_graph.add(line)
            lines.add(line)

        label1 = (
            Tex("0.8").scale(0.7).move_to(lines[0].get_center() + 0.3 * (LEFT + UP))
        )
        label2 = Tex("0.5").scale(0.7).move_to(lines[1].get_center() + 0.4 * RIGHT)
        label3 = Tex("0.9").scale(0.7).move_to(lines[2].get_center() + 0.4 * DOWN)
        label4 = (
            Tex("0.8").scale(0.7).move_to(lines[3].get_center() + 0.3 * (LEFT + DOWN))
        )
        label5 = Tex("0.6").scale(0.7).move_to(lines[4].get_center() + 0.4 * LEFT)

        txt_hd_graph = (
            Tex("High Dimensional Representation").scale(0.7).to_edge(UP, buff=0.5)
        )
        txt_hd_graph_ul = Underline(txt_hd_graph)

        self.play(
            FadeIn(hd_graph, label1, label2, label3, label4, label5),
            Write(txt_hd_graph),
            GrowFromEdge(txt_hd_graph_ul, LEFT),
        )

        self.wait(2)

        # Define the vertices
        vertices = [0, 1, 2, 3, 4, 5]

        # Define the edges
        edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]

        # Create the graph
        ld_graph = (
            Graph(
                vertices,
                edges,
                layout="spring",
                layout_scale=3,
                vertex_config={"color": WHITE},
            )
            .scale(0.8)
            .to_edge(RIGHT, buff=1.0)
        )

        ld_graph_txt = (
            Tex("Low Dimensional Representation")
            .scale(0.7)
            .next_to(graph_ld, UP)
            .to_edge(UP, buff=0.5)
        )
        ld_graph_txt_ul = Underline(ld_graph_txt)

        self.play(
            FadeOut(label1, label2, label3, label4, label5),
            VGroup(hd_graph, txt_hd_graph, txt_hd_graph_ul).animate.to_edge(
                LEFT, buff=1.0
            ),
        )

        self.play(
            FadeIn(ld_graph),
            Write(ld_graph_txt),
            GrowFromEdge(ld_graph_txt_ul, LEFT),
        )

        self.wait(0.9)

        txt = Tex("What's next?").set_z_index(1).scale(2.5)
        rect_txt = BackgroundRectangle(
            txt, color=BLACK, fill_opacity=0.9, z_index=0, buff=0.8
        )

        self.play(FadeIn(rect_txt), Write(txt), run_time=2)

        self.wait(0.9)

        self.play(
            FadeOut(
                ld_graph,
                ld_graph_txt,
                ld_graph_txt_ul,
                hd_graph,
                txt_hd_graph,
                txt_hd_graph_ul,
                rect_txt,
                txt,
            )
        )

        self.wait(2)


if __name__ == "__main__":
    scene = Scene3_2()
    scene.render()
