from manim import *
from manim_voiceover import VoiceoverScene
import numpy as np
import matplotlib.pyplot as plt


class Scene2_1(VoiceoverScene):
    def construct(self):
        self.wait(2)

        tsne = Tex("t-SNE")
        tsne_extended = Tex("t-Distributed Stochastic Neighbor Embedding")

        self.play(Write(tsne))

        self.wait(0.8)

        self.play(Transform(tsne, tsne_extended))
        self.wait(0.5)
        self.play(tsne.animate.shift(DOWN * 2))

        laurens = ImageMobject("images/laurens.png").scale_to_fit_height(2)
        hinton = ImageMobject("images/hinton.jpg").scale_to_fit_height(2)
        bengio = ImageMobject("images/bengio.jpg").scale_to_fit_height(2)
        lecun = ImageMobject("images/lecun.jpg").scale_to_fit_height(2)

        laurens.move_to(UP + LEFT * 3)
        hinton.move_to(UP + RIGHT * 3)

        laurens_txt = Tex("Laurens van der Maaten").next_to(laurens, UP).scale(0.8)
        laurens_underline = Underline(laurens_txt, buff=0.1)
        hinton_txt = Tex("Geoffrey Hinton").next_to(hinton, UP).scale(0.8)
        hinton_underline = Underline(hinton_txt, buff=0.1)

        self.play(FadeIn(laurens, laurens_txt, laurens_underline))
        self.play(FadeIn(hinton, hinton_txt, hinton_underline))

        self.wait(0.4)

        self.play(FadeOut(laurens, laurens_txt, laurens_underline, tsne))
        self.play(
            Group(hinton, hinton_txt, hinton_underline).animate.move_to(ORIGIN + UP)
        )

        godfathers_txt = Tex("The Godfathers of AI").move_to(DOWN * 2)
        self.play(Write(godfathers_txt))

        bengio.next_to(hinton, LEFT, buff=2)
        lecun.next_to(hinton, RIGHT, buff=2)
        bengio_txt = Tex("Yoshua Bengio").next_to(bengio, UP).scale(0.8)
        bengio_underline = Underline(bengio_txt, buff=0.1)
        lecun_txt = Tex("Yann LeCun").next_to(lecun, UP).scale(0.8)
        lecun_underline = Underline(lecun_txt, buff=0.1)

        self.play(FadeIn(bengio, bengio_txt, bengio_underline))
        self.play(FadeIn(lecun, lecun_txt, lecun_underline))

        self.wait(0.6)

        self.play(
            FadeOut(
                bengio,
                bengio_txt,
                lecun,
                lecun_txt,
                godfathers_txt,
                lecun_underline,
                bengio_underline,
            )
        )

        backprojection = Tex("Back-Propagation").move_to(4 * LEFT)
        dropout = Tex("Dropout").move_to(4 * RIGHT)
        autoencoders = Tex("Autoencoders").move_to(3 * LEFT + 1.5 * DOWN)
        distillation = Tex("Knowledge distillation").move_to(3 * RIGHT + 1.5 * DOWN)
        constrastive = Tex("Contrastive learning").move_to(3 * DOWN)

        self.play(GrowFromPoint(backprojection, ORIGIN))
        self.play(GrowFromPoint(dropout, ORIGIN))
        self.play(GrowFromPoint(autoencoders, ORIGIN))
        self.play(GrowFromPoint(distillation, ORIGIN))
        self.play(GrowFromPoint(constrastive, ORIGIN))

        self.wait(0.5)

        self.play(
            FadeOut(
                backprojection,
                dropout,
                autoencoders,
                distillation,
                constrastive,
                hinton,
                hinton_txt,
                hinton_underline,
            )
        )

        sne = Tex("Stochastic Neighbour Embedding")
        self.play(Write(sne))

        self.wait(0.4)

        self.play(FadeOut(sne))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene2_1()
    scene.render()
