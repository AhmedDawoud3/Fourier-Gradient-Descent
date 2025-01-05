import math
import pickle as pkl

import numpy as np
from manim import *  # pylint: disable=W0614, W0401


def model(x, coefficients, L=math.pi):
    out = np.zeros_like(x)
    for j in range(coefficients.shape[0]):
        out += coefficients[j, 0] * np.sin(j * math.pi * x / L) + coefficients[
            j, 1
        ] * np.cos(j * math.pi * x / L)
    return out


def square_wave(x, L=3.14):
    return np.sign(np.sin(x * math.pi / L))


class FourierSquareWave(Scene):
    def construct(self):
        L = 3.14

        self.camera.background_color = GRAY_E  # type: ignore
        axes = Axes(
            x_range=[-math.pi * 2, math.pi * 2],
            y_range=[-5, 5],
            axis_config={
                "include_numbers": True,
                "color": WHITE,
            },
        )

        function_curve = axes.plot(
            square_wave,
            color=RED,
            discontinuities=[k * L for k in range(-4, 5)],
        )

        function_label = axes.get_graph_label(
            function_curve, label="Square Wave"
        ).shift(UP * 0.5)
        self.play(Create(axes), run_time=3)
        self.play(Create(function_curve), Write(function_label), run_time=2)  # type: ignore

        with open("sqr.pkl", "rb") as f:
            history = np.array(pkl.load(f))

        approximation_curve = axes.plot(lambda x: 0, color=BLUE)
        approximation_label = axes.get_graph_label(
            approximation_curve, label="Approximation"
        ).shift(DOWN)
        self.play(Create(approximation_curve), Write(approximation_label), run_time=2) # type: ignore
        self.wait(1)

        for i, coefficients in enumerate(history):
            if i % 10 != 0:
                continue
            if i > 100 and i % 100 != 0:
                continue
            if i > 1000 and i % 1000 != 0:
                continue

            new_curve = axes.plot(lambda x: model(x, coefficients, L), color=BLUE)
            self.play(
                Transform(approximation_curve, new_curve), run_time=1 if i < 50 else 0.5
            )

        self.play(
            Transform(
                approximation_curve,
                axes.plot(lambda x: model(x, history[-1], L), color=BLUE),
            ),
            run_time=0.1,
        )
        self.wait(1)
        self.play(ApplyWave(approximation_curve), run_time=3)
        self.wait(2)


class FourierEx2(Scene):
    def construct(self):
        L = 5

        # set background color
        self.camera.background_color = GRAY_E  # type: ignore
        axes = Axes(
            x_range=[-math.pi * 2, math.pi * 2],
            y_range=[-0.5, 1.5],
            axis_config={"include_numbers": True, "color": WHITE},
        )
        function_curve = axes.plot(
            lambda x: np.exp(-(x**2)),
            color=RED,
        )
        # Latex formula for the function
        function_label = axes.get_graph_label(
            function_curve, label=MathTex("e^{-x^2}")
        ).shift(UP * 0.5)
        self.play(Create(axes), run_time=3)
        self.play(Create(function_curve), Write(function_label), run_time=2)  # type: ignore

        with open("e.pkl", "rb") as f:
            history = np.array(pkl.load(f))

        approximation_curve = axes.plot(lambda x: 0, color=BLUE)
        approximation_label = axes.get_graph_label(
            approximation_curve, label="Approximation"
        ).shift(DOWN)
        self.play(Create(approximation_curve), Write(approximation_label), run_time=2) # type: ignore
        self.wait(1)
        for i, coefficients in enumerate(history):
            if i % 10 != 0:
                continue
            if i > 100 and i % 100 != 0:
                continue
            if i > 1000 and i % 1000 != 0:
                continue

            new_curve = axes.plot(lambda x: model(x, coefficients, L), color=BLUE)
            self.play(
                Transform(approximation_curve, new_curve), run_time=1 if i < 50 else 0.5
            )

        self.play(
            Transform(
                approximation_curve,
                axes.plot(lambda x: model(x, history[-1], L), color=BLUE),  # type: ignore
            ),
            run_time=0.1,
        )
        self.wait(1)
        self.play(ApplyWave(approximation_curve), run_time=3)
        self.wait(2)
