# Copyright 2024 CNRS

# Author: Florent Lamiraux

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import time
from typing import Callable, Self, Sequence, Any

import numpy as np
from math import atan2, cos, sin

from pinocchio import forwardKinematics
from talos import Robot
from scipy.optimize import fmin_slsqp
from cop_des import CoPDes
from walking_motion import WalkingMotion


class Bezier:
    """
    Bezier curve with any number of control points
    Evaluation is performed with de Casteljau algorithm.
    """

    def __init__(self, control_points: np.ndarray):
        self.control_points = control_points

    def __call__(self, t: float) -> np.ndarray:
        cp = self.control_points[:]
        while len(cp) > 1:
            cp1 = list()
            for p0, p1 in zip(cp, cp[1:]):
                cp1.append((1 - t) * p0 + t * p1)
            cp = cp1[:]
        return cp[0]

    def derivative(self) -> Self:
        """
        Return the derivative as a new Bezier curve
        """
        n = len(self.control_points) - 1
        cp = list()
        for p0, p1 in zip(self.control_points, self.control_points[1:]):
            cp.append(n * (p1 - p0))
        return Bezier(np.array(cp))


def simpson(f, t_init, t_end, n_intervals):
    """
    Computation of an integral with Simpson formula
    """
    l = (t_end - t_init) / n_intervals
    t0 = t_init
    res = f(t0) / 6
    for i in range(n_intervals):
        t1 = t0 + .5 * l
        t2 = t0 + l
        res += 2 / 3 * f(t1) + 1 / 3 * f(t2)
        t0 = t2
    res -= f(t_end) / 6
    res *= l
    return res


class Integrand:
    """
    Computes the integrand defining the integral cost for a given Bezier curve
    and a given parameter t as

         1     2           2
    I = --- (v   + alpha v  )
         2     T           N

    where
      - v  and v  are the tangent and normal velocities.
         T      N
    """
    alpha = 8

    def __init__(self, bezier: Bezier):
        self.function = bezier
        self.derivative = bezier.derivative()

    def __call__(self, t: float) -> float:
        b_t = self.function(t)
        db_t = self.derivative(t)
        theta = b_t[2]
        v_t = np.dot([np.cos(theta), np.sin(theta), 0], db_t)
        v_n = np.dot([-np.sin(theta), np.cos(theta), 0], db_t)
        return 0.5 * (v_t ** 2 + self.alpha * v_n ** 2)


class SlidingMotion:
    """
    Defines a sliding motion of the robot using Bezier curve and minimizing
    an integral cost favoring forward motions
    """
    beta = 100
    stepLength = .25

    def __init__(self, robot: Robot, q0: np.ndarray, end: np.ndarray):
        """ Constructor

        - input: q0 initial configuration of the robot,
        - end: end configuration specified as (x, y, theta) for the position
                and orientation in the plane.
        """
        self.robot = robot
        self.q0 = q0
        self.end = end
        self.control_points = np.linspace(q0[:3], end, 6)

    @staticmethod
    def cost(x: np.ndarray) -> float:
        """
        Compute the cost of a trajectory represented by a Bezier curve
        """
        assert (len(x.shape) == 1)
        bezier = Bezier(x.reshape(-1, 3))
        integrand = Integrand(bezier)
        return simpson(integrand, 0, 1, 100)

    def boundary_constraints(self, x: np.ndarray) -> list[float]:
        """
        Computes the scalar product of the x-y velocity at the beginning
        (resp. at the end) of the trajectory with the unit vector of initial
        (resp. end) orientation.
        """
        bezier = Bezier(x.reshape(-1, 3))
        db0 = bezier.derivative()(0)
        db1 = bezier.derivative()(1)
        theta0 = self.q0[2]
        theta1 = self.end[2]
        return [
            np.dot([np.cos(theta0), np.sin(theta0), 0], db0),
            np.dot([np.cos(theta1), np.sin(theta1), 0], db1)
        ]

    def solve(self) -> None:
        """
        Solve the optimization problem. Initialize with a straight line
        """
        X0 = self.control_points.flatten()
        result = fmin_slsqp(self.cost, X0, f_eqcons=self.boundary_constraints)
        self.control_points = result.reshape(-1, 3)

    @staticmethod
    def left_foot_pose(pose: np.ndarray) -> np.ndarray:
        res = np.zeros(3)
        res[:2] = pose[:2] + np.array([0.1, 0])
        res[2] = pose[2]
        return res

    @staticmethod
    def right_foot_pose(pose: np.ndarray) -> np.ndarray:
        res = np.zeros(3)
        res[:2] = pose[:2] + np.array([-0.1, 0])
        res[2] = pose[2]
        return res

    def compute_motion(self) -> list[np.ndarray]:
        configs = list()
        self.solve()
        bezier = Bezier(self.control_points)
        configs.append(self.q0)
        for t in np.linspace(0, 1, 100):
            pose = bezier(t)
            configs.append(np.hstack((pose, self.q0[3:])))
        return configs

    def sliding_path(self, t: float) -> np.ndarray:
        bezier = Bezier(self.control_points)
        return bezier(t)


if __name__ == '__main__':
    from talos import Robot

    robot = Robot()
    q0 = np.array([
        0.00000000e+00, 0.00000000e+00, 9.50023790e-01, 3.04115703e-04,
        0.00000000e+00, 0.00000000e+00, 9.99999957e-01, 0.00000000e+00,
        2.24440496e-02, -5.88127845e-01, 1.21572430e+00, -6.27580400e-01,
        -2.29184434e-02, 0.00000000e+00, -2.95804462e-02, -5.88175279e-01,
        1.21608861e+00, -6.27902977e-01, 2.91293666e-02, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 2.00000000e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, -2.00000000e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00])

    end = np.array([2, 1, 1.57])
    sm = SlidingMotion(robot, q0, end)
    configs = sm.compute_motion()
    for q in configs:
        time.sleep(1e-2)
        robot.display(q)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    times = 1e-2 * np.arange(101)
    X = np.array(list(map(sm.sliding_path, times)))
    ax1.plot(X[:, 0], X[:, 1], label="x-y path")
    ax2.plot(times, X[:, 2])
    plt.show()
