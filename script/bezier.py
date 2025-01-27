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
from typing import Self, MutableSequence

import numpy as np
from math import atan2, cos, sin

from jedi.inference.gradual.typing import Callable
from jedi.inference.value.iterable import Sequence
from numpy._typing import NDArray
from pinocchio import forwardKinematics
from talos import Robot
from scipy.optimize import fmin_slsqp
from cop_des import CoPDes
from walking_motion import WalkingMotion

Point = np.ndarray[(3,), float]


class Bezier:
    """
    Bezier curve with any number of control points
    Evaluation is performed with de Casteljau algorithm.
    """

    def __init__(self, control_points: list[Point]):
        self.controlPoints = control_points

    def __call__(self, t: float) -> Point:
        cp = self.controlPoints[:]
        while len(cp) > 1:
            cp1 = [(1 - t) * p0 + t * p1 for p0, p1 in zip(cp, cp[1:])]
            cp = cp1.copy()
        return cp[0]

    def derivative(self) -> Self:
        """
        Return the derivative as a new Bezier curve
        """
        n = len(self.controlPoints) - 1
        cp = [n * (P1 - P0) for P0, P1 in zip(self.controlPoints, self.controlPoints[1:])]
        return Bezier(cp)


def simpson(f: 'Integrand', t_init: float, t_end: float, n_intervals: int) -> float:
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
    alpha = 4

    def __init__(self, bezier: Bezier):
        self.bezier = bezier
        self.derivative = bezier.derivative()

    def __call__(self, t: float) -> float:
        b_t = self.bezier(t)
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

    def __init__(self, robot, q0, end):
        """ Constructor

        - input: q0 initial configuration of the robot,
        - end: end configuration specified as (x, y, theta) for the position
                and orientation in the plane.
        """
        self.robot = robot
        self.q0 = q0
        self.end = end

    def cost(self, X) -> float:
        """
        Compute the cost of a trajectory represented by a Bezier curve
        """
        assert (len(X.shape) == 1)
        integrand = Integrand(Bezier(X.reshape(-1, 3)))
        integral_cost = simpson(integrand, 0, 1, 100)

        theta0 = integrand.bezier(0)[2]
        theta1 = integrand.bezier(1)[2]

        db0 = integrand.derivative(0)
        db1 = integrand.derivative(1)

        boundary_cost = np.dot([-np.sin(theta0), np.cos(theta0), 0], db0) + \
                        np.dot([-np.sin(theta1), np.cos(theta1), 0], db1)

        total_cost = integral_cost + self.beta * boundary_cost
        return total_cost

    def boundaryConstraints(self, X):
        """
        Computes the scalar product of the x-y velocity at the beginning
        (resp. at the end) of the trajectory with the unit vector of initial
        (resp. end) orientation.
        """
        bezier = Bezier(X.reshape(-1, 3))
        derivative = bezier.derivative()

        # Initial orientation
        theta0 = bezier(0)[2]
        db0 = derivative(0)
        initial_velocity = np.dot([np.cos(theta0), np.sin(theta0), 0], db0)

        # Final orientation
        theta1 = bezier(1)[2]
        db1 = derivative(1)
        final_velocity = np.dot([np.cos(theta1), np.sin(theta1), 0], db1)

        return [initial_velocity, final_velocity]


    def solve(self):
        """
        Solve the optimization problem. Initialize with a straight line
        """

    def leftFootPose(self, pose):
        res = np.zeros(3)
        return res

    def rightFootPose(self, pose):
        res = np.zeros(3)
        return res

    def computeMotion(self):
        configs = list()
        return configs


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
    configs = sm.computeMotion()
    for q in configs:
        time.sleep(1e-2)
        robot.display(q)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    times = 1e-2 * np.arange(101)
    X = np.array(list(map(sm.slidingPath, times)))
    ax1.plot(X[:, 0], X[:, 1], label="x-y path")
    ax2.plot(times, X[:, 2])
    plt.show()
