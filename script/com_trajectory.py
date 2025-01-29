# Copyright 2023 CNRS

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

import numpy as np
from numpy.linalg import norm, pinv
from scipy.optimize import fmin_bfgs
from cop_des import CoPDes


# Computation of the trajectory of the center of mass by optimal control
class ComTrajectory(object):
    # time discretization step
    delta_t = 1e-2
    g = 9.81
    N = -1

    # Constructor
    #  - start, end projection in the horizontal plane of the initial and end position of
    #    the center of mass,
    #  - steps, list of 2D vectors representing the successive positions of the feet
    def __init__(self, start, steps, end, z_com):
        self.start = start
        self.end = end
        self.steps = steps
        self.z_com = z_com

    def compute(self):
        I2 = np.identity(2)
        n = len(self.steps)
        T = (1 + n) * CoPDes.double_support_time + n * CoPDes.single_support_time
        N = int(T / self.delta_t) + 1
        self.N = N
        # write your code here
        self.D = np.zeros(shape=(2 * N + 2, 2 * N))
        for i in range(N):
            self.D[2 * i:2 * i + 2, 2 * i:2 * i + 2] = I2
            self.D[2 * i + 2:2 * i + 4, 2 * i:2 * i + 2] = -I2

        self.d0 = np.zeros(2 * N + 2)
        self.d0[0:2] = -self.start
        self.d0[2 * N:2 * N + 2] = self.end

        I2N = np.identity(2 * N)
        self.cop_des = CoPDes(self.start, self.steps, self.end)

        # Compute CoP des
        COPdes = np.zeros(2 * N)
        for i in range(N):
            t = self.delta_t * i
            COPdes[2 * i:2 * i + 2] = self.cop_des(t)

        # Matrice A et vecteur b
        A = I2N + (self.z_com / (self.g * self.delta_t ** 2)) * np.dot(self.D.T, self.D)
        b = COPdes - (self.z_com / (self.g * self.delta_t ** 2)) * np.dot(self.D.T, self.d0)

        # Matrice C et vecteur d
        C = np.zeros(shape=(2 * N, 2 * N - 4))
        C[2:2 * N - 2] = np.identity(2 * N - 4)
        d = np.zeros(2 * N)
        d[0:2] = self.start
        d[2 * N - 2:2 * N] = self.end

        # Résolution du système
        Xbar = np.dot(np.linalg.pinv(np.dot(A, C)), (b - np.dot(A, d)))
        self.X = np.dot(C, Xbar) + d

    # Return projection of center of mass on horizontal plane at time t
    def __call__(self, t):
        if self.N < 0:
            raise RuntimeError("You should call method compute first.")
        i = (int)(t / self.delta_t + .5)
        res = np.zeros(3)
        res[2] = self.z_com
        if i <= 0:
            res[:2] = self.start
        elif i >= self.N + 1:
            res[:2] = self.end
        else:
            res[:2] = self.X[2 * i - 2:2 * i]
        return res


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    start = np.array([0., 0.])
    steps = [np.array([0, .2]), np.array([.4, -.2]), np.array([.8, .2]), np.array([1.2, -.2])]
    end = np.array([1.2, 0.])
    com_trajectory = ComTrajectory(start, steps, end, .7)
    X = com_trajectory.compute()
    times = 0.01 * np.arange(len(X) // 2)
    com = np.array(list(map(com_trajectory, times)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("time")
    ax.set_ylabel("m")
    ax.plot(times, com[:, 0], label="x_com")
    ax.plot(times, com[:, 1], label="y_com")
    ax.legend()
    plt.show()
    # com_trajectory.solve()
    # com = np.array(list(map(com_trajectory, times)))
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_xlabel("time")
    # ax.set_ylabel("m")
    # ax.plot(times, com[:,0], label="x_com")
    # ax.plot(times, com[:,1], label="y_com")
    # ax.legend()
    # plt.show()
