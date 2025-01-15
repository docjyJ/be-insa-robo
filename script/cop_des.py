# Copyright 2023 CNRS
from numpy import ndarray

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

from tools import Affine, Constant, Piecewise

# Compute a desired trajectory for the center of pressure, piecewise affine linking the
# input step positions in the plane
class CoPDes(Piecewise):
    single_support_time = .2
    double_support_time = .07

    def __init__(self, start, steps, end):
        super().__init__()
        t, step = 0, start
        t, step = self.append_edge_step(t, start, steps[0])
        for new_step in steps[1:]:
            t, step = self.append_step(t, step, new_step)
        t, step = self.append_edge_step(t, step, end)
        self.segments.append(Constant(t, t+2, end))

    def append_edge_step(self, t0, start, end):
        t1 = t0 + self.double_support_time
        self.segments.append(Affine(t0, t1, start, end))
        return t1, end

    def append_step(self, t0, start, end):
        t1 = t0 + self.single_support_time
        t2 = t1 + self.double_support_time
        self.segments.append(Constant(t0, t1, start))
        self.segments.append(Affine(t1, t2, start, end))
        return t2, end

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    start = np.array([0.,0.])
    steps = [np.array([0, -.1]), np.array([0.4, .1]),
             np.array([.8, -.1]), np.array([1.2, .1]),
             np.array([1.6, -.1]), np.array([1.6, .1])]
    end = np.array([1.6,0.])
    cop_des = CoPDes(start, steps, end)
    times = 0.01 * np.arange(500)
    cop = np.array(list(map(cop_des, times)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("second")
    ax.set_ylabel("meter")
    ax.plot(times, cop[:,0], label="x_cop")
    ax.plot(times, cop[:,1], label="y_cop")
    ax.legend()
    plt.show()
