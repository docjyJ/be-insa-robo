# Copyright 2018 CNRS

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
import numpy as np
from math import atan2, cos, sin
from pinocchio import centerOfMass, forwardKinematics
from cop_des import CoPDes
from com_trajectory import ComTrajectory
from inverse_kinematics import InverseKinematics
from tools import Constant, Piecewise

# Computes the trajectory of a swing foot.
#
# Input data are
#  - initial and final time of the trajectory,
#  - initial and final pose of the foot,
#  - maximal height of the foot,
#
# The trajectory is polynomial with zero velocities at start and end.
# The orientation of the foot is kept as in intial pose.
class SwingFootTrajectory(object):
    def __init__(self, t_init, t_end, init, end, height):
        assert(init[2] == end[2])
        self.t_init = t_init
        self.t_end = t_end
        self.height = height
        # Write your code here
        self.a3 = -2*(end-init)/(t_end-t_init)**3
        self.a2 =  3*(end-init)/(t_end-t_init)**2
        self.a1 = 0
        self.a0 = init

    def __call__(self, t):
        # write your code here
        t1 = t-self.t_init
        res = self.a0 + t1*self.a1 + t1**2*self.a2 + t1**3*self.a3
        res[2] += t1**2*(self.t_end-t)**2*16*self.height/((self.t_end-self.t_init)**4)
        return res

def Rz(theta):
    """
    Return rotation matrix of angle theta around z-axis
    """
    res = np.identity(3)
    res[0,0] = cos(theta); res[0,1] = -sin(theta);
    res[1,0] = sin(theta); res[1,1] =  cos(theta);
    return res

# Computes a walking whole-body motion
#
# Input data are
#  - an initial configuration of the robot,
#  - a sequence of step positions (x,y,theta) on the ground,
#  - a mapping from time to R corresponding to the desired orientation of the
#    waist. If not provided, keep constant orientation.
#
class WalkingMotion(object):
    step_height = 0.05

    def __init__(self, robot):
        self.robot = robot

    def compute(self, q0, steps, waistOrientation = None):
        # Test input data
        if len(steps) < 4:
            raise RuntimeError("sequence of step should be of length at least 4 instead of " +
                               f"{len(steps)}")
        # Copy steps in order to avoid modifying the input list.
        steps_ = steps[:]
        # Compute offset between waist and center of mass since we control the center of mass
        # indirectly by controlling the waist.
        data = self.robot.model.createData()
        forwardKinematics(self.robot.model, data, q0)
        com = centerOfMass(self.robot.model, data, q0)
        waist_pose = data.oMi[self.robot.waistJointId]
        com_offset = waist_pose.translation - com
        # Trajectory of left and right feet
        self.lf_traj = Piecewise()
        self.rf_traj = Piecewise()
        # write your code here
        # Compute initial position of feet
        lf_init = data.oMi[self.robot.leftFootJointId].translation
        Rlf_init = data.oMi[self.robot.leftFootJointId].rotation
        rf_init = data.oMi[self.robot.rightFootJointId].translation
        Rrf_init = data.oMi[self.robot.rightFootJointId].rotation
        # Compute final position of center of mass: between the feet in final pose
        end = .5*(steps_[-1] + steps_[-2])
        # Compute desired trajectory of center of pressure
        # CoP des first goes under the left foot. We add the position of the
        # left foot as the first step.
        self.com_trajectory = ComTrajectory(com[:2],
                                            list(map(lambda v:v[:2], steps_)),
                                            end[:2], com[2])
        self.com_trajectory.compute()
        t_ss = CoPDes.single_support_time
        t_ds = CoPDes.double_support_time
        # Keep the feet fixed while CoP moves to first support foot
        lf_pose = np.zeros(4)
        lf_pose[:3] = lf_init
        lf_pose[3] = atan2(Rlf_init[1,0], Rlf_init[0,0])
        rf_pose = np.zeros(4)
        rf_pose[:3] = rf_init
        rf_pose[3] = atan2(Rrf_init[1,0], Rrf_init[0,0])
        self.lf_traj.segments.append(Constant(0, t_ds, lf_pose))
        self.rf_traj.segments.append(Constant(0, t_ds, rf_pose))
        t = t_ds
        swing_foot = "left"
        # Iteratively move right then left foot
        for step in steps_[1:]:
            final_swing_foot = np.zeros(4)
            final_swing_foot[:2] = step[:2]
            # store orientation of foot in 4-th component
            final_swing_foot[3] = step[2]
            if swing_foot == "right":
                final_swing_foot[2] = rf_pose[2]
                self.lf_traj.segments.append(Constant(t, t+t_ss, lf_pose.copy()))
                self.rf_traj.segments.append(SwingFootTrajectory(
                    t, t+t_ss, rf_pose, final_swing_foot, self.step_height))
                rf_pose = final_swing_foot
                t += t_ss
                swing_foot = "left"
            else:
                final_swing_foot[2] = lf_pose[2]
                self.rf_traj.segments.append(Constant(t, t+t_ss, rf_pose.copy()))
                self.lf_traj.segments.append(SwingFootTrajectory(
                    t, t+t_ss, lf_pose, final_swing_foot, self.step_height))
                lf_pose = final_swing_foot
                t += t_ss
                swing_foot = "right"
            #keep feet static for double support time
            self.rf_traj.segments.append(Constant(t, t+t_ds, rf_pose.copy()))
            self.lf_traj.segments.append(Constant(t, t+t_ds, lf_pose.copy()))
            t += t_ds
        # Compute whole body trajectory
        ik = InverseKinematics(self.robot)
        configs = list()
        q = q0.copy()
        for i in range(self.com_trajectory.N+1):
            t = self.com_trajectory.delta_t * i
            footPose = self.lf_traj(t)
            ik.leftFootRefPose.translation = footPose[:3]
            ik.leftFootRefPose.rotation = Rz(footPose[3])
            footPose = self.rf_traj(t)
            ik.rightFootRefPose.translation = footPose[:3]
            ik.rightFootRefPose.rotation = Rz(footPose[3])
            ik.waistRefPose.translation = self.com_trajectory(t) + com_offset
            if not waistOrientation is None:
                theta = waistOrientation(t)
                ik.waistRefPose.rotation[0,0] = cos(theta)
                ik.waistRefPose.rotation[1,0] = sin(theta)
                ik.waistRefPose.rotation[0,1] = -sin(theta)
                ik.waistRefPose.rotation[1,1] = cos(theta)
            q = ik.solve(q)
            configs.append(q)
        return configs

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from talos import Robot
    from pinocchio import neutral
    import numpy as np
    from inverse_kinematics import InverseKinematics
    import eigenpy

    robot = Robot ()
    ik = InverseKinematics (robot)
    ik.rightFootRefPose.translation = np.array ([0, -0.1, 0.1])
    ik.leftFootRefPose.translation = np.array ([0, 0.1, 0.1])
    ik.waistRefPose.translation = np.array ([0, 0, 0.95])

    q0 = neutral (robot.model)
    q0 [robot.name_to_config_index["leg_right_4_joint"]] = .2
    q0 [robot.name_to_config_index["leg_left_4_joint"]] = .2
    q0 [robot.name_to_config_index["arm_left_2_joint"]] = .2
    q0 [robot.name_to_config_index["arm_right_2_joint"]] = -.2
    q = ik.solve (q0)
    robot.display(q)
    wm = WalkingMotion(robot)
    # First two values correspond to initial position of feet
    # Last two values correspond to final position of feet
    steps = [np.array([0, -.1, 0.]), np.array([0.4, .1, 0.]),
             np.array([.8, -.1, 0.]), np.array([1.2, .1, 0.]),
             np.array([1.6, -.1, 0.]), np.array([1.6, .1, 0.])]
    configs = wm.compute(q, steps)
    for q in configs:
        time.sleep(1e-2)
        robot.display(q)
    delta_t = wm.com_trajectory.delta_t
    times = delta_t*np.arange(wm.com_trajectory.N+1)
    lf = np.array(list(map(wm.lf_traj, times)))
    rf = np.array(list(map(wm.rf_traj, times)))
    cop_des = np.array(list(map(wm.com_trajectory.cop_des, times)))
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.plot(times, lf[:,0], label="x left foot")
    ax1.plot(times, rf[:,0], label="x right foot")
    ax1.plot(times, cop_des[:,0], label="x CoPdes")
    ax1.legend()
    ax2.plot(times, lf[:,1], label="y left foot")
    ax2.plot(times, rf[:,1], label="y right foot")
    ax2.plot(times, cop_des[:,1], label="y CoPdes")
    ax2.legend()
    ax3.plot(times, lf[:,2], label="z left foot")
    ax3.plot(times, rf[:,2], label="z right foot")
    ax3.legend()
    plt.show()

