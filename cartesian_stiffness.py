import os
import casadi as ca

import pinocchio as pin
from pinocchio import visualize
from pinocchio import casadi as cpin

def load_robot():
    PKG_PATH = os.path.dirname(__file__)
    URDF_PATH = os.path.join(PKG_PATH, "double_link.urdf")

    robot = pin.RobotWrapper.BuildFromURDF(URDF_PATH, package_dirs = [PKG_PATH])
    robot.gravity = pin.Motion.Zero()

    return robot

def create_visualizer(robot):
    viz = visualize.MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)

    robot.setVisualizer(viz)
    robot.initViewer()
    robot.loadViewerModel()

    robot.display(pin.neutral(robot.model))

    return viz

def symbolic_fk(robot, frame_names):
    def _calc_fk(q):
        cmodel = cpin.Model(robot.model)
        cdata = cmodel.createData()
        
        cpin.forwardKinematics(cmodel, cdata, q)
        cpin.updateFramePlacements(cmodel, cdata)

        return ca.vertcat(*(
            cdata.oMf[cmodel.getFrameId(frame_name)].translation.T
            for frame_name in frame_names
        ))

    q = ca.SX.sym("q", robot.nq, 1)
    return ca.Function("fk", [q], [_calc_fk(q)])


if __name__ == "__main__":
    robot = load_robot()
    create_visualizer(robot)