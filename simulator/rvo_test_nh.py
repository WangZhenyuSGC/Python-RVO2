import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Continuous-CBS")
from orca_simulator_base import OrcaSimulator
import numpy as np
import math
import random
import CCBS
import xml.etree.ElementTree as ET
from xml.dom import minidom

def circle(device_index, center_x, center_y, circle_radius, max_device_num):
    target_x = circle_radius * np.cos(2 * np.pi * device_index / max_device_num) + center_x
    target_y = circle_radius * np.sin(2 * np.pi * device_index / max_device_num) + center_y

    return target_x, target_y

class NHRVOSimulator(OrcaSimulator):
    def __init__(self, config_file_name = 'rvo_config_NH.yaml', robot_number = 5):
        super().__init__(config_file_name, robot_number)      
        self.set_pos()
        self.agents_init()
    
    def parameter_init(self, robot_number):
        self.minErrorHolo = self.config["minErrorHolo"] * 1000.0
        self.maxErrorHolo = self.config["maxErrorHolo"] * 1000.0
        self.velMaxW = self.config["velMaxW"] * 1000.0
        self.wMax = self.config["wMax"]
        self.curAllowedError = self.config["curAllowedError"] * 1000.0
        self.timeToHolo = self.config["timeToHolo"]
        self.last_twist_ang = 0.0
        super().parameter_init(robot_number)

    def agents_init(self):
        self.agent_ids = {} # RVO agents
        for i, robot_id in enumerate(self.robots):
            target_id = f"target{(i % len(self.targets)) + 1}"
            self.robots[robot_id]["goal"] = self.targets[target_id]["pose"]
            # 初期位置を
            agent_id = self.add_agent((self.robots[robot_id]["pose"]["x"],self.robots[robot_id]["pose"]["y"]))
            self.agent_ids[robot_id] = agent_id

            # Non-holonomic関連のパラメーターを設定
            self.sim.setHoloParams(agent_id, self.minErrorHolo, self.maxErrorHolo, self.velMaxW, self.wMax, self.curAllowedError, self.timeToHolo, self.L)
    
    def update_status_enlarge(self, robot_id, robot):
        target = robot["goal"]

        vector_to_goal = np.array(
            [
                target["x"] - robot["pose"]["x"], 
                target["y"] - robot["pose"]["y"]
            ]
        )

        pref_vel = tuple(vector_to_goal)
        distance_to_goal = np.linalg.norm(vector_to_goal)    
        goal_flag = False

        agent_id = self.agent_ids[robot_id]
        self.distances[agent_id] = {"distance": distance_to_goal}
        
        if robot_id in self.fallen_agents:         
            self.sim.setAgentMaxSpeed(agent_id, 0)
            self.sim.setAgentCollabCoeff(agent_id, 0.0)
            self.sim.setAgentPosition(agent_id, (robot["pose"]["x"], robot["pose"]["y"]))
            self.sim.setAgentVelocity(agent_id, (0.0,0.0))
            self.sim.setAgentPrefVelocity(agent_id, (0, 0))
            return
        else:
            self.sim.setAgentMaxSpeed(agent_id, self.lv_limit)
            self.sim.setAgentCollabCoeff(agent_id, 0.5)
    
        # 目標点の一定距離内に入ったら理想速度を調整する
        # 正式のGOAL判定は有効中心ではなく実際の中心で行う
        goal_flag = distance_to_goal <= self.pos_threshold  

        current_pose = tuple((robot["pose"]["x"], robot["pose"]["y"], robot["pose"]["theta"]))
        current_vel = tuple((robot["velocity"]["v"], robot["velocity"]["w"]))   

        agent_id = self.agent_ids[robot_id]

        # RVO2に現在の位置と速度を伝える
        self.sim.setAgentPosition(agent_id, tuple((current_pose[0], current_pose[1])))
        self.sim.setAgentVelocity(agent_id, tuple((current_vel[0] * math.cos(current_pose[2]), current_vel[0] * math.sin(current_pose[2]))))

        # RVO2に目標位置の方向（理想速度）を伝える
        self.sim.setAgentPrefVelocity(agent_id, tuple((pref_vel[0], pref_vel[1])))

        # 追加: 向きと角速度も伝える
        self.sim.setAgentAngularInfo(agent_id, current_pose[2], current_vel[1])
        robot["goal_flag"] = goal_flag

    def enlarge_twist(self, agent_id, robot):
        target_v = self.sim.getAgentVelocity(agent_id)
        target_speed = np.linalg.norm(target_v)
        target_ang = math.atan2(target_v[1], target_v[0])
        dif_ang = self.normalize_angle(target_ang) - self.normalize_angle(robot["pose"]["theta"])
        dif_ang = self.normalize_angle(dif_ang)

        # print("Planned vel, dif_ang:", agent_id, target_v, dif_ang)

        if abs(dif_ang) >= self.ang_threshold:
            vstar = calcVstar(target_speed, dif_ang)
        else:
            vstar = target_speed

        lv = max(min(vstar, self.lv_limit), -self.lv_limit) 
        
        if (abs(dif_ang) > 3.0 * np.pi / 4.0):
            if self.last_twist_ang != 0.0:
                w = np.sign(self.last_twist_ang) * min(abs(dif_ang / self.timeToHolo), self.wMax)
            else:
                w = np.sign(dif_ang) * min(abs(dif_ang / self.timeToHolo), self.wMax)
            if abs(dif_ang) > np.pi:
                lv = 0
            self.last_twist_ang = w
        else:
            w = np.sign(dif_ang) * min(abs(dif_ang / self.timeToHolo), self.wMax)
            self.last_twist_ang = 0.0
        # print('--------------------------robot id, v, w before clip', (agent_id, lv, w))
        return lv, w

    # def set_pos(self):
    #     self.targets = {}
    #     self.robots = {}
    #     for i in range(self.robot_number):
    #         target_id = f"target{i+1}"
    #         circle_radius = 200
    #         center_x = 400
    #         center_y = 700
    #         target_x, target_y = circle(i, center_x, center_y, circle_radius, self.robot_number) 
    #         target_pose = {"x": target_x, "y": target_y, "theta": random.uniform(-np.pi, np.pi)} 
    #         self.targets[target_id] = {
    #             "id": target_id,
    #             "pose": target_pose
    #         }

    #     for i in range(self.robot_number):
    #         robot_id = f"robot{i+1}"            
    #         robot_pose = {"x": 600, "y": 720 + (-1) ** i * 250 * int((i + 1) / 2), "theta": random.uniform(-np.pi, np.pi)}
    #         self.robots[robot_id] = {
    #             "id": robot_id,
    #             "pose": robot_pose,
    #             "goal": None,
    #             "velocity": {"v": 0.0, "w": 0.0},
    #             "pos_flag": False, 
    #             "goal_flag": False
    #         }

def calcVstar(vh, theta):
    return vh * ((theta * math.sin(theta)) / (2.0 * (1.0 - math.cos(theta))))

simulation = NHRVOSimulator('rvo_config_NH.yaml', 14)
simulation.start_animation()