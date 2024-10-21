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
class RVOCCBSSimulator(OrcaSimulator):
    def __init__(self, config_file_name = 'rvo_config.yaml', robot_number = 5):
        super().__init__(config_file_name, robot_number)      
        self.set_pos()
        self.agents_init()
        self.path_planning()

        # vertices = []
        # vertices.append((450,450))
        # vertices.append((450, 550))
        # vertices.append((350, 550))
        # vertices.append((350, 450))
        # vertices.append((450,450))

        # vertices.append((250, 350))
        # vertices.append((250, 650))
        # vertices.append((150, 650))
        # vertices.append((150, 250))
        # vertices.append((650, 250))
        # vertices.append((650, 650))
        # vertices.append((550, 650))
        # vertices.append((550, 350))
        # vertices.append((250, 350))

        # # 障害物設定
        # self.obsts.append(vertices)
        # self.sim.addObstacle(vertices)
        # self.sim.processObstacles()

    def find_closest_cell(self, x, y):
        # Setting the map as 14x14 grid, while each cell is 0.1 x 0.1. Unit is meter.
        x = x / 100.0
        y = y / 100.0
        return x, y

    def path_planning(self):
        root = ET.Element("root")
        for i in range(self.robot_number):
            robot_id = f"robot{i+1}"
            robot = self.robots[robot_id]
            target_id = f"target{i+1}"
            target = self.targets[target_id]
            start_i, start_j = self.find_closest_cell(robot["pose"]["x"], robot["pose"]["y"])
            goal_i, goal_j = self.find_closest_cell(target["pose"]["x"], target["pose"]["y"])
            # Save these info to grid_task.xml
            agent = ET.SubElement(root, "agent")
            agent.set("start_i", str(start_i))
            agent.set("start_j", str(start_j))
            agent.set("goal_i", str(goal_i))
            agent.set("goal_j", str(goal_j))
        # ElementTreeを文字列に変換
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_string = reparsed.toprettyxml(indent="   ")

        # ファイルに書き込み
        with open('grid_task.xml', "w", encoding="utf-8") as f:
            f.write(pretty_string)

        self.ccbs = CCBS.PyCCBS(b'grid_map.xml', b'grid_task.xml', b'config.xml')
        self.planned_paths = self.ccbs.find_solution()
        print(self.planned_paths)

    def set_pos(self):
        self.targets = {}
        self.robots = {}
        for i in range(self.robot_number):
            target_id = f"target{i+1}"
            circle_radius = 200
            center_x = 400
            center_y = 700
            target_x, target_y = circle(i, center_x, center_y, circle_radius, self.robot_number) 
            target_pose = {"x": target_x, "y": target_y, "theta": random.uniform(-np.pi, np.pi)} 
            self.targets[target_id] = {
                "id": target_id,
                "pose": target_pose
            }
    
        # self.targets["target1"]["pose"]= {'x': 400, 'y': 100, 'theta': 0}
        # self.targets["target2"]["pose"]= {'x': 400, 'y': 1100, 'theta': 0}
        # self.targets["target3"]["pose"]= {'x': 50, 'y': 750, 'theta': 0}
        # self.targets["target4"]["pose"]= {'x': 750, 'y': 950, 'theta': 0}

        for i in range(self.robot_number):
            robot_id = f"robot{i+1}"            
            robot_pose = {"x": 600, "y": 720 + (-1) ** i * 250 * int((i + 1) / 2), "theta": random.uniform(-np.pi, np.pi)}
            self.robots[robot_id] = {
                "id": robot_id,
                "pose": robot_pose,
                "goal": None,
                "velocity": {"v": 0.0, "w": 0.0},
                "pos_flag": False, 
                "goal_flag": False
            }
        # self.robots["robot1"]["pose"]= {'x': 400, 'y': 1100, 'theta': 0}
        # self.robots["robot2"]["pose"]= {'x': 400, 'y': 100, 'theta': 0}
        # self.robots["robot3"]["pose"]= {'x': 750, 'y': 550, 'theta': 0}
        # self.robots["robot4"]["pose"]= {'x': 250, 'y': 850, 'theta': 0}

    def specify_current_section(self, current_time, robot_id):
        for path in self.planned_paths['paths']:
            if path['agentID'] == robot_id:
                elapsed_time = 0
                for section in path['sections']:
                    elapsed_time += section['duration']
                    if current_time <= elapsed_time:
                        return section['goal_i'], section['goal_j']
        return None, None

    def update_status_enlarge(self, robot_id, robot):
        # Overwrite the update_status function in OrcaSimulator to update the pref vel to the next waypoint
        # The current time since the simulation starts
        current_time = self.sim.getGlobalTime()
        # Find the corresponding section based on the current time from self.planned_paths
        agent_id = self.agent_ids[robot_id]
        goal_i, goal_j = self.specify_current_section(current_time, agent_id)
        if goal_i is None and goal_j is None:
            target = robot["goal"]
            final_flag = True
        else:
            # print("Current target should be:", goal_i, goal_j)
            target = {"x": goal_i * 100, "y": goal_j * 100}
            final_flag = False
        
        effective_center = [robot["pose"]["x"] + self.D * math.cos(robot["pose"]["theta"]),
                            robot["pose"]["y"] + self.D * math.sin(robot["pose"]["theta"])]

        vector_to_goal = np.array(
            [
                target["x"] - robot["pose"]["x"], 
                target["y"] - robot["pose"]["y"]
            ]
        )

        effective_vector_to_goal = np.array(
            [
                target["x"] - effective_center[0],
                target["y"] - effective_center[1]
            ]
        )

        pref_vel = tuple(effective_vector_to_goal)
        distance_to_goal = np.linalg.norm(vector_to_goal)
        effective_distance_to_goal = np.linalg.norm(effective_vector_to_goal)        
        goal_flag = False
        distance_to_goal = np.linalg.norm(effective_vector_to_goal)

        agent_id = self.agent_ids[robot_id]
        self.distances[agent_id] = {"distance": distance_to_goal}
        
        if robot_id in self.fallen_agents:         
            self.sim.setAgentMaxSpeed(agent_id, 0)
            self.sim.setAgentCollabCoeff(agent_id, 0.0)
            self.sim.setAgentPosition(agent_id, (effective_center[0], effective_center[1]))
            self.sim.setAgentVelocity(agent_id, (0.0,0.0))
            self.sim.setAgentPrefVelocity(agent_id, (0, 0))
            return
        else:
            self.sim.setAgentMaxSpeed(agent_id, self.lv_limit)
            self.sim.setAgentCollabCoeff(agent_id, 0.5)

        # 目標点の一定距離内に入ったら理想速度を調整する
        if effective_distance_to_goal <= self.pos_threshold or distance_to_goal <= self.pos_threshold and final_flag:
            pref_vel = tuple(vector_to_goal)
            # 正式のGOAL判定は有効中心ではなく実際の中心で行う
            goal_flag = distance_to_goal <= self.pos_threshold  

        current_pose = tuple((robot["pose"]["x"], robot["pose"]["y"], robot["pose"]["theta"]))
        current_vel = tuple((robot["velocity"]["v"], robot["velocity"]["w"]))   
        effective_cur_vel = self.effective_vel_transform(current_pose, current_vel)

        agent_id = self.agent_ids[robot_id]

        # RVO2に現在の位置と速度を伝える
        self.sim.setAgentPosition(agent_id, tuple((effective_center[0], effective_center[1])))
        
        # 現在の速度を偏移された有効速度に変換してRVO2に伝える
        self.sim.setAgentVelocity(agent_id, tuple((effective_cur_vel[0], effective_cur_vel[1])))

        # RVO2に目標位置の方向（理想速度）を伝える
        self.sim.setAgentPrefVelocity(agent_id, tuple((pref_vel[0], pref_vel[1])))
        robot["goal_flag"] = goal_flag

simulation = RVOCCBSSimulator('rvo_config.yaml', 5)
simulation.start_animation()