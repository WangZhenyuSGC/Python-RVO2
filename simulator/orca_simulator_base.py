import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../AVO2/")

import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import threading
import rvo2
import avo2

from time import time
import yaml
from abc import ABCMeta, abstractmethod
class OrcaFactory(object):
    def __init__(self) -> None:
        pass

    def __call__(self, library_name, time_step, neighbor_dist, max_neighbors, time_horizon, time_horizon_obst, effective_radius, lv_limit):
        if library_name == "rvo":
            return rvo2.PyRVOSimulator(time_step, neighbor_dist, max_neighbors, time_horizon, time_horizon_obst, effective_radius, lv_limit)
        elif library_name == "avo":
            return avo2.PyAVOSimulator()
        else:
            raise ValueError("The `library_name` must be set correctly.")

class OrcaSimulator:
    __metaclass__ = ABCMeta

    def __init__(self, config_file_name, robot_number):
        # Load parameters from the YAML file
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file_name)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.parameter_init(robot_number)

        # The first three characters of config file name are the library name
        library_name = config_file_name[:3]

        orca_factory = OrcaFactory()
        self.sim = orca_factory(library_name,self.time_step,
                                          self.neighbor_dist,
                                          self.max_neighbors,
                                          self.time_horizon,
                                          self.time_horizon_obst,
                                          self.effective_radius,
                                          self.lv_limit)
        self.robot_init()
        self.obst_init()
    
    def parameter_init(self, robot_number):
        self.time_step = self.config['time_step']
        self.neighbor_dist = self.config['neighbor_dist'] * 1000.0
        self.max_neighbors = self.config['max_neighbors']
        self.time_horizon = self.config['time_horizon']
        self.time_horizon_obst = self.config['time_horizon_obst']
        self.robot_radius = self.config['robot_radius'] * 1000.0
        self.max_speed = self.config['max_speed'] * 1000.0
        self.pos_threshold = self.config['pos_threshold'] * 1000.0
        self.ang_threshold = self.config['ang_threshold'] * 1000.0 
        self.lv_limit = self.config['lv_limit'] * 1000.0
        self.w_limit = self.config['w_limit']
        
        # Non-holonomic対応：有効半径と有効中心を使う
        self.D = self.config['enlarge_distance'] * 1000.0
        self.effective_radius = (self.D + self.robot_radius)
        self.L = self.config['wheel_distance'] * 1000.0

        # 描画用
        self.robot_number = robot_number
        self.stop_animation = False
        self.fig, self.ax = plt.subplots()
        self.ani = None
        self.ARROWLENGTH = 50
        self.FIELD_WIDTH = 820
        self.FIELD_HEIGHT = 1440 # unit: mm
        self.center_x = 400
        self.center_y = 700
        self.circle_radius = 200
        
        # Deadlock処理
        self.distances = {}
        self.low_speed_time = {} 
        self.finished_agents = set()
        self.dead_agents = set()        
        self.deadlock_time_limit = 2.5 # unit: s
        self.vel_threshold = 40 # unit: mm/s

        # 転倒処理
        self.fallen_agents = set()

    def obst_init(self):
        # Boundary設定
        vertices = []
        vertices.append((0,0))
        vertices.append((0, self.FIELD_HEIGHT))
        vertices.append((self.FIELD_WIDTH, self.FIELD_HEIGHT))
        vertices.append((self.FIELD_WIDTH, 0))
        vertices.append((0,0))

        # 障害物設定
        self.obsts = []
        self.obst_num = None
        self.obsts.append(vertices)
        self.sim.processObstacles()

    def add_agent(self, pos):
        return self.sim.addAgent(pos)
    
    def robot_init(self):
        self.targets = {}
        for i in range(self.robot_number): 
            target_pose = self.generate_safe_pose(self.targets)
            target_id = f"target{i+1}"
            self.targets[target_id] = {
                "id": target_id,
                "pose": target_pose
            }
        
        self.robots = {}
        for i in range(self.robot_number):  
            robot_pose = self.generate_safe_pose(self.robots)
            robot_id = f"robot{i+1}"
            self.robots[robot_id] = {
                "id": robot_id,
                "pose": robot_pose,
                "goal": None,
                "velocity": {"v": 0.0, "w": 0.0},
                "pos_flag": False, 
                "goal_flag": False
            }

    @abstractmethod
    def set_pos(self):
        pass
    
    def agents_init(self):
        self.agent_ids = {} # RVO agents

        for i, robot_id in enumerate(self.robots):
            target_id = f"target{(i % len(self.targets)) + 1}"
            self.robots[robot_id]["goal"] = self.targets[target_id]["pose"]
            # 初期位置を
            agent_id = self.add_agent((self.robots[robot_id]["pose"]["x"],self.robots[robot_id]["pose"]["y"]))
            self.agent_ids[robot_id] = agent_id

    def generate_safe_pose(self, group):
        position_safe = False
        while not position_safe:
            x = random.uniform(self.effective_radius, self.FIELD_WIDTH - self.effective_radius)
            y = random.uniform(self.effective_radius, self.FIELD_HEIGHT - self.effective_radius)
            theta = random.uniform(-math.pi, math.pi)

            position_safe = True
            for member_id, member in group.items():
                dist = math.sqrt((member["pose"]["x"] - x) ** 2 + (member["pose"]["y"] - y) ** 2)
                if dist < 2 / 3 * self.neighbor_dist :
                    position_safe = False
                    break

            if position_safe:
                return {"x": x, "y": y, "theta": theta}

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def enlarge_twist(self, agent_id, robot):
        target_v = self.sim.getAgentVelocity(agent_id)
        A = 0.5 * math.cos(robot["pose"]["theta"]) + self.D * math.sin(robot["pose"]["theta"]) / self.L
        B = 0.5 * math.cos(robot["pose"]["theta"]) - self.D * math.sin(robot["pose"]["theta"]) / self.L
        C = 0.5 * math.sin(robot["pose"]["theta"]) - self.D * math.cos(robot["pose"]["theta"]) / self.L
        D = 0.5 * math.sin(robot["pose"]["theta"]) + self.D * math.cos(robot["pose"]["theta"]) / self.L

        vx = target_v[0]
        vy = target_v[1]
        vr = (vy - C / A * vx) / (D - B * C / A)
        vl = (vx - B * vr) / A

        w = (vr - vl) / self.L
        lv = 0.5 * (vl + vr)

        return lv, w

    def update_status_enlarge(self, robot_id, robot):
        target = robot["goal"]
        
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
        if effective_distance_to_goal <= self.pos_threshold or distance_to_goal <= self.pos_threshold:
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

    def effective_vel_transform(self, current_pose, current_vel):
        v = current_vel[0]
        w = current_vel[1]
        theta = current_pose[2]

        vr = v + 0.5 * w * self.L
        vl = 2 * v - vr

        A = 0.5 * math.cos(theta) + self.D * math.sin(theta) / self.L
        B = 0.5 * math.cos(theta) - self.D * math.sin(theta) / self.L
        C = 0.5 * math.sin(theta) - self.D * math.cos(theta) / self.L
        D = 0.5 * math.sin(theta) + self.D * math.cos(theta) / self.L

        x_vel = A * vl + B * vr # v * cos(theta) - w * D sin(theta)
        y_vel = C * vl + D * vr # v * sin(theta) + w * D cos(theta)
        
        return np.array([x_vel, y_vel])

    def add_finished_agent(self, agent_id):
        if agent_id in self.dead_agents:
            self.dead_agents.remove(agent_id)

        if agent_id not in self.finished_agents:
            self.finished_agents.add(agent_id)
        
        self.sim.setAgentCollabCoeff(agent_id, 1.0)

    def update_low_speed_time(self, robot, agent_id):
        if (abs(robot["velocity"]["v"]) < self.vel_threshold) and (abs(robot["velocity"]["v"]) > 0) and agent_id not in self.finished_agents:
            self.low_speed_time[agent_id] = self.low_speed_time.get(agent_id, 0) + self.time_step
        else:
            self.low_speed_time[agent_id] = 0
    
    def add_dead_agent(self, agent_id):
        if self.low_speed_time.get(agent_id, 0) >= self.deadlock_time_limit and agent_id not in self.dead_agents:
            print("---------------------------This robot seems to suffer from a dead lock---------------------------", agent_id)
            self.dead_agents.add(agent_id)

        elif self.low_speed_time.get(agent_id, 0) < self.deadlock_time_limit and agent_id in self.dead_agents:
            self.dead_agents.remove(agent_id)
                                    
    def update_command(self):
        for robot_id, robot in self.robots.items():
            agent_id = self.agent_ids[robot_id]
            v,w = self.enlarge_twist(agent_id, robot)

            if abs(w) > abs(self.w_limit):
                print('--------------------------robot id, v, w before clip', (agent_id, v, w))
                pass

            v = clip(v, -self.lv_limit, self.lv_limit)
            w = clip(w, -self.w_limit, self.w_limit)   
            robot["velocity"]["v"] = v
            robot["velocity"]["w"] = w
            # print('robot id, v, w after clip', (agent_id, v, w))

    def move(self, robot_id, robot):
        # 実機には要らない、twistを投げればOK
        dt = self.time_step

        robot["pose"]["x"] += robot["velocity"]["v"] * math.cos(robot["pose"]["theta"]) * dt
        robot["pose"]["y"] += robot["velocity"]["v"] * math.sin(robot["pose"]["theta"]) * dt
        robot["pose"]["theta"] += robot["velocity"]["w"] * dt
        robot["pose"]["theta"] = self.normalize_angle(robot["pose"]["theta"])

    def update_trajectory(self,robot_id, robot):
        # 描画用、実機には要らない
        x = robot["pose"]["x"]
        y = robot["pose"]["y"]
        theta = robot["pose"]["theta"]
        target_x = robot["goal"]["x"]
        target_y = robot["goal"]["y"]
        target_theta = robot["goal"]["theta"]

        agent_id = self.agent_ids[robot_id]

        # Initialize a dictionary to store past positions if it doesn't exist
        if not hasattr(self, 'past_positions'):
            self.past_positions = {}

        # Initialize the past positions list for the robot if it doesn't exist
        if robot_id not in self.past_positions:
            self.past_positions[robot_id] = []

        # Update the past positions list with the current position
        self.past_positions[robot_id].append((x, y))

        # draw the obstacle
        for obst in self.obsts:
            self.ax.plot([obst[i][0] for i in range(len(obst))], [obst[i][1] for i in range(len(obst))], 'r-')

        if self.sim.getAgentCollabCoeff(agent_id) == 1.0:
            self.ax.add_artist(plt.Circle([x, y], self.effective_radius, color='green', fill=True))
        elif self.sim.getAgentCollabCoeff(agent_id) == 0:
            self.ax.add_artist(plt.Circle([x, y], self.effective_radius, color='red', fill=True))
        else:
            self.ax.add_artist(plt.Circle([x, y], self.effective_radius, color='blue', fill=True))

        self.ax.plot([x, target_x], [y, target_y], 'g--')  # Dashed green line
        self.ax.add_artist(plt.Circle([target_x, target_y], self.effective_radius, color='grey', fill=True))
        self.ax.annotate('', [x + self.ARROWLENGTH * np.cos(theta), y + self.ARROWLENGTH * np.sin(theta)], [x, y],
                        arrowprops=dict(arrowstyle='-|>', facecolor='red', edgecolor='red'))
        self.ax.text(x, y, agent_id + 1, color='white', ha='center', va='center')

        # Plot the past positions once
        if len(self.past_positions[robot_id]) > 1:
            past_positions = self.past_positions[robot_id]
            self.ax.plot([pos[0] for pos in past_positions], [pos[1] for pos in past_positions], 'bo-')  # Blue solid points and lines

    def update_loop(self, robot_id, robot):
        self.move(robot_id, robot)
        self.update_status_enlarge(robot_id, robot)
        self.update_trajectory(robot_id, robot)
            
    def main_loop(self, frame):
        if self.stop_animation:
            # plt.close()
            return
        
        self.ax.clear()

        if self.dead_agents != set():
            print("dead agents: ", self.dead_agents)

        threads = []
        for robot_id, robot in self.robots.items():
            thread = threading.Thread(target=self.update_loop, args=(robot_id, robot, ))
            threads.append(thread)
            thread.start()
            target = robot["goal"]
        
            vector_to_goal = np.array(
            [
                target["x"] - robot["pose"]["x"],
                target["y"] - robot["pose"]["y"]
            ]
            )

            distance_to_goal = np.linalg.norm(vector_to_goal)

            if abs(distance_to_goal) <= self.pos_threshold:
                robot["pos_flag"] = True
            else:
                robot["pos_flag"] = False

        for thread in threads:
            thread.join()

        if not all(robot["pos_flag"] for robot in self.robots.values()):
            self.sim.doStep()
            self.update_command()
        else:
            print("DONE!!!!")
            for robot_id, robot in self.robots.items():
                robot["velocity"]["v"] = 0
                robot["velocity"]["w"] = 0
                self.sim.removeAgent(self.agent_ids[robot_id])
            self.stop_animation = True
       
        self.ax.set_xlim(0, self.FIELD_WIDTH)
        self.ax.set_ylim(0, self.FIELD_HEIGHT)
        self.ax.set_aspect("equal")
        self.ax.set_title('Robot Simulation')
        plt.draw()
        # Do step計算と速度更新は0.5msぐらいかかります

    def initial_rotation(self):
        for robot_id, robot in self.robots.items():
            target = robot["goal"]
        
            vector_to_goal = np.array(
                [
                    target["x"] - robot["pose"]["x"],
                    target["y"] - robot["pose"]["y"]
                ]
            )
            initial_angle = math.atan2(vector_to_goal[1], vector_to_goal[0])
            initial_rotation = self.normalize_angle(initial_angle - robot["pose"]["theta"])
            robot["velocity"]["v"] = 0
            robot["velocity"]["w"] = initial_rotation / self.time_step

    def on_key(self, event):
        if event.key == 'q':
            self.stop_animation = True
        if event.key == 'k':
            self.ani = animation.FuncAnimation(self.fig,   self.main_loop, frames=30, interval=self.time_step)
        if event.key == 'd':
            dead_pos = self.sim.getAgentPosition(self.agent_ids["robot1"])

            self.robots["robot1"]["velocity"]["v"] = 0
            self.robots["robot1"]["velocity"]["w"] = 0

            self.fallen_agents.add("robot1")
            # generate several ellipse vertices 
            vertices = []
            for i in range(0, 390, 30):
                x = dead_pos[0] + 1.5 * self.effective_radius * math.cos(math.radians(i))
                y = dead_pos[1] + 0.8 * self.effective_radius * math.sin(math.radians(i))
                vertices.append((x, y))            
            # rotate the vertices around a random angle
            angle = random.uniform(-math.pi, math.pi)
            for i in range(len(vertices)):
                x = (vertices[i][0] - dead_pos[0]) * math.cos(angle) - (vertices[i][1] - dead_pos[1]) * math.sin(angle) + dead_pos[0]
                y = (vertices[i][0] - dead_pos[0]) * math.sin(angle) + (vertices[i][1] - dead_pos[1]) * math.cos(angle) + dead_pos[1]
                vertices[i] = (x, y)

            self.obsts.append(vertices)
            self.obst_num = self.sim.addObstacle(vertices)
            self.sim.processObstacles()

        if event.key == 'r':
            self.sim.removeObstacle(self.obst_num)
            self.sim.processObstacles()
            self.obsts.pop()
            self.fallen_agents.remove("robot1")            

    def start_animation(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.initial_rotation()
        plt.show()

def clip(value, min_v, max_v):
    return min(max(value, min_v), max_v)