import sys
sys.path.append('/home/0000410764/workspace/aii-proto/remote_pc/ros2_ws/src/Python-RVO2')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import threading
import rvo2
from time import time

print(rvo2.__file__)

class RobotSimulation:
    def __init__(self):
        self.ROBOT_RADIUS = 40 # ロボット半径
        self.AVOIDANCE_THRESHOLD = 200 #衝突判定用距離
        
        self.ARROWLENGTH = 50
        self.FIELD_WIDTH = 820
        self.FIELD_HEIGHT = 1440 # unit: mm

        self.POS_THRESHOLD = 20 # 目標点の到達判定用
        self.ANGLE_THRESHOLD = 0.05
        self.FREQUENCY = 0.033 # unit: s
        self.PERTURB_FLAG = False

        self.deadlock_time_limit = 2.5 # unit: s
        self.vel_threshold = 40 # unit: mm/s

        self.lv_limit = 100
        self.w_limit = np.pi
        
        self.time_horizon = 100 * self.FREQUENCY
        
        self.center_x = 400
        self.center_y = 700
        self.circle_radius = 200

        self.D = 0.3 * self.ROBOT_RADIUS
        self.L = 25
        self.effective_radius = self.D + self.ROBOT_RADIUS

        self.robot_number = 5
        self.stop_animation = False
        # RVOSimulator(float timeStep, 
        #              float neighborDist, 
        # 　　　　　　　size_t maxNeighbors,
        # 　　　　　　　float timeHorizon,
        # 　　　　　　　float timeHorizonObst,
        # 　　　　　　　float radius,
        # 　　　　　　　float maxSpeed,
        # 　　　　　　　const Vector2 &velocity)　OPTIONAL 

        self.rvosim = rvo2.PyRVOSimulator(self.FREQUENCY,
                                          1.5 * self.AVOIDANCE_THRESHOLD,
                                          10,
                                          self.time_horizon,
                                          0.5 * self.time_horizon,
                                          self.effective_radius,
                                          self.lv_limit)
        self.fig, self.ax = plt.subplots()
        self.ani = None

        self.robot_init()

    def robot_init(self):
        self.targets = {}
        for i in range(self.robot_number): 
            target_pose = self.generate_safe_pose(self.targets)
            target_id = f"target{i+1}"
            self.targets[target_id] = {
                "id": target_id,
                "pose": target_pose
            }
        
        # for i in range(self.robot_number):
        #     target_id = f"target{i+1}"
        #     #self.targets[target_id]["pose"] = {"x": 200, "y": 720 + (-1) ** i * 250 * int((i + 1) / 2), "theta": 0}
        #     self.targets[target_id]["pose"] = {"x": 200, "y": 1120 - i * 800, "theta": 0}
        
        for i in range(self.robot_number):
            target_id = f"target{i+1}"
            self.targets[target_id]["pose"] = {"x": self.circle_radius * np.cos(2 * np.pi * i / self.robot_number) + self.center_x,
                                               "y": self.circle_radius * np.sin(2 * np.pi * i/ self.robot_number) + self.center_y,
                                               "theta": 0
                                              }
        
        # self.targets = { "target1": {"id": "target1", "pose": {"x": 200, "y": 200, "theta": 0}},
        #                  "target2": {"id": "target2", "pose": {"x": 300, "y": 400, "theta": 0}},
        #                  "target3": {"id": "target3", "pose": {"x": 400, "y": 200, "theta": 0}},
        #                 }
        
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

        # for i in range(self.robot_number):
        #     robot_id = f"robot{i+1}"
        #     self.robots[robot_id]["pose"] = {"x": 200, "y": 720 + (-1) ** i * 250 * int((i + 1) / 2), "theta": random.uniform(-np.pi, np.pi)} 
        #     # self.robots[robot_id]["pose"] = {"x": 200, "y": 320 + i * 800, "theta": random.uniform(-np.pi, np.pi)} 

        # self.robots["robot1"]["pose"] = {"x": 600, "y": 500, "theta": 0}
        # self.robots["robot2"]["pose"] = {"x": 400, "y": 1000, "theta": 0}
        # self.robots["robot3"]["pose"] = {"x": 100, "y": 800, "theta": 0}

        self.agent_ids = {} # RVO agents

        for i, robot_id in enumerate(self.robots):
            target_id = f"target{(i % len(self.targets)) + 1}"
            self.robots[robot_id]["goal"] = self.targets[target_id]["pose"]
            # 初期位置を
            agent_id = self.rvosim.addAgent((self.robots[robot_id]["pose"]["x"],self.robots[robot_id]["pose"]["y"]))
            self.agent_ids[robot_id] = agent_id
            self.rvosim.setAgentCollabCoeff(agent_id, 0.5)

        self.distances = {}
        self.low_speed_time = {} 
        self.finished_agents = set()
        self.dead_agents = set()

        # Boundary設定
        vertices = []
        vertices.append((0,0))
        vertices.append((0, self.FIELD_HEIGHT))
        vertices.append((self.FIELD_WIDTH, self.FIELD_HEIGHT))
        vertices.append((self.FIELD_WIDTH, 0))
        vertices.append((0,0))

        self.rvosim.addObstacle(vertices)
        self.rvosim.processObstacles()

    def generate_safe_pose(self, group):
        position_safe = False
        while not position_safe:
            x = random.uniform(self.effective_radius, self.FIELD_WIDTH - self.effective_radius)
            y = random.uniform(self.effective_radius, self.FIELD_HEIGHT - self.effective_radius)
            theta = random.uniform(-math.pi, math.pi)

            position_safe = True
            for member_id, member in group.items():
                dist = math.sqrt((member["pose"]["x"] - x) ** 2 + (member["pose"]["y"] - y) ** 2)
                if dist < self.AVOIDANCE_THRESHOLD :
                    position_safe = False
                    break

            if position_safe:
                return {"x": x, "y": y, "theta": theta}

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def enlarge_twist(self, agent_id, robot):
        target_v = self.rvosim.getAgentVelocity(agent_id)
        A = 0.5 * math.cos(robot["pose"]["theta"]) + self.D * math.sin(robot["pose"]["theta"]) / self.L
        B = 0.5 * math.cos(robot["pose"]["theta"]) - self.D * math.sin(robot["pose"]["theta"]) / self.L
        C = 0.5 * math.sin(robot["pose"]["theta"]) - self.D * math.cos(robot["pose"]["theta"]) / self.L
        D = 0.5 * math.sin(robot["pose"]["theta"]) + self.D * math.cos(robot["pose"]["theta"]) / self.L

        # M = np.array([
        #     [A,B],
        #     [C,D],
        # ])
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
        
        effective_vector_to_goal = np.array(
            [
                target["x"] - effective_center[0],
                target["y"] - effective_center[1]
            ]
        )

        distance_to_goal = np.linalg.norm(effective_vector_to_goal)

        perturb_x = 0.0
        perturb_y = 0.0

        # Add a perturb to avoid dead lock
        if self.PERTURB_FLAG:
            perturb_angle = random.uniform(-1.0, 1.0) * math.pi
            perturb_dist = random.uniform(-self.POS_THRESHOLD, self.POS_THRESHOLD)
            perturb_x = perturb_dist * math.cos(perturb_angle)
            perturb_y = perturb_dist * math.sin(perturb_angle)
        
        agent_id = self.agent_ids[robot_id]
        self.distances[agent_id] = {"distance": distance_to_goal}

        if abs(distance_to_goal) <= self.POS_THRESHOLD:
            robot["pos_flag"] = True
            final_vector = [effective_vector_to_goal[0] + self.D * math.cos(robot["pose"]["theta"]), effective_vector_to_goal[1] + self.D * math.sin(robot["pose"]["theta"])]
            v = tuple(final_vector)
            
            self.add_finished_agent(agent_id)

        else:
            robot["pose_flag"] = False
            effective_vector_to_goal += perturb_x
            effective_vector_to_goal += perturb_y
            v = tuple(effective_vector_to_goal)

            if agent_id in self.finished_agents:
                self.finished_agents.remove(agent_id)

            self.add_dead_agent(agent_id)

        self.update_low_speed_time(robot, agent_id)   
        # RVO2に現在の位置と速度を伝える (実機はglobal poseを)
        self.rvosim.setAgentPosition(agent_id, (effective_center[0], effective_center[1]))

        # 実機は直前の速度指令より、odometryで実際の速度を測った方がいい
        self.rvosim.setAgentVelocity(agent_id, 
                                    (
                                     robot["velocity"]["v"] * math.cos(robot["pose"]["theta"]),
                                     robot["velocity"]["v"] * math.sin(robot["pose"]["theta"])
                                    )
                                    )
        
        # RVO2に目標位置の方向（理想速度）を伝える
        self.rvosim.setAgentPrefVelocity(agent_id, v)

    def add_finished_agent(self, agent_id):
        if agent_id in self.dead_agents:
            self.dead_agents.remove(agent_id)

        if agent_id not in self.finished_agents:
            self.finished_agents.add(agent_id)
        
        self.rvosim.setAgentCollabCoeff(agent_id, 1.0)

    def update_low_speed_time(self, robot, agent_id):
        if (abs(robot["velocity"]["v"]) < self.vel_threshold) and (abs(robot["velocity"]["v"]) > 0) and agent_id not in self.finished_agents:
            self.low_speed_time[agent_id] = self.low_speed_time.get(agent_id, 0) + self.FREQUENCY
        else:
            self.low_speed_time[agent_id] = 0
    
    def add_dead_agent(self, agent_id):
        if self.low_speed_time.get(agent_id, 0) >= self.deadlock_time_limit and agent_id not in self.dead_agents:
            print("---------------------------This robot seems to suffer from a dead lock---------------------------", agent_id)
            self.dead_agents.add(agent_id)


        elif self.low_speed_time.get(agent_id, 0) < self.deadlock_time_limit and agent_id in self.dead_agents:
            self.dead_agents.remove(agent_id)

    def adjust_pref_vel(self, agent_id):
        num_neighbors = self.rvosim.getAgentNumAgentNeighbors(agent_id)
        print("---------------------------The neighboring agent num---------------------------", num_neighbors)
        neighbor_distances = {}
        neighbor_ids = []
        
        for i in range(num_neighbors):
            neighbor_id = self.rvosim.getAgentAgentNeighbor(agent_id, i)
            neighbor_ids.append(neighbor_id)
            neighbor_pos = self.rvosim.getAgentPosition(neighbor_id)
            dead_pos = self.rvosim.getAgentPosition(agent_id)
            distance = np.linalg.norm(np.array(dead_pos) - np.array(neighbor_pos))
            neighbor_distances[neighbor_id] = distance
        
        sorted_neighbors = sorted(neighbor_distances, key=neighbor_distances.get)
        
        for index, neighbor_id in enumerate(sorted_neighbors):
            dead_pos = self.rvosim.getAgentPosition(agent_id)
            neighbor_pos = self.rvosim.getAgentPosition(neighbor_id)
            
            # Adjust the pref velocity by rotating the vector from the dead agent to the neighbor agent by 90 degrees
            vector = (neighbor_pos[0] - dead_pos[0], neighbor_pos[1] - dead_pos[1])
            vector_rot = (-vector[1], vector[0])

            mirrored_pos = (neighbor_pos[0] + vector_rot[0], neighbor_pos[1] + vector_rot[1])

            if (self.rvosim.queryVisibility(mirrored_pos, neighbor_pos, self.ROBOT_RADIUS) and self.rvosim.queryVisibility(neighbor_pos, mirrored_pos, self.ROBOT_RADIUS)):        
                self.rvosim.setAgentPrefVelocity(neighbor_id, vector_rot)
                self.rvosim.setAgentCollabCoeff(neighbor_id, 1.0)
                self.rvosim.setAgentMaxSpeed(agent_id, self.lv_limit)
                print("---------------------------Adjust the pref velocity obst safe---------------------------", neighbor_id, vector_rot)
            else:
                vector_rot_inv = (vector[1], -vector[0])
                self.rvosim.setAgentPrefVelocity(neighbor_id, vector_rot_inv)
                self.rvosim.setAgentCollabCoeff(neighbor_id, 1.0)
                self.rvosim.setAgentMaxSpeed(agent_id, self.lv_limit)
                print("---------------------------Adjust the pref velocity by avoiding obst---------------------------", neighbor_id, vector_rot_inv)
        self.rvosim.setAgentCollabCoeff(agent_id, 0)
        self.rvosim.setAgentMaxSpeed(agent_id, 0.5 * self.lv_limit)
                                    
    def update_command(self):
        for robot_id, robot in self.robots.items():
            agent_id = self.agent_ids[robot_id]

            # v,w = self.calc_slalom(agent_id, robot)
            # v,w = self.pos_twist(agent_id, robot)
            # v,w = self.vel_twist(agent_id, robot)
            # v,w = self.vel_slalom(agent_id, robot)
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
        dt = self.FREQUENCY

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

        if self.rvosim.getAgentCollabCoeff(agent_id) == 1.0:
            self.ax.add_artist(plt.Circle([x, y], self.ROBOT_RADIUS, color='green', fill=True))
        elif self.rvosim.getAgentCollabCoeff(agent_id) == 0:
            self.ax.add_artist(plt.Circle([x, y], self.ROBOT_RADIUS, color='red', fill=True))
        else:
            self.ax.add_artist(plt.Circle([x, y], self.ROBOT_RADIUS, color='blue', fill=True))
        
        self.ax.plot([x, target_x], [y, target_y], 'g--')  # Dashed green line
        self.ax.add_artist(plt.Circle([target_x, target_y], self.ROBOT_RADIUS, color='grey', fill=True))
        self.ax.annotate('', [x + self.ARROWLENGTH * np.cos(theta), y + self.ARROWLENGTH * np.sin(theta)], [x, y],
                            arrowprops=dict(arrowstyle='-|>', facecolor='red', edgecolor='red'))
        # self.ax.annotate('', [target_x + self.ARROWLENGTH * np.cos(target_theta), target_y + self.ARROWLENGTH * np.sin(target_theta)], [target_x, target_y],
                            # arrowprops=dict(arrowstyle='-|>', facecolor='blue', edgecolor='blue'))
        self.ax.text(x, y, agent_id, color='white', ha='center', va='center')

    def update_loop(self, robot_id, robot):
        self.move(robot_id, robot)
        # self.update_status(robot_id, robot)
        self.update_status_enlarge(robot_id, robot)
        # agent_id = self.agent_ids[robot_id]
        # if agent_id in self.dead_agents:
        #     self.adjust_pref_vel(agent_id)
        self.update_trajectory(robot_id, robot)
            
    def main_loop(self, frame):
        if self.stop_animation:
            plt.close()
            return
        
        # 毎回のループは50msらしい

        self.ax.clear()
        start_time = time()
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

            if abs(distance_to_goal) <= self.POS_THRESHOLD:
                robot["pos_flag"] = True
            else:
                robot["pos_flag"] = False

        for thread in threads:
            thread.join()

        end_time = time()  
        # print("Time taken by Update Status: ", end_time - start_time)
        # 13msぐらいかかります

        if not all(robot["pos_flag"] for robot in self.robots.values()):
            self.rvosim.doStep()
            self.update_command()
        else:
            print("DONE!!!!")
            for robot_id, robot in self.robots.items():
                robot["velocity"]["v"] = 0
                robot["velocity"]["w"] = 0
                self.rvosim.removeAgent(self.agent_ids[robot_id])
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
            robot["velocity"]["w"] = initial_rotation / self.FREQUENCY

    def on_key(self, event):
        if event.key == 'q':
            self.stop_animation = True
        if event.key == 'k':
            self.ani = animation.FuncAnimation(self.fig, self.main_loop, frames=30, interval=self.FREQUENCY)

    def start_animation(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.initial_rotation()
        plt.show()

def clip(value, min_v, max_v):
    return min(max(value, min_v), max_v)

simulation = RobotSimulation()
simulation.start_animation()