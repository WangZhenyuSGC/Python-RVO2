import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../python_motion_planning")
from orca_simulator_base import OrcaSimulator
import numpy as np
import math
import random
from python_motion_planning.utils import Grid, Map, SearchFactory

class RVOSimulator(OrcaSimulator):
    def __init__(self, config_file_name = 'rvo_config.yaml', robot_number = 5):
        super().__init__(config_file_name, robot_number)      
        self.set_pos()
        self.agents_init()

    def agents_init(self):
        self.agent_ids = {} # RVO agents

        self.agent_paths = {} # Paths generated by Theta*
        search_factory = SearchFactory()
        env = Grid(self.FIELD_WIDTH, self.FIELD_HEIGHT)

        for i, robot_id in enumerate(self.robots):
            target_id = f"target{(i % len(self.targets)) + 1}"
            self.robots[robot_id]["goal"] = self.targets[target_id]["pose"]
            # 初期位置を
            agent_id = self.add_agent((self.robots[robot_id]["pose"]["x"],self.robots[robot_id]["pose"]["y"]))
            self.agent_ids[robot_id] = agent_id
            # Path planning
            start = (int(self.robots[robot_id]["pose"]["x"]), int(self.robots[robot_id]["pose"]["y"]))
            goal = (int(self.robots[robot_id]["goal"]["x"]), int(self.robots[robot_id]["goal"]["y"]))
            planner = search_factory("theta_star", start=start, goal=goal, env=env)
            cost, path, expand = planner.plan()
            # planner.run()

            self.agent_paths[robot_id] = path
            print(f"robot_id: {robot_id}, path: {path}")

    # def set_pos(self):
    #     for i in range(self.robot_number):
    #         target_id = f"target{i+1}"
    #         self.targets[target_id]["pose"] = {"x": 200, "y": 720 + (-1) ** i * 250 * int((i + 1) / 2), "theta": 0}
    
    #     for i in range(self.robot_number):
    #         robot_id = f"robot{i+1}"
    #         self.robots[robot_id]["pose"] = {"x": 500, "y": 720 + (-1) ** i * 250 * int((i + 1) / 2), "theta": random.uniform(-np.pi, np.pi)} 

simulation = RVOSimulator('rvo_config.yaml', 5)
simulation.start_animation()