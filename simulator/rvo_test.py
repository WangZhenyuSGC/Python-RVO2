from orca_simulator_base import OrcaSimulator
import numpy as np
import math
import random

class RVOSimulator(OrcaSimulator):
    def __init__(self, config_file_name = 'rvo_config.yaml'):
        super().__init__(config_file_name)        
        self.set_pos()
        self.agents_init()

    def set_pos(self):
        for i in range(self.robot_number):
            target_id = f"target{i+1}"
            self.targets[target_id]["pose"] = {"x": 200, "y": 720 + (-1) ** i * 250 * int((i + 1) / 2), "theta": 0}
    
        for i in range(self.robot_number):
            robot_id = f"robot{i+1}"
            self.robots[robot_id]["pose"] = {"x": 500, "y": 720 + (-1) ** i * 250 * int((i + 1) / 2), "theta": random.uniform(-np.pi, np.pi)} 

simulation = RVOSimulator()
simulation.start_animation()