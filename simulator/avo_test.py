from orca_simulator_base import OrcaSimulator
import numpy as np
import math
import random

class AVOSimulator(OrcaSimulator):
    def __init__(self, config_file_name = 'avo_config.yaml'):
        super().__init__(config_file_name)

        # AVOの初期化方式は違うため
        self.max_accel = self.config['max_accel'] * 1000.0
        self.accel_interval = self.config['accel_interval']
        self.sim.setTimeStep(self.time_step)
        
        self.set_pos()
        self.agents_init()

    def add_agent(self, pos):
        return self.sim.addAgent(pos, self.neighbor_dist, self.max_neighbors, 
                             self.time_horizon, self.effective_radius, self.max_speed,
                             self.max_accel, self.accel_interval, tuple([0,0]))
    
    def obst_init(self):
        # AVOは障害物の設定がないため
        self.obsts = []
        self.obst_num = None
    
    def set_pos(self):
        for i in range(self.robot_number):
            target_id = f"target{i+1}"
            self.targets[target_id]["pose"] = {"x": 200, "y": 720 + (-1) ** i * 250 * int((i + 1) / 2), "theta": 0}
    
        for i in range(self.robot_number):
            robot_id = f"robot{i+1}"
            self.robots[robot_id]["pose"] = {"x": 500, "y": 720 + (-1) ** i * 250 * int((i + 1) / 2), "theta": random.uniform(-np.pi, np.pi)} 

simulation = AVOSimulator()
simulation.start_animation()