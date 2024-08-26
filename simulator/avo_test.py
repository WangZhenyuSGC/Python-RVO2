from orca_simulator_base import OrcaSimulator

class AVOSimulator(OrcaSimulator):
    def __init__(self, config_file_name = 'avo_config.yaml'):
        super().__init__(config_file_name)

        # AVOの初期化方式は違うため
        self.max_accel = self.config['max_accel'] * 1000.0
        self.accel_interval = self.config['accel_interval']
        self.sim.setTimeStep(self.time_step)
        
        self.agents_init()

    def add_agent(self, pos):
        return self.sim.addAgent(pos, self.neighbor_dist, self.max_neighbors, 
                             self.time_horizon, self.effective_radius, self.max_speed,
                             self.max_accel, self.accel_interval, tuple([0,0]))
    
    def obst_init(self):
        # AVOは障害物の設定がないため
        self.obsts = []
        self.obst_num = None

simulation = AVOSimulator()
simulation.start_animation()