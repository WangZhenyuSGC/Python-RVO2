from orca_simulator_base import OrcaSimulator

class RVOSimulator(OrcaSimulator):
    def __init__(self, config_file_name = 'rvo_config.yaml'):
        super().__init__(config_file_name)        
        self.agents_init()

simulation = RVOSimulator()
simulation.start_animation()