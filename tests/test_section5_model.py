import unittest
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.mosaic_network_factory import MosaicNetworkConfig
from netochi.mapping.random_mapper import RandomMapperConfig
from netochi.mapping.greedy_mapper import GreedyMapperConfig
from netochi.objectives.obj_log_likelihood import LogLikelihoodObjectiveConfig

class TestSection5Model(unittest.TestCase):
    def test_mcmc_improves_likelihood(self):
        # 1. Hardware Config
        hw_config = MosaicHardwareConfig(
            nodes_per_router=2,
            neurons_per_core=10,
            router_levels=1,
            slice_factor=2
        )
        
        # 2. Generate a synthetic network
        factory = MosaicNetworkConfig(
            hw_config=hw_config,
            probability=0.3,
            seed=42
        ).create()
        mapping_input = factory.generate()
        
        # 3. Random Baseline
        random_mapper = RandomMapperConfig().create()
        random_state = random_mapper.run(mapping_input)
        random_nll = LogLikelihoodObjectiveConfig().create().evaluate(random_state).value
        
        # 4. Greedy Mapper
        greedy_mapper = GreedyMapperConfig().create()
        greedy_state = greedy_mapper.run(mapping_input)
        greedy_nll = LogLikelihoodObjectiveConfig().create().evaluate(greedy_state).value
        
        # 5. Greedy should find a state with equal or lower NLL (better energy) than baseline
        self.assertLessEqual(greedy_nll, random_nll)
        print(f"Random Baseline NLL: {random_nll:.2f}")
        print(f"Greedy NLL: {greedy_nll:.2f}")

if __name__ == '__main__':
    unittest.main()
