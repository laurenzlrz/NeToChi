import unittest
from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.mosaic_network_factory import MosaicNetworkFactory

class TestRandomGeneratorSliceConstraints(unittest.TestCase):

    def test_distance_zero_single_core_connects_all_neurons_except_self(self):
        g = 8
        hw_config = MosaicHardwareConfig(
            nodes_per_router=1,
            neurons_per_core=g,
            router_levels=1,
            slice_factor=2
        )
        factory = MosaicNetworkFactory(hw_config=hw_config, probability=1.0, seed=123)
        result = factory.generate()

        # One core at distance 0: every neuron receives from all other neurons in that core.
        for target in range(g):
            predecessors = set(int(src) for src in result.graph.vertex(target).in_neighbors())
            expected = set(range(g)) - {target}
            self.assertSetEqual(predecessors, expected)

    def test_distance_one_uses_exactly_one_half_of_source_core(self):
        g = 8
        hw_config = MosaicHardwareConfig(
            nodes_per_router=2,
            neurons_per_core=g,
            router_levels=1,
            slice_factor=2
        )
        factory = MosaicNetworkFactory(hw_config=hw_config, probability=1.0, seed=7)

        # Two cores under one level-1 router are immediate neighbors: distance must be 1.
        self.assertEqual(hw_config.core_distance(0, 1), 1)
        self.assertEqual(hw_config.core_distance(1, 0), 1)

        result = factory.generate()

        source_core = 0
        target_core = 1
        source_start = source_core * g
        source_end = source_start + g
        half = g // 2

        for target in range(target_core * g, (target_core + 1) * g):
            chosen_slice = int(result.assignment.neuron_slice_assignment[target, 1])
            local_start, local_end = hw_config.get_slice_bounds(1, chosen_slice)
            expected_sources = set(range(source_start + local_start, source_start + local_end))

            predecessors = set(int(src) for src in result.graph.vertex(target).in_neighbors())
            incoming_from_source_core = {src for src in predecessors if source_start <= src < source_end}

            self.assertEqual(len(incoming_from_source_core), half)
            self.assertSetEqual(incoming_from_source_core, expected_sources)
