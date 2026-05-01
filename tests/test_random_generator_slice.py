import unittest

from netochi.network_generator.random_generator import RandomGenerator, RandomGeneratorConfig


class TestRandomGeneratorSliceConstraints(unittest.TestCase):

    def test_distance_zero_single_core_connects_all_neurons_except_self(self):
        g = 8
        config = RandomGeneratorConfig(
            nodes_per_router=1,
            neurons_per_core=g,
            router_levels=1,
            edge_probability=1.0,
        )
        result = RandomGenerator(seed=123).generate(config)

        # One core at distance 0: every neuron receives from all other neurons in that core.
        for target in range(g):
            predecessors = set(result.graph.predecessors(target))
            expected = set(range(g)) - {target}
            self.assertSetEqual(predecessors, expected)

    def test_distance_one_uses_exactly_one_half_of_source_core(self):
        g = 8
        config = RandomGeneratorConfig(
            nodes_per_router=2,
            neurons_per_core=g,
            router_levels=1,
            edge_probability=1.0,
        )
        generator = RandomGenerator(seed=7)

        # Two cores under one level-1 router are immediate neighbors: distance must be 1.
        self.assertEqual(generator._core_distance(0, 1, config), 1)
        self.assertEqual(generator._core_distance(1, 0, config), 1)

        result = generator.generate(config)

        source_core = 0
        target_core = 1
        source_start = source_core * g
        source_end = source_start + g
        half = g // 2

        for target in range(target_core * g, (target_core + 1) * g):
            chosen_slice = int(result.assignment[target, 1])
            local_start, local_end = generator._slice_bounds(g, 2, chosen_slice)
            expected_sources = set(range(source_start + local_start, source_start + local_end))

            predecessors = set(result.graph.predecessors(target))
            incoming_from_source_core = {src for src in predecessors if source_start <= src < source_end}

            self.assertEqual(len(incoming_from_source_core), half)
            self.assertSetEqual(incoming_from_source_core, expected_sources)


if __name__ == "__main__":
    unittest.main()

