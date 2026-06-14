"""Integration tests for MosaicNetworkFactory fan-in connectivity."""

from __future__ import annotations

import unittest

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig
from netochi.input_generator.mosaic_network_factory import MosaicNetworkFactory

from tests.input_generator.helpers import predecessors_in_core


class TestMosaicNetworkFactoryDistanceZero(unittest.TestCase):
    """Two neurons on the same core: distance 0, full intra-core fan-in except self."""

    def test_single_core_connects_all_neurons_except_self(self) -> None:
        neurons_per_core = 8
        hw = MosaicHardwareConfig(
            nodes_per_router=2,
            neurons_per_core=neurons_per_core,
            router_levels=1,
            slice_factor=2,
        )
        factory = MosaicNetworkFactory(hw_config=hw, probability=1.0, seed=123)
        result = factory.generate()

        for target in range(neurons_per_core):
            predecessors = predecessors_in_core(
                result.graph,
                target,
                core=0,
                neurons_per_core=neurons_per_core,
            )
            expected = set(range(neurons_per_core)) - {target}
            self.assertSetEqual(predecessors, expected)


class TestMosaicNetworkFactoryDistanceOne(unittest.TestCase):
    def test_neighboring_core_uses_exactly_one_slice_half(self) -> None:
        neurons_per_core = 8
        hw = MosaicHardwareConfig(
            nodes_per_router=2,
            neurons_per_core=neurons_per_core,
            router_levels=1,
            slice_factor=2,
        )
        factory = MosaicNetworkFactory(hw_config=hw, probability=1.0, seed=7)
        result = factory.generate()

        self.assertEqual(hw.core_distance(0, 1), 1)

        source_core = 0
        target_core = 1
        source_start = source_core * neurons_per_core
        half = neurons_per_core // 2

        for target in range(target_core * neurons_per_core, (target_core + 1) * neurons_per_core):
            chosen_slice = int(result.assignment.neuron_slice_assignment[target, 1])
            local_start, local_end = hw.get_slice_bounds(1, chosen_slice)
            expected_sources = set(range(source_start + local_start, source_start + local_end))

            incoming = predecessors_in_core(
                result.graph,
                target,
                core=source_core,
                neurons_per_core=neurons_per_core,
            )

            self.assertEqual(len(incoming), half)
            self.assertSetEqual(incoming, expected_sources)


class TestMosaicNetworkFactoryDistanceTwo(unittest.TestCase):
    def test_distant_core_uses_one_quarter_of_source_core(self) -> None:
        neurons_per_core = 8
        hw = MosaicHardwareConfig(
            nodes_per_router=2,
            neurons_per_core=neurons_per_core,
            router_levels=2,
            slice_factor=2,
        )
        factory = MosaicNetworkFactory(hw_config=hw, probability=1.0, seed=11)
        result = factory.generate()

        self.assertEqual(hw.core_distance(0, 2), 2)

        source_core = 0
        target_neuron = 2 * neurons_per_core
        chosen_slice = int(result.assignment.neuron_slice_assignment[target_neuron, 2])
        local_start, local_end = hw.get_slice_bounds(2, chosen_slice)
        expected_sources = set(range(source_core * neurons_per_core + local_start, source_core * neurons_per_core + local_end))

        incoming = predecessors_in_core(
            result.graph,
            target_neuron,
            core=source_core,
            neurons_per_core=neurons_per_core,
        )

        self.assertEqual(len(incoming), neurons_per_core // 4)
        self.assertSetEqual(incoming, expected_sources)


if __name__ == "__main__":
    unittest.main()
