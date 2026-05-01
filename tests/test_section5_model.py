import unittest
import networkx as nx
import graph_tool.all as gt
from netochi.network_generator.random_generator import RandomGeneratorConfig, RandomNetwork
from netochi.mapping.hardware_config import HardwareConfig
from netochi.mapping.random_mapper import RandomMapper
from netochi.mapping.mcmc_mapper import Section5SBM_MCMCMapper
from netochi.mapping.hybrid_mapper import HybridMapper

def nx_to_gt(nx_g: nx.DiGraph) -> gt.Graph:
    """Helper to convert networkx graph to graph-tool graph."""
    gt_g = gt.Graph(directed=nx_g.is_directed())
    gt_g.add_vertex(nx_g.number_of_nodes())
    gt_g.add_edge_list(list(nx_g.edges()))
    return gt_g

class TestSection5Model(unittest.TestCase):
    def test_mcmc_improves_likelihood(self):
        # 1. Hardware Config
        hw_config = HardwareConfig(
            nodes_per_router=2,
            neurons_per_core=10,
            router_levels=1,
            slice_factor=2
        )
        
        # 2. Generate a synthetic network
        net_config = RandomGeneratorConfig(
            nodes_per_router=2,
            neurons_per_core=10,
            router_levels=1,
            edge_probability=0.3,
            seed=42
        )
        generator = RandomNetwork(net_config)
        generator.generate()
        nx_graph = generator._graph
        graph = nx_to_gt(nx_graph)
        
        # 3. Random Baseline
        random_mapper = RandomMapper(graph, hw_config)
        baseline_ll = random_mapper.map_network(seed=42)
        
        # 4. MCMC Mapper
        mcmc_mapper = Section5SBM_MCMCMapper(graph, hw_config)
        optimized_ll = mcmc_mapper.optimize(iterations=1000, initial_temp=5.0, seed=42)
        
        # 5. Hybrid Mapper
        hybrid_mapper = HybridMapper(graph, hw_config)
        hybrid_ll = hybrid_mapper.map_network()
        
        # 6. MCMC should find a state at least as good as the baseline
        self.assertGreaterEqual(optimized_ll, baseline_ll)
        print(f"Baseline LL: {baseline_ll:.2f}")
        print(f"Hybrid LL: {hybrid_ll:.2f}")
        print(f"MCMC Optimized LL: {optimized_ll:.2f}")

if __name__ == '__main__':
    unittest.main()
