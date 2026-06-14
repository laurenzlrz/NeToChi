"""Tests for input_generator utility helpers."""

from __future__ import annotations

import unittest

import networkx as nx

from netochi.input_generator.utils import nx_to_gt


class TestNxToGt(unittest.TestCase):
    def test_converts_directed_graph(self) -> None:
        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from([(0, 1), (1, 2)])

        gt_graph = nx_to_gt(nx_graph)

        self.assertTrue(gt_graph.is_directed())
        self.assertEqual(gt_graph.num_vertices(), 3)
        self.assertEqual(gt_graph.num_edges(), 2)

    def test_converts_undirected_graph(self) -> None:
        nx_graph = nx.DiGraph()
        nx_graph.add_edge(0, 1)

        gt_graph = nx_to_gt(nx_graph)

        self.assertFalse(gt_graph.is_directed())
        self.assertEqual(gt_graph.num_vertices(), 2)
        self.assertEqual(gt_graph.num_edges(), 1)


if __name__ == "__main__":
    unittest.main()
