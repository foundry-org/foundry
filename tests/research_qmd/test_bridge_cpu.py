"""CPU-only failure-order tests; no torch/CUDA imports or GPU launches."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = '_foundry_bridge_cpu'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / 'python/foundry')]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    PACKAGE + '.research_qmd', ROOT / 'python/foundry/research_qmd.py')
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)

GOOD = {'aligned': True, 'nonkernel_compatible': True,
        'kernel_nodes_compared': 130, 'nonkernel_nodes_compared': 1,
        'source': {'node_count': 131}, 'device_updatable_nodes': 0,
        'changed_attribute_node_counts': {}}


class Context:
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False


class Driver:
    repair_version = 3
    repair_writes_enabled = True
    repair_capture_expect_memcpy = True
    def __init__(self, events):
        self.events = events
        self.next_source = 100
    def clone(self, graph):
        self.next_source += 1
        self.events.append(('clone', graph, self.next_source))
        return self.next_source
    def census(self, graph):
        return {'nodes': 131, 'node_types': {'0': 130, '7': 1}}
    def flags(self, graph):
        self.events.append(('flags', graph))
        return 4
    def instantiate(self, source, flags, **kwargs):
        self.events.append(('instantiate', source, flags, kwargs))
        return 200, 12
    def upload(self, executable, stream):
        self.events.append(('upload', executable, stream))
    def update(self, executable, target, **kwargs):
        self.events.append(('update', executable, target, kwargs))
        return {'success': True, 'api_result': 0, 'update_result': 0}
    def repair_counter_snapshot(self):
        return {'available': True}


class Graph:
    def __init__(self, events, member=2):
        self.events = events
        self.info = {'shared_graph': 10, 'template_exec': 20, 'graph_id': member,
                     'current_params_id': 1, 'has_rng': False, 'device': 0}
    def _research_info(self):
        return dict(self.info)
    def _research_rewrite(self):
        self.events.append(('archive_rewrite', self.info['graph_id']))
        self.info['current_params_id'] = self.info['graph_id']
    def _research_replay_exec(self, executable):
        self.events.append(('launch', executable))


class TestBridge(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.driver = Driver(self.events)
        self.graph = Graph(self.events)
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            device=lambda _: Context(),
            synchronize=lambda: self.events.append(('sync',)),
            current_stream=lambda: types.SimpleNamespace(cuda_stream=99)))
        self.patches = [patch.dict(sys.modules, {'torch': torch}),
                        patch.dict('os.environ', {'FOUNDRY_QMD_RESEARCH_MODE': 'repair'}),
                        patch.object(bridge, '_driver_instance', return_value=self.driver),
                        patch.object(bridge, 'diff_graphs', return_value=dict(GOOD)),
                        patch.object(bridge, '_receipt', side_effect=lambda x: x)]
        for p in self.patches:
            p.start()
        bridge._states.clear()
        bridge._owners.clear()
    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
    def test_register_pristine_before_real_archive_rewrite(self):
        value = bridge._prepare(self.graph)
        instantiate = next(x for x in self.events if x[0] == 'instantiate')
        self.assertEqual(instantiate[1:3], (101, 4))
        self.assertTrue(instantiate[3]['register'])
        self.assertEqual(instantiate[3]['expected_kernel_count'], 130)
        self.assertLess(self.events.index(instantiate), self.events.index(('archive_rewrite', 2)))
        self.assertEqual(value['candidate_exec'], 200)
        self.assertEqual(value['initial_source_graph'], 101)
        self.assertEqual(value['source_graph'], 102)
        self.assertEqual([x[1] for x in self.events if x[0] == 'update'], [200])
        self.assertNotIn(('update', 20), self.events)
    def test_same_member_has_no_sync_update_or_rewrite(self):
        bridge._prepare(self.graph)
        self.events.clear()
        bridge._prepare(self.graph)
        self.assertEqual(self.events, [])
    def test_return_to_initial_member_updates_same_exec(self):
        value = bridge._prepare(self.graph)
        self.graph.info['graph_id'] = 1
        self.events.clear()
        again = bridge._prepare(self.graph)
        self.assertIs(value, again)
        self.assertEqual(again['candidate_exec'], 200)
        self.assertEqual(again['current_member_id'], 1)
        self.assertEqual([x[0] for x in self.events],
                         ['sync', 'archive_rewrite', 'clone', 'update', 'upload', 'sync'])
    def test_unknown_attribute_transition_blocks_update(self):
        bridge._prepare(self.graph)
        self.graph.info['graph_id'] = 3
        self.events.clear()
        bad = {**GOOD, 'changed_attribute_node_counts': {'cluster_dimensions': 1}}
        with patch.object(bridge, 'diff_graphs', return_value=bad):
            with self.assertRaisesRegex(RuntimeError, 'strict full DAG'):
                bridge._prepare(self.graph)
        self.assertFalse(any(x[0] in ('update', 'upload', 'launch') for x in self.events))
    def test_public_guard_rejects_before_candidate_instantiation(self):
        with patch.object(bridge, 'diff_graphs', return_value={**GOOD, 'nonkernel_compatible': False}):
            with self.assertRaisesRegex(RuntimeError, 'pristine graph'):
                bridge._prepare(self.graph)
        self.assertFalse(any(x[0] in ('instantiate', 'update', 'upload', 'launch') for x in self.events))
        self.assertFalse(bridge._states)
    def test_update_failure_never_uploads_or_launches(self):
        with patch.object(self.driver, 'update', return_value={'success': False}):
            with self.assertRaisesRegex(RuntimeError, 'execUpdate failed'):
                bridge._prepare(self.graph)
        # Initial pristine upload is expected; no candidate update upload.
        self.assertEqual(sum(x[0] == 'upload' for x in self.events), 1)
        self.assertFalse(any(x[0] == 'launch' for x in self.events))
        self.assertEqual(bridge._states[10]['current_member_id'], 1)
    def test_external_builder_write_detected_before_launch(self):
        bridge._prepare(self.graph)
        self.events.clear()
        self.graph.info['current_params_id'] = 999
        with self.assertRaisesRegex(RuntimeError, 'External writer'):
            bridge._prepare(self.graph)
        self.assertEqual(self.events, [])
    def test_rng_graph_declined_before_cuda(self):
        self.graph.info['has_rng'] = True
        with self.assertRaisesRegex(ValueError, 'RNG'):
            bridge._prepare(self.graph)
        self.assertEqual(self.events, [])
    def test_state_is_json_serializable(self):
        import json
        bridge._prepare(self.graph)
        self.assertEqual(json.loads(json.dumps(bridge.state(self.graph)))['candidate_exec'], 200)
    def test_changed_device_updatable_or_partial_census_declined(self):
        self.assertFalse(bridge._guard({**GOOD, 'device_updatable_nodes': 1}))
        self.assertFalse(bridge._guard({**GOOD, 'kernel_nodes_compared': 129}))
        self.assertFalse(bridge._guard({**GOOD, 'aligned': False}))


if __name__ == '__main__':
    unittest.main()
