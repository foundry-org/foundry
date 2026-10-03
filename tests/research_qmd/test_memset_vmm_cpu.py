"""Counterexamples for the narrow archive-LOAD memset VMM extension."""
import copy
import unittest
from test_bridge_cpu import bridge
from importlib import import_module
m = import_module(bridge.__package__ + '._qmd_research.graph_dag_diff')


def fixture():
    attrs = {name: {'api_result': 0, 'value': value} for name, value in {
        'context': None, 'memory_type': 2, 'is_managed': 0, 'mapped': 1,
        'device_ordinal': 0, 'buffer_id': 8, 'memory_block_id': 9}.items()}
    return {'dst': 4096, 'width': 4, 'elementSize': 1, 'height': 1, 'pitch': 0, 'value': 0,
        'allocation_context': {'api_result': 0, 'context': None},
        'vmm_proof': {'current_context': {'api_result': 0, 'value': 32},
            'current_device': {'api_result': 0, 'value': 0}, 'byte_width': 4,
            'operand': {'effective_address': 4096, 'attributes': attrs,
                'mapped_range': {'api_result': 0, 'base': 4096, 'size': 65536},
                'vmm': {'retain_api_result': 0, 'properties_api_result': 0, 'release_api_result': 0,
                    'allocation_handle': 40, 'properties': {'type': 1, 'location_type': 1,
                        'location_id': 0, 'win32_metadata': None, 'usage': 0, 'reserved': [0,0,0,0]},
                    'access': [{'address': 4096, 'api_result': 0, 'flags': 3},
                               {'address': 4099, 'api_result': 0, 'flags': 3}]}}}}


class MemsetVmmTest(unittest.TestCase):
    def test_complete_same_backing_passes(self):
        a = fixture()
        self.assertEqual(m._memset_rejections(a, copy.deepcopy(a)), [])
    def test_null_context_without_proof_rejects(self):
        a = fixture(); del a['vmm_proof']
        self.assertTrue(m._memset_rejections(a, a))
    def test_exact_nonnull_context_policy_preserved(self):
        a = fixture(); del a['vmm_proof']; a['allocation_context']['context'] = 32
        self.assertEqual(m._memset_rejections(a, copy.deepcopy(a)), [])
    def test_parameter_or_backing_change_rejects(self):
        for key in ['dst','width','elementSize','height','pitch','value']:
            a = fixture(); b = copy.deepcopy(a); b[key] += 1
            self.assertTrue(m._memset_rejections(a, b), key)
        for key in ['allocation_handle','properties','access']:
            a = fixture(); b = copy.deepcopy(a); b['vmm_proof']['operand']['vmm'][key] = None
            self.assertTrue(m._memset_rejections(a, b), key)
    def test_invalid_retain_properties_release_rejects(self):
        for key in ['retain_api_result','properties_api_result','release_api_result']:
            a = fixture(); a['vmm_proof']['operand']['vmm'][key] = 400
            self.assertTrue(m._memset_rejections(a, a), key)
    def test_missing_mapped_bounds_or_out_of_bounds_rejects(self):
        for change in [{'api_result':400}, {'base':4097}, {'size':3}, {'base':2**64-2,'size':8}]:
            a = fixture(); a['vmm_proof']['operand']['mapped_range'].update(change)
            self.assertTrue(m._memset_rejections(a, a), change)
    def test_no_reserved_va_fallback(self):
        a = fixture(); a['vmm_proof']['operand']['mapped_range'] = {}
        a['vmm_proof']['operand']['reserved_range'] = {'base':4096,'size':65536}
        self.assertTrue(m._memset_rejections(a,a))
    def test_unmapped_host_managed_wrong_device_rejects(self):
        for key,value in [('memory_type',1),('is_managed',1),('mapped',0),('device_ordinal',1),('context',17),('buffer_id',0),('memory_block_id',0)]:
            a = fixture(); a['vmm_proof']['operand']['attributes'][key]['value'] = value
            self.assertTrue(m._memset_rejections(a,a),key)
    def test_missing_or_read_only_endpoint_rejects(self):
        a = fixture(); a['vmm_proof']['operand']['vmm']['access'].pop()
        self.assertTrue(m._memset_rejections(a,a))
        a = fixture(); a['vmm_proof']['operand']['vmm']['access'][0]['flags'] = 1
        self.assertTrue(m._memset_rejections(a,a))
    def test_2d_pitched_or_bad_width_rejects(self):
        for key,value in [('height',2),('pitch',4),('width',0),('elementSize',8)]:
            a=fixture();a[key]=value
            self.assertTrue(m._memset_rejections(a,a),key)
    def test_missing_current_context_device_rejects(self):
        for key in ['current_context','current_device']:
            a=fixture();a['vmm_proof'][key]['api_result']=201
            self.assertTrue(m._memset_rejections(a,a),key)
    def test_non_device_vmm_backing_rejects(self):
        for key,value in [('type',0),('location_type',2),('location_id',1),('usage',1),('reserved',[1,0,0,0])]:
            a=fixture();a['vmm_proof']['operand']['vmm']['properties'][key]=value
            self.assertTrue(m._memset_rejections(a,a),key)


if __name__ == '__main__':
    unittest.main()
