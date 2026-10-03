"""NULL generic memcpy context needs real LOAD creation provenance + two VMM operands."""
import copy
import unittest
from test_memcpy_guard_legacy import MemcpyDriver
from test_bridge_cpu import bridge
from importlib import import_module
m = import_module(bridge.__package__ + '._qmd_research.graph_dag_diff')

WITNESS = {'provenance':'foundry_explicit_cuGraphAddMemcpyNode_ctx',
           'builder_graph':1000,'context':77,'device':0,'source_graph':101,'target_graph':102}


def fixture():
    driver=MemcpyDriver()
    driver.changes[2]={'srcDevice':0x1000,'dstDevice':0x5000}
    driver.attribute_changes[1,1]=driver.attribute_changes[2,1]=None
    driver.generic_changes[1]={'copyCtx':None}
    driver.generic_changes[2]={'copyCtx':None}
    return driver


class LoadCopyContext(unittest.TestCase):
    def report(self,d,witness=WITNESS):
        return m.compare_nonkernels(d,[1],[2],[1],memcpy_creation_context=witness)
    def test_owned_load_vmm_copy_passes_without_inventing_context(self):
        d=fixture();r=self.report(d)
        self.assertTrue(r['nonkernel_compatible'],r['nonkernel_rejection_reasons'])
        n=r['nonkernel_nodes'][0]
        self.assertIsNone(n['source']['copy_context'])
        self.assertEqual(n['source']['creation_context_witness'],WITNESS)
        self.assertEqual(d.retained,d.released)
    def test_default_compare_does_not_accept_null(self):
        self.assertFalse(self.report(fixture(),None)['nonkernel_compatible'])
    def test_missing_wrong_origin_builder_context_device_or_clone_rejects(self):
        for key,value in [('provenance','guessed_current'),('builder_graph',0),('context',0),
                          ('context',78),('device',1),('source_graph',0),('target_graph',0)]:
            w={**WITNESS,key:value}
            self.assertFalse(self.report(fixture(),w)['nonkernel_compatible'],(key,value))
    def test_null_generic_does_not_cover_non_vmm_operand(self):
        for node in (1,2):
            d=fixture();d.attribute_changes[node,1]=77
            self.assertFalse(self.report(d)['nonkernel_compatible'])
    def test_missing_vmm_backing_or_readwrite_rejects(self):
        for query in ('retain','properties','release','access','properties_exception'):
            d=fixture();d.vmm_errors[query]=1
            self.assertFalse(self.report(d)['nonkernel_compatible'],query)
    def test_wrong_device_and_managed_operands_reject(self):
        for attribute,value in [(9,1),(8,1),(13,0),(2,1),(7,0),(20,0)]:
            d=fixture();d.attribute_changes[2,attribute]=value
            self.assertFalse(self.report(d)['nonkernel_compatible'],attribute)
    def test_changed_mapping_or_allocation_identity_rejects(self):
        for attribute,value in [(7,999),(20,999),(19,0x1234),(18,8192)]:
            d=fixture();d.attribute_changes[2,attribute]=value
            self.assertFalse(self.report(d)['nonkernel_compatible'],attribute)
    def test_failed_current_context_query_rejects(self):
        d=fixture();d.query_errors[1,'current_context']=201
        self.assertFalse(self.report(d)['nonkernel_compatible'])
    def test_changed_witness_between_source_target_rejects(self):
        d=fixture();n=self.report(d)['nonkernel_nodes'][0]
        a,b=copy.deepcopy(n['source']),copy.deepcopy(n['target'])
        b['creation_context_witness']['builder_graph']=2222
        self.assertTrue(m._memcpy_rejections(a,b))
    def test_generic_context_transition_not_normalized(self):
        d=fixture();d.generic_changes[2]['copyCtx']=77
        self.assertFalse(self.report(d)['nonkernel_compatible'])


if __name__=='__main__':unittest.main()
