"""CPU-only public-ABI and fail-closed memcpy compatibility tests."""
import ctypes as C
import types
import unittest

from test_bridge_cpu import bridge
from _foundry_bridge_cpu._qmd_research.graph_dag_diff import (GraphNodeParams, Memcpy3D, MemcpyNodeParams, MemLocation, MemAllocationProp,
                            compare_nonkernels)


class MemcpyDriver:
    def __init__(self):
        self.changes, self.attribute_changes, self.contexts = {}, {}, {}
        self.generic_changes, self.query_errors = {}, {}
        self.node = None
        self.range_result = 0
        self.vmm_errors, self.vmm_props, self.retained, self.released = {}, {}, [], []
        self.vmm_access = 3
        self.lib = types.SimpleNamespace(
            cuGraphMemsetNodeGetParams=lambda *_: 1,
            cuGraphEventRecordNodeGetEvent=lambda *_: 1,
            cuGraphEventWaitNodeGetEvent=lambda *_: 1,
            cuGraphMemcpyNodeGetParams=self.params,
            cuGraphNodeGetParams=self.generic,
            cuPointerGetAttribute=self.attribute,
            cuCtxGetCurrent=self.context,
            cuCtxGetDevice=self.device,
            cuMemGetAddressRange_v2=self.address_range,
            cuMemRetainAllocationHandle=self.retain,
            cuMemGetAllocationPropertiesFromHandle=self.properties,
            cuMemRelease=self.release,
            cuMemGetAccess=self.access,
        )

    def _bind(self, name, arguments):
        getattr(self.lib, name)

    def check(self, rc, operation):
        if rc:
            raise RuntimeError(operation)

    def fill(self, node, params):
        params.srcMemoryType = params.dstMemoryType = 2
        params.srcDevice, params.dstDevice = (0x1000, 0x5000) if node == 1 else (0x2000, 0x6000)
        params.WidthInBytes = 64 if node == 1 else 128
        params.Height = params.Depth = 1
        for key, value in self.changes.get(node, {}).items():
            setattr(params, key, value)

    def params(self, node, pointer):
        self.node = node
        self.fill(node, pointer._obj)
        return self.query_errors.get((node, "params"), 0)

    def generic(self, node, pointer):
        self.node = node
        value = pointer._obj
        value.type = 1
        value.memcpy.copyCtx = self.contexts.get(node, 77)
        self.fill(node, value.memcpy.copyParams)
        for key, changed in self.generic_changes.get(node, {}).items():
            if key == "type":
                value.type = changed
            else:
                setattr(value.memcpy, key, changed)
        return self.query_errors.get((node, "generic"), 0)

    def context(self, pointer):
        pointer._obj.value = self.contexts.get(self.node, 77)
        return self.query_errors.get((self.node, "current_context"), 0)

    def device(self, pointer):
        pointer._obj.value = 0
        return 0

    def attribute(self, pointer, attribute, address):
        values = {1: self.contexts.get(self.node, 77), 2: 2,
                  7: address // 4096, 8: 0, 9: 0, 11: 0x1000,
                  12: 0x100000, 13: 1, 18: 4096, 19: address & ~4095,
                  20: address // 4096}
        value = self.attribute_changes.get((self.node, attribute), values[attribute])
        if isinstance(value, tuple):
            rc, value = value
        else:
            rc = 0
        pointer._obj.value = value
        return rc

    def address_range(self, base, size, address):
        base._obj.value, size._obj.value = address & ~4095, 4096
        return self.range_result

    def retain(self, handle, address):
        handle._obj.value = address & ~4095
        rc = self.vmm_errors.get("retain", 0)
        if not rc:
            self.retained.append(handle._obj.value)
        return rc

    def properties(self, pointer, handle):
        if self.vmm_errors.get("properties_exception"):
            raise RuntimeError("property getter error")
        prop = pointer._obj
        prop.type, prop.location.type, prop.location.id = 1, 1, 0
        for key, value in self.vmm_props.items():
            owner, name = (prop.location, key[9:]) if key.startswith("location_") else (prop, key)
            setattr(owner, name, value)
        return self.vmm_errors.get("properties", 0)

    def release(self, handle):
        self.released.append(handle)
        return self.vmm_errors.get("release", 0)

    def access(self, flags, location, address):
        flags._obj.value = self.vmm_access
        return self.vmm_errors.get("access", 0)


class MemcpyGuardTests(unittest.TestCase):
    def report(self, driver):
        return compare_nonkernels(driver, [1], [2], [1])

    def assert_rejects(self, driver, reason):
        report = self.report(driver)
        self.assertFalse(report["nonkernel_compatible"])
        self.assertIn(reason, "\n".join(report["nonkernel_rejection_reasons"]))
        return report

    def test_public_struct_layout_64_bit(self):
        self.assertEqual(C.sizeof(Memcpy3D), 200)
        self.assertEqual(C.sizeof(MemcpyNodeParams), 216)
        self.assertEqual(C.sizeof(GraphNodeParams), 256)
        self.assertEqual(Memcpy3D.dstXInBytes.offset, 88)
        self.assertEqual(Memcpy3D.WidthInBytes.offset, 176)
        self.assertEqual(MemcpyNodeParams.copyParams.offset, 16)
        self.assertEqual(GraphNodeParams.memcpy.offset, 16)
        self.assertEqual(C.sizeof(MemLocation), 8)
        self.assertEqual(C.sizeof(MemAllocationProp), 32)

    def vmm_driver(self):
        driver = MemcpyDriver()
        # The VMM extension requires the exact same retained mappings across
        # source and target, but permits byte offsets and width changes.
        driver.changes[2] = {"srcDevice": 0x1000, "dstDevice": 0x5000}
        driver.attribute_changes[1, 1] = driver.attribute_changes[2, 1] = None
        return driver

    def test_null_context_requires_proven_same_vmm_allocation_and_access(self):
        driver = self.vmm_driver()
        driver.changes[2]["srcXInBytes"] = 16
        report = self.report(driver)
        self.assertTrue(report["nonkernel_compatible"], report["nonkernel_rejection_reasons"])
        self.assertEqual(driver.retained, driver.released)
        self.assertEqual(len(driver.retained), 4)
        vmm = report["nonkernel_nodes"][0]["source"]["operands"]["src"]["vmm"]
        self.assertEqual(vmm["properties"]["location_type"], 1)
        self.assertEqual(vmm["access"][0]["flags"], 3)

    def test_null_context_vmm_api_failures_close_gate_and_balance_retain(self):
        for query in ("retain", "properties", "release", "access", "properties_exception"):
            with self.subTest(query=query):
                driver = self.vmm_driver()
                driver.vmm_errors[query] = 1
                self.assert_rejects(driver, "VMM")
                self.assertEqual(driver.retained, driver.released)

    def test_vmm_foreign_device_host_backing_or_restricted_access_rejected(self):
        for field, value in (("type", 2), ("location_type", 2), ("location_id", 1)):
            with self.subTest(field=field):
                driver = self.vmm_driver()
                driver.vmm_props[field] = value
                self.assert_rejects(driver, "VMM backing")
        driver = self.vmm_driver()
        driver.vmm_access = 1
        self.assert_rejects(driver, "READWRITE access")

    def test_vmm_backing_mapping_and_identity_changes_rejected(self):
        for attribute, value in ((7, 12345), (20, 12345), (18, 8192)):
            with self.subTest(attribute=attribute):
                driver = self.vmm_driver()
                driver.attribute_changes[2, attribute] = value
                self.assert_rejects(driver, "VMM")
        driver = self.vmm_driver()
        driver.changes[2]["srcDevice"] = 0x2000
        self.assert_rejects(driver, "VMM")
        driver = self.vmm_driver()
        driver.attribute_changes[2, 20] = (1, 0)
        self.assert_rejects(driver, "identity is unavailable")

    def test_address_size_and_allocation_id_changes_permitted_within_same_context(self):
        report = self.report(MemcpyDriver())
        self.assertTrue(report["nonkernel_compatible"])
        entry = report["nonkernel_nodes"][0]
        self.assertFalse(entry["parameters_identical"])
        self.assertTrue(entry["memcpy_compatible"])
        self.assertNotEqual(entry["source"]["operands"]["src"]["attributes"]["buffer_id"],
                            entry["target"]["operands"]["src"]["attributes"]["buffer_id"])
        self.assertFalse(report["memcpy_lifetime_or_mapping_stability_proven_by_public_getters"])
        self.assertTrue(any("Retain" in s for s in report["memcpy_caller_obligations"]))
        self.assertTrue(any("wait" in s for s in report["memcpy_caller_obligations"]))

    def test_one_dimensional_byte_offsets_and_tight_pitches_supported(self):
        driver = MemcpyDriver()
        driver.changes[2] = {"srcXInBytes": 8, "srcPitch": 136, "dstPitch": 128, "srcHeight": 1}
        self.assertTrue(self.report(driver)["nonkernel_compatible"])

    def test_host_array_unified_or_unknown_operands_rejected(self):
        for side in ("src", "dst"):
            for kind in (1, 3, 4, 99):
                with self.subTest(side=side, kind=kind):
                    driver = MemcpyDriver()
                    driver.changes[2] = {side + "MemoryType": kind}
                    self.assert_rejects(driver, "not CU_MEMORYTYPE_DEVICE")

    def test_multidimensional_zero_or_noncanonical_copies_rejected(self):
        for field, value, reason in (
            ("Height", 2, "1D"), ("Depth", 2, "1D"), ("WidthInBytes", 0, "nonzero"),
            ("srcY", 1, "linear"), ("dstZ", 1, "linear"), ("srcLOD", 1, "linear"),
            ("dstHeight", 2, "linear"), ("srcPitch", 1024, "linear"),
            ("srcHost", 4, "linear"), ("dstArray", 4, "linear"),
            ("srcDevice", 0, "linear"), ("reserved0", 1, "reserved pointers"),
        ):
            with self.subTest(field=field):
                driver = MemcpyDriver()
                driver.changes[2] = {field: value}
                self.assert_rejects(driver, reason)

    def test_generic_flags_type_and_disagreement_rejected(self):
        for field, value in (("flags", 1), ("reserved", 1), ("type", 2)):
            with self.subTest(field=field):
                driver = MemcpyDriver()
                driver.generic_changes[2] = {field: value}
                self.assert_rejects(driver, "type/flags/reserved")
        driver = MemcpyDriver()
        original = driver.lib.cuGraphNodeGetParams
        def inconsistent(node, pointer):
            rc = original(node, pointer)
            pointer._obj.memcpy.copyParams.WidthInBytes += 1
            return rc
        driver.lib.cuGraphNodeGetParams = inconsistent
        self.assert_rejects(driver, "parameters disagree")

    def test_context_change_or_unavailable_context_rejected(self):
        driver = MemcpyDriver()
        driver.contexts[2] = 88
        self.assert_rejects(driver, "context changed")
        for changed in (0, (1, 0), 88):
            with self.subTest(attribute=changed):
                driver = MemcpyDriver()
                driver.attribute_changes[2, 1] = changed
                self.assert_rejects(driver, "allocation context")
        driver = MemcpyDriver()
        driver.generic_changes[2] = {"copyCtx": 88}
        self.assert_rejects(driver, "not the current context")

    def test_other_device_managed_unmapped_or_type_mismatch_rejected(self):
        for attribute, changed, name in ((9, 1, "device_ordinal"), (8, 1, "is_managed"),
                                         (13, 0, "mapped"), (2, 1, "memory_type")):
            with self.subTest(attribute=attribute):
                driver = MemcpyDriver()
                driver.attribute_changes[2, attribute] = changed
                self.assert_rejects(driver, "allocation " + name)

    def test_bounds_are_actual_mapping_not_larger_reserved_va(self):
        driver = MemcpyDriver()
        driver.changes[2] = {"srcXInBytes": 4090}
        report = self.assert_rejects(driver, "full copy range")
        src = report["nonkernel_nodes"][0]["target"]["operands"]["src"]
        self.assertGreater(src["attributes"]["reserved_size"]["value"], src["mapped_range"]["size"])

    def test_mapping_attributes_can_fallback_to_public_mapped_range_query(self):
        driver = MemcpyDriver()
        driver.attribute_changes[2, 18] = driver.attribute_changes[2, 19] = (1, 0)
        report = self.report(driver)
        self.assertTrue(report["nonkernel_compatible"])
        self.assertEqual(report["nonkernel_nodes"][0]["target"]["operands"]["src"]["mapped_range"]["method"],
                         "cuMemGetAddressRange_v2")
        driver.range_result = 1
        self.assert_rejects(driver, "full copy range")

    def test_zero_mapping_overflow_and_overlapping_copy_rejected(self):
        driver = MemcpyDriver()
        driver.attribute_changes[2, 18] = 0
        self.assert_rejects(driver, "full copy range")
        driver = MemcpyDriver()
        driver.changes[2] = {"srcDevice": 2**64 - 8, "srcXInBytes": 16}
        self.assert_rejects(driver, "full copy range")
        driver = MemcpyDriver()
        driver.changes[2] = {"dstDevice": 0x2008}
        self.assert_rejects(driver, "ranges overlap")

    def test_getter_error_or_missing_new_generic_getter_fails_closed(self):
        for query in ("params", "generic"):
            with self.subTest(query=query):
                driver = MemcpyDriver()
                driver.query_errors[2, query] = 1
                self.assert_rejects(driver, "public parameter query failed")
        driver = MemcpyDriver()
        del driver.lib.cuGraphNodeGetParams
        self.assert_rejects(driver, "parameters unavailable")


if __name__ == "__main__":
    unittest.main()
