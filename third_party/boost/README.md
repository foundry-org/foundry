# Vendored Boost headers (not yet populated)

This directory holds the header-only Boost subset Foundry compiles against:
Boost.JSON (compiled header-only through `csrc/boost_json_src.cpp`),
`boost::concurrent_flat_map`, Boost.CRC and Boost.Format. No Boost shared
library is linked by `libcuda_hook.so` or `foundry.ops`.

It is populated by a script, not by hand:

```bash
tools/release/vendor_boost.sh   # pinned Boost 1.90.0, SHA-256 checked, bcp subset
git add third_party/boost
```

The script downloads the pinned release, builds `bcp`, copies the subset for
`boost/json/src.hpp`, `boost/unordered/concurrent_flat_map.hpp`,
`boost/crc.hpp` and `boost/format.hpp` into `third_party/boost/boost`, adds
the Boost Software License (`LICENSE_1_0.txt`), runs a header-only link check
and rewrites this README with the version and file count.

**Run it and commit the result before cutting a release.** The release
workflow (`.github/workflows/release.yml`) fails when
`third_party/boost/boost/version.hpp` is missing, so every wheel is built
against the same Boost.

Until then, `setup.py` and `CMakeLists.txt` fall back to a system Boost
(1.83 or newer, headers only): `FOUNDRY_BOOST_INCLUDE_DIR`,
`BOOST_INCLUDEDIR`, `$BOOST_ROOT/include`, `$CONDA_PREFIX/include`,
`/usr/local/include`, `/usr/include`, in that order.
