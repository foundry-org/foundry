# Vendored Boost headers

Boost 1.90.0 (`boost_1_90_0.tar.bz2`, SHA-256 `49551aff3b22cbc5c5a9ed3dbc92f0e23ea50a0f7325b0d198b705e8ee3fc305`), header
subset produced by `bcp` for:

- `boost/version.hpp`
- `boost/json.hpp`
- `boost/json/src.hpp`
- `boost/unordered/concurrent_flat_map.hpp`
- `boost/unordered/concurrent_flat_map_fwd.hpp`
- `boost/crc.hpp`
- `boost/format.hpp`

830 files, 11M. Foundry uses Boost header-only (Boost.JSON through
`csrc/boost_json_src.cpp`), so nothing here is compiled on its own and no
Boost shared library is linked. Distributed under the Boost Software License
1.0 (`LICENSE_1_0.txt`).

Regenerate (and commit the result) with:

```bash
tools/release/vendor_boost.sh
# or another release:
BOOST_VERSION=<x.y.z> BOOST_SHA256=<sha256> tools/release/vendor_boost.sh
```

`setup.py` and `CMakeLists.txt` use this directory when
`third_party/boost/boost/version.hpp` exists and otherwise fall back to a
system Boost (1.83 or newer). Release wheels are built only from this copy.
