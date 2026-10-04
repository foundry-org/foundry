#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
#
# Populate third_party/boost with the header-only Boost subset Foundry compiles
# against (Boost.JSON via boost/json/src.hpp, Boost.Unordered concurrent_flat_map,
# Boost.CRC, Boost.Format). Downloads a pinned Boost release, verifies its
# SHA-256, builds the `bcp` tool and copies the subset plus the license.
#
# Usage:  tools/release/vendor_boost.sh
#         BOOST_VERSION=1.92.0 BOOST_SHA256=<sha256 of boost_1_92_0.tar.bz2> tools/release/vendor_boost.sh
# Needs:  curl, tar (bzip2), a C++ compiler (for bcp and the link check).
# The SHA-256 of a release is published at
#   https://archives.boost.io/release/<ver>/source/boost_<ver_>.tar.bz2.json
#
# Commit the result (third_party/boost) before tagging a release; the release
# workflow refuses to build wheels without it.
set -euo pipefail

BOOST_VERSION="${BOOST_VERSION:-1.90.0}"
if [[ -z "${BOOST_SHA256:-}" ]]; then
    if [[ "$BOOST_VERSION" == "1.90.0" ]]; then
        BOOST_SHA256="49551aff3b22cbc5c5a9ed3dbc92f0e23ea50a0f7325b0d198b705e8ee3fc305"
    else
        echo "vendor_boost: set BOOST_SHA256 for Boost $BOOST_VERSION" >&2
        exit 1
    fi
fi

# Entry headers; bcp copies everything they include, transitively.
HEADERS=(
    boost/version.hpp
    boost/json.hpp
    boost/json/src.hpp
    boost/unordered/concurrent_flat_map.hpp
    boost/unordered/concurrent_flat_map_fwd.hpp
    boost/crc.hpp
    boost/format.hpp
)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$ROOT/third_party/boost"
UNDERSCORED="${BOOST_VERSION//./_}"
TARBALL="boost_${UNDERSCORED}.tar.bz2"
URL="https://archives.boost.io/release/${BOOST_VERSION}/source/${TARBALL}"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
CXX="${CXX:-c++}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "vendor_boost: downloading $URL"
curl -fL --retry 3 -o "$WORK/$TARBALL" "$URL"

if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$WORK/$TARBALL" | cut -d' ' -f1)"
else
    actual="$(shasum -a 256 "$WORK/$TARBALL" | cut -d' ' -f1)"
fi
if [[ "$actual" != "$BOOST_SHA256" ]]; then
    echo "vendor_boost: SHA-256 mismatch for $TARBALL: expected $BOOST_SHA256, got $actual" >&2
    exit 1
fi

echo "vendor_boost: extracting"
tar -xjf "$WORK/$TARBALL" -C "$WORK"
SRC="$WORK/boost_${UNDERSCORED}"

echo "vendor_boost: building bcp"
(
    cd "$SRC"
    ./bootstrap.sh >"$WORK/bootstrap.log" 2>&1 || {
        cat "$WORK/bootstrap.log" >&2
        exit 1
    }
    ./b2 -j"$JOBS" tools/bcp >"$WORK/b2.log" 2>&1 || {
        tail -50 "$WORK/b2.log" >&2
        exit 1
    }
)
BCP="$SRC/dist/bin/bcp"
[[ -x "$BCP" ]] || {
    echo "vendor_boost: bcp was not built at $BCP" >&2
    exit 1
}

echo "vendor_boost: copying the subset for ${HEADERS[*]}"
mkdir -p "$WORK/subset"
(cd "$SRC" && "$BCP" --boost="$SRC" "${HEADERS[@]}" "$WORK/subset" >"$WORK/bcp.log")

# Link check: everything Foundry uses must compile and link with headers only.
cat >"$WORK/check.cpp" <<'CPP'
#include <boost/json/src.hpp>
#include <boost/unordered/concurrent_flat_map.hpp>
#include <boost/crc.hpp>
#include <boost/format.hpp>
#include <cstdint>
#include <string>
int main() {
  boost::json::value v = boost::json::parse(R"({"a":[1,2.5,"x"]})");
  boost::json::object o;
  o["k"] = boost::json::serialize(v);
  boost::concurrent_flat_map<std::uint64_t, int> m;
  m.emplace(1, 2);
  boost::crc_optimal<64, 0x42F0E1EBA9EA3693ULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
                     true, true> crc;
  crc.process_bytes("foundry", 7);
  std::string s = boost::str(boost::format("%016llx") % (unsigned long long)crc.checksum());
  return (m.size() == 1 && !s.empty() && o.size() == 1) ? 0 : 1;
}
CPP
if command -v "$CXX" >/dev/null 2>&1; then
    echo "vendor_boost: header-only link check with $CXX"
    "$CXX" -std=c++17 -O0 -isystem "$WORK/subset" "$WORK/check.cpp" -o "$WORK/check" -pthread
    "$WORK/check"
else
    echo "vendor_boost: WARNING: no C++ compiler ($CXX); skipped the link check" >&2
fi

rm -rf "$DEST/boost"
mkdir -p "$DEST"
cp -R "$WORK/subset/boost" "$DEST/boost"
cp "$SRC/LICENSE_1_0.txt" "$DEST/LICENSE_1_0.txt"
SIZE="$(du -sh "$DEST/boost" | cut -f1)"
NFILES="$(find "$DEST/boost" -type f | wc -l | tr -d ' ')"

cat >"$DEST/README.md" <<MD
# Vendored Boost headers

Boost ${BOOST_VERSION} (\`${TARBALL}\`, SHA-256 \`${BOOST_SHA256}\`), header
subset produced by \`bcp\` for:

$(printf -- '- `%s`\n' "${HEADERS[@]}")

${NFILES} files, ${SIZE}. Foundry uses Boost header-only (Boost.JSON through
\`csrc/boost_json_src.cpp\`), so nothing here is compiled on its own and no
Boost shared library is linked. Distributed under the Boost Software License
1.0 (\`LICENSE_1_0.txt\`).

Regenerate (and commit the result) with:

\`\`\`bash
tools/release/vendor_boost.sh
# or another release:
BOOST_VERSION=<x.y.z> BOOST_SHA256=<sha256> tools/release/vendor_boost.sh
\`\`\`

\`setup.py\` and \`CMakeLists.txt\` use this directory when
\`third_party/boost/boost/version.hpp\` exists and otherwise fall back to a
system Boost (1.83 or newer). Release wheels are built only from this copy.
MD

echo "vendor_boost: wrote $DEST/boost ($NFILES files, $SIZE)"
if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    ignored="$(cd "$ROOT" && find third_party/boost -type f | git check-ignore --stdin || true)"
    if [[ -n "$ignored" ]]; then
        echo "vendor_boost: WARNING: .gitignore matches some vendored files; add them with" >&2
        echo "  git add -f third_party/boost" >&2
        echo "$ignored" | head -20 >&2
    fi
    echo "vendor_boost: next: git add third_party/boost && git commit -m '[build] Vendor Boost ${BOOST_VERSION} headers'"
fi
