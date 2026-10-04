// Header-only Boost.JSON: the library's out-of-line definitions.
//
// Compiled exactly once into each shared object that uses Boost.JSON:
// libcuda_hook.so (CMakeLists.txt) and foundry.ops (setup.py). Every other
// translation unit includes only <boost/json.hpp>. Including
// <boost/json/src.hpp> anywhere else in the same shared object defines the
// same non-inline symbols twice and fails at link time.
//
// libcuda_hook.so is LD_PRELOADed into the serving process, so this TU is
// compiled with -fvisibility=hidden there: the Boost.JSON definitions bind
// inside the hook and never interpose another library's libboost_json.
// foundry.ops is built with -fvisibility=hidden as a whole.

#ifdef BOOST_JSON_SOURCE
#error "boost/json/src.hpp was already included before csrc/boost_json_src.cpp"
#endif

#include <boost/json/src.hpp>
