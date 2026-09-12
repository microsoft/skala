# SkalaXC public C API.
#
# The extern "C" wrapper over SkalaXC::XCIntegrator. These sources are added
# directly to the `skalaxc` shared library (defined in ../CMakeLists.txt) rather
# than to an archive, so their skalaxc_* symbols survive --exclude-libs,ALL and
# remain the only exported entry points.
target_sources(skalaxc PRIVATE
  skalaxc_c_api.cxx
)
