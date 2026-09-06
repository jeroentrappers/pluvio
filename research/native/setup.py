"""Build the native kernels:  python setup.py build_ext --inplace

Produces `pluvio_native<abi>.so` next to this file; `model.motion` and
`tools.radar_single_site` import it opportunistically (PLUVIO_NATIVE=0
disables) and fall back to the pure-Python reference when it is absent, so a
box without a compiler keeps working.
"""

from setuptools import Extension, setup
import pybind11

setup(
    name="pluvio_native",
    version="0.1.0",
    ext_modules=[
        Extension(
            "pluvio_native",
            sources=["src/pluvio_native.cpp"],
            include_dirs=[pybind11.get_include()],
            # -ffp-contract=off keeps the arithmetic bit-identical to numpy's
            # (no FMA contraction); the equivalence tests assert exactly that.
            extra_compile_args=["-O3", "-std=c++17", "-fvisibility=hidden", "-ffp-contract=off"],
            extra_link_args=["-pthread"],
            language="c++",
        )
    ],
)
