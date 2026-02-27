#include "tensor_py.h"

namespace py = pybind11;

void bind_tensor(py::module_& m) {
    // No explicit bindings needed - the type casters work automatically
    // This function exists to ensure tensor_py.cpp is included in the build
    
    // Optional: Add a docstring for the module
    m.doc() = "Type conversions for Tensor3 objects (handled automatically by type casters)";
}
