#include <pybind11/pybind11.h>
#include "binding_registry.h"

namespace py = pybind11;

PYBIND11_MODULE(saltro_py, m)
{
    m.doc() = "saltro python bindings";

    for (auto& binder : get_binders())
        binder(m);
}