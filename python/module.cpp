#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_compute_orbit(py::module_&);
void bind_compute_magnetic_dipole(py::module_&);

PYBIND11_MODULE(saltro_py, m)
{
    bind_compute_orbit(m);
    bind_compute_magnetic_dipole(m);
}