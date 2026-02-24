#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <saltro/pybind/disturbances/geometryconfig.h>

namespace py = pybind11;
using namespace saltro::disturbances;

void bind_geometryconfig(py::module_& m) {
    // Bind GeometryFace struct
    py::class_<GeometryFace>(m, "GeometryFace")
        .def(py::init<>(),
             "Construct a GeometryFace with all values initialized to zero")
        .def(py::init<double, const GeometryFace::Vec3&, const GeometryFace::Vec3&,
                      double, double, double, double>(),
             py::arg("area"),
             py::arg("centroid"),
             py::arg("normal"),
             py::arg("eta_s") = 0.0,
             py::arg("eta_d") = 0.0,
             py::arg("eta_a") = 0.0,
             py::arg("CD") = 0.0,
             R"doc(
Construct a GeometryFace with specified parameters.

Represents a single geometric face/surface element of a satellite with
geometric, optical, and aerodynamic properties.

Parameters
----------
area : float
    Surface area [m²]
centroid : ndarray (3,)
    Centroid position in body frame [m]
normal : ndarray (3,)
    Surface normal unit vector in body frame
eta_s : float, optional
    Specular reflection coefficient [0-1] (default: 0.0)
eta_d : float, optional
    Diffuse reflection coefficient [0-1] (default: 0.0)
eta_a : float, optional
    Absorptivity coefficient [0-1] (default: 0.0)
CD : float, optional
    Drag coefficient (default: 0.0)
)doc")
        .def_readwrite("area", &GeometryFace::area,
                       "Surface area [m²]")
        .def_readwrite("centroid", &GeometryFace::centroid,
                       "Centroid position in body frame [m]")
        .def_readwrite("normal", &GeometryFace::normal,
                       "Surface normal unit vector in body frame")
        .def_readwrite("eta_s", &GeometryFace::eta_s,
                       "Specular reflection coefficient [0-1]")
        .def_readwrite("eta_d", &GeometryFace::eta_d,
                       "Diffuse reflection coefficient [0-1]")
        .def_readwrite("eta_a", &GeometryFace::eta_a,
                       "Absorptivity coefficient [0-1]")
        .def_readwrite("CD", &GeometryFace::CD,
                       "Drag coefficient");

    // Bind GeometryConfig class
    py::class_<GeometryConfig>(m, "GeometryConfig")
        .def(py::init<>(),
             R"doc(
Construct an empty GeometryConfig.

GeometryConfig defines the geometric and aerodynamic properties of
the satellite's external surfaces. This configuration is used by
disturbance models (e.g., drag, solar radiation pressure) to compute
forces and torques.
)doc")
        .def("addFace", &GeometryConfig::addFace,
             py::arg("face"),
             R"doc(
Add a face to the geometry configuration.

Parameters
----------
face : GeometryFace
    The GeometryFace to add

Returns
-------
bool
    True if face was added successfully, False if maximum capacity reached

Notes
-----
Maximum number of faces is limited to MAX_NUM_GEOMETRY_FACES.
)doc")
        .def_property_readonly("numFaces", &GeometryConfig::numFaces,
                               R"doc(
Get the number of faces currently configured.

Returns
-------
int
    Number of faces
)doc")
        .def("getFace",
             py::overload_cast<size_t>(&GeometryConfig::getFace, py::const_),
             py::arg("index"),
             py::return_value_policy::reference_internal,
             R"doc(
Get a reference to a specific face.

Parameters
----------
index : int
    Face index [0, numFaces)

Returns
-------
GeometryFace
    Reference to the face

Raises
------
IndexError
    If index is out of range
)doc")
        .def("clear", &GeometryConfig::clear,
             "Clear all faces from the configuration")
        .def_static("maxFaces", &GeometryConfig::maxFaces,
                    R"doc(
Get the maximum number of faces that can be stored.

Returns
-------
int
    Maximum capacity (MAX_NUM_GEOMETRY_FACES)
)doc")
        .def("__len__", &GeometryConfig::numFaces,
             "Return the number of faces (supports len(config))")
        .def("__getitem__",
             [](const GeometryConfig& config, size_t index) -> const GeometryFace& {
                 if (index >= config.numFaces()) {
                     throw py::index_error("Face index out of range");
                 }
                 return config.getFace(index);
             },
             py::return_value_policy::reference_internal,
             py::arg("index"),
             "Access face by index (supports config[i])")
        .def("__iter__",
             [](const GeometryConfig& config) {
                 return py::make_iterator(
                     config.faces().begin(),
                     config.faces().begin() + config.numFaces()
                 );
             },
             py::keep_alive<0, 1>(),
             "Iterate over all faces (supports for face in config)");
}
