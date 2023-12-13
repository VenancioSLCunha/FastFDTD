#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "solver.cu"

namespace py = pybind11;

// Binding for the FDTD class
void bind_FDTD(py::module &m) {
    py::class_<FastFDTD>(m, "FastFDTD")
        .def(py::init<>())

        // Setters
        .def("setGridSize", &FastFDTD::setGridSize)
        .def("setSimulationParameters", &FastFDTD::setSimulationParameters)
        .def("setSimulationParameters", &FastFDTD::setSimulationParameters, py::arg("dx"), py::arg("dt"), py::arg("di"), py::arg("dj"), py::arg("omega"), py::arg("tau"))
        .def("setSource", &FastFDTD::setSource)
        .def("setDetector", &FastFDTD::setDetector)
        .def("setObject", &FastFDTD::setObject)
        .def("setBoundary", &FastFDTD::setBoundary)

        // executar e get dos resultados.
        .def("run", &FastFDTD::runSim, py::arg("N"))
        .def("getResult", &FastFDTD::getResult);
}

PYBIND11_MODULE(fast_fdtd, m) {

    bind_FDTD(m)
}