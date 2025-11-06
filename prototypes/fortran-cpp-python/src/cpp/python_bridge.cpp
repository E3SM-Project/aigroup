#include "python_bridge.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

// Single global: the Python interpreter
static std::unique_ptr<py::scoped_interpreter> interpreter;

extern "C" {

int initialize_python() {
    std::cout << "[C++] Initializing Python interpreter..." << std::endl;

    try {
        interpreter = std::make_unique<py::scoped_interpreter>();

        py::module_ sys = py::module_::import("sys");

        std::string source_file = __FILE__;
        auto this_dir = source_file.substr(0, source_file.find_last_of("/\\"));

        std::cout << "[C++] Adding to sys.path: " << this_dir + "/../python" << std::endl;
        sys.attr("path").attr("insert")(0, this_dir + "/../python");

        py::module_::import("python_main").attr("initialize")();

        std::cout << "[C++] Python initialized" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[C++] Error: " << e.what() << std::endl;
        return -1;
    }
}

int call_main_python(const float* data, int size, float* result) {
    std::cout << "[C++] Calling call_main_python..." << std::endl;

    try {
        py::module_ main = py::module_::import("python_main");

        // Create pybind11 array_t from C array (buffer protocol compatible)
        // This will be accessible as numpy array, torch tensor, etc. in Python
        py::array_t<float> array(
            {size},                   // shape
            {sizeof(float)},          // strides
            data,                     // data pointer
            py::cast(nullptr)         // owner (nullptr = not owned by Python)
        );

        // Call the function - python_main will dispatch to implementation
        py::object py_result = main.attr("call_main_python")(array, size);
        *result = py_result.cast<float>();

        std::cout << "[C++] Result: " << *result << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[C++] Error: " << e.what() << std::endl;
        return -1;
    }
}

void finalize_python() {
    std::cout << "[C++] Finalizing Python..." << std::endl;
    interpreter.reset();
    std::cout << "[C++] Done" << std::endl;
}

} // extern "C"
