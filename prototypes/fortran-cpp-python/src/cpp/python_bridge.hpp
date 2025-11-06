#ifndef PYTHON_BRIDGE_HPP
#define PYTHON_BRIDGE_HPP

extern "C" {

int initialize_python();

int call_main_python(const float* data, int size, float* result);

void finalize_python();

}

#endif // PYTHON_BRIDGE_HPP
