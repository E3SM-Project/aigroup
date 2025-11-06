"""
Python main is a lightweight entry point
"""


def initialize():
    """
    Nothing to do here ... 
    """
    print("[Python] Dispatcher initialized")
    return {"status": "ready"}


def call_main_python(data, size):
    """
    Dispatch to some torch calculation
    """
    print(f"[Python] Dispatching tensor operation (size={size})")

    # we are importing here so we don't need to make the 
    # cpp bridge aware of this stuff and such
    from python_torch import compute_tensor_operation as impl

    result = impl(data, size)
    print(f"[Python] Tensor operation complete: {result}")
    return result
