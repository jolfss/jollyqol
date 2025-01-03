"""Some quality-of-life improvements."""
import functools
import sys
import inspect

#typing
from typing import Dict, Set, Tuple, assert_never

def sizes(**expected_shapes: Tuple[int|str, ...]):
    """
    Runtime assertion that Tensors match their expected sizes.
    Example: `@sizes(tensor_a=(3,'N',3), tensor_b=('B','N'))`
    """

    def sizes_decorator(tensor_func):
        actual_func_sig = inspect.signature(tensor_func)

        @functools.wraps(tensor_func)
        def sizes_wrapper(*args, **kwargs):
            try:
                actual_func_bindings = actual_func_sig.bind(*args, **kwargs)
                actual_func_bindings.apply_defaults()

                collected_tensors : Dict[str,Tuple[Tuple[int|str,...],Tuple[int,...]]] = {}
                # Maps the tensor name to its (expected,actual) shapes.

                for expected_tensor_name, expected_shape in expected_shapes.items():
                    # Ensure the tensor belongs to the function signature
                    if expected_tensor_name not in actual_func_sig.parameters:
                        raise ValueError(F"@sizes: Function is missing Tensor argument {expected_tensor_name}.")

                    # Ensure the argument is tensor-like
                    maybe_tensor = actual_func_bindings.arguments[expected_tensor_name]
                    if not hasattr(maybe_tensor, "shape"):
                        raise ValueError(F"@sizes: Expected tensor-like object, but {expected_tensor_name} has no shape attribute.")
                    actual_tensor = maybe_tensor

                    # Ensure tensor dimensions match expected dimensions
                    actual_shape = tuple(actual_tensor.shape)
                    if len(actual_shape) != len(expected_shape):
                        raise ValueError(F"@sizes: Tensor {expected_tensor_name} with shape {actual_shape} cannot match {expected_shape}.")
                    
                    collected_tensors[expected_tensor_name] = (expected_shape, actual_shape)

                exacts_good : bool = True
                # Keeps track if all static/non-wildcard dimensions are valid.

                wildcards : Dict[str,Set[int]] = {}
                # Keeps track of the possible values wildcards can take.

                for _, (expected, actual) in collected_tensors.items():
                    for (expected_dim, actual_dim) in zip(expected, actual):
                        match expected_dim:
                            case int(exact_dim):
                                exacts_good &= (actual_dim == exact_dim)
                            case str(wildcard_label):
                                wildcards.setdefault(wildcard_label,set()).add(actual_dim)
                            case _ as impossible:
                                assert_never(impossible)

                inconsistent_wildcards = {key for key, value in wildcards.items() if len(value) != 1}
                
                # Put together error message if shape errors exist
                if not exacts_good or inconsistent_wildcards:
                    tensor_strings = []
                    for tensor_name, (expected, actual) in collected_tensors.items():
                        tensor_tuple_strings = []
                        for (expected_dim, actual_dim) in zip(expected, actual):
                            match expected_dim:
                                case int(exact_dim):
                                    if actual_dim != exact_dim:
                                        tensor_tuple_strings.append(F"{exact_dim}\033[0;31m={actual_dim}\033[0m")
                                    else:
                                        pass
                                        tensor_tuple_strings.append(F"{actual_dim}")
                                case str(wildcard_label):
                                    if wildcard_label in inconsistent_wildcards:
                                        tensor_tuple_strings.append(F"{wildcard_label}\033[0;31m={actual_dim}\033[0m")
                                    else:
                                        tensor_tuple_strings.append(F"{wildcard_label}")
                        tensor_strings.append(F"{tensor_name}: ({','.join(tensor_tuple_strings)})")

                    # Filter out correct-shaped tensors by looking for escape character
                    tensor_strings = [s for s in tensor_strings if s.find('\033') != -1]

                    error_msg = None
                    if exacts_good and inconsistent_wildcards:
                        error_msg = F"@sizes: Tensor wildcard(s) were {inconsistent_wildcards} in the Tensor(s):"
                    elif not exacts_good and not inconsistent_wildcards:
                        error_msg = F"@sizes: Tensor dimensions were inconsistent in the Tensor(s):"
                    else:
                        error_msg = F"@sizes: Tensor dimensions and wildcards {', '.join(inconsistent_wildcards)} were inconsistent in Tensor(s):"
                    
                    raise ValueError(error_msg + '\n  ' + ", ".join(tensor_strings))
                
                return tensor_func(*actual_func_bindings.args, **actual_func_bindings.kwargs)
            
            except ValueError as e:
                tb = e.__traceback__
                while tb is not None:
                    code_name = tb.tb_frame.f_code.co_name
                    if code_name != 'sizes_wrapper':
                        break
                    tb = tb.tb_next
                raise e.with_traceback(tb) ### NOTE: Error caused by invalid tensor shapes coming from prior stack frame. ###


        return sizes_wrapper

    return sizes_decorator

shapes = sizes
"""An alias for the `@sizes` decorator."""

if __name__ == "__main__":
    # A quick test-demo script.
    from torch import rand, Tensor

    @sizes(a=(3,'N'), b=(4,4,'N'))
    def tensorfunc(a:Tensor, b:Tensor):
        print(a.size())
        print(b.size())

    tensorfunc(rand(3,3),rand(4,4,3)) # Works
    tensorfunc(rand(3,3),rand(3,4,4)) # Does not work