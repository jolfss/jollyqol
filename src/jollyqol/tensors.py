"""Some quality-of-life improvements."""
import functools
import sys
import inspect

#typing
from typing import Dict, Set, Tuple, assert_never

#--------------------------------------------------------------------------------------------------#
#   This code is an augmented version of my implementation of @sizes to handle the ellipse (...)   #
#--------------------------------------------------------------------------------------------------#
#@generated_edited_code
def sizes(**expected_shapes: Tuple[[int|str|type(...)], ...]):
    """
    Runtime assertion that Tensors match their expected sizes.
    Example:
      @sizes(tensor_a=(3,'N',3), tensor_b=('B','N'))
    Supports ellipsis:
      @sizes(x=(..., 4))  means 'x' can be of shape (*, 4) in PyTorch-speak.
      @sizes(y=(2, ..., 'N', 5))  means shape must start with 2, end with 5,
        and have any number of dims in-between (which can match 'N' if that dimension is named).
    """
    def sizes_decorator(tensor_func):
        actual_func_sig = inspect.signature(tensor_func)

        @functools.wraps(tensor_func)
        def sizes_wrapper(*args, **kwargs):
            try:
                actual_func_bindings = actual_func_sig.bind(*args, **kwargs)
                actual_func_bindings.apply_defaults()

                # Maps the tensor name -> (expected, actual) shapes
                collected_tensors : Dict[str, Tuple[Tuple[int|str|type(...)],...], Tuple[int,...]] = {}

                for expected_tensor_name, expected_shape in expected_shapes.items():
                    # Ensure the tensor belongs to the function signature
                    if expected_tensor_name not in actual_func_sig.parameters:
                        raise ValueError(
                            f"@sizes: Function is missing Tensor argument {expected_tensor_name}."
                        )

                    # Ensure the argument is tensor-like
                    maybe_tensor = actual_func_bindings.arguments[expected_tensor_name]
                    if not hasattr(maybe_tensor, "shape"):
                        raise ValueError(
                            f"@sizes: Expected tensor-like object, but {expected_tensor_name} has no shape attribute."
                        )
                    actual_tensor = maybe_tensor

                    # Build (expected, actual) shape pair
                    actual_shape = tuple(actual_tensor.shape)
                    collected_tensors[expected_tensor_name] = (expected_shape, actual_shape)

                # We will collect dimension constraints in two passes:
                #    1) Check if the prefix/suffix dimension counts match
                #    2) Collect wildcard constraints and see if they unify
                exacts_good = True
                wildcards: Dict[str, Set[int]] = {}

                # A small helper to unify a single dimension (one expected, one actual).
                def unify_dim(edim, adim):
                    nonlocal exacts_good
                    match edim:
                        case int(exact_dim):
                            if exact_dim != adim:
                                exacts_good = False
                        case str(wildcard_label):
                            wildcards.setdefault(wildcard_label, set()).add(adim)
                        case _ as impossible:
                            assert_never(impossible)

                # We track which tensors are dimension-mismatched for a nice error message later
                shape_mismatch_tensor_names = set()

                for tname, (expected, actual) in collected_tensors.items():
                    # Check if there is an ellipsis in the shape
                    try:
                        if ... not in expected:
                            if len(expected) != len(actual):
                                shape_mismatch_tensor_names.add(tname)
                                continue
                            # Unify dimension by dimension
                            for (edim, adim) in zip(expected, actual):
                                unify_dim(edim, adim)
                        else:
                            # We allow exactly one ellipsis
                            if sum(1 for x in expected if x is ...) > 1:
                                raise ValueError(
                                    "@sizes: Only one ellipsis allowed per shape specification."
                                )
                            # Split the shape on the ellipsis
                            ellipsis_index = expected.index(...)
                            prefix = expected[:ellipsis_index]
                            suffix = expected[ellipsis_index+1:]

                            # We must have at least prefix+suffix length <= actual length
                            if len(prefix) + len(suffix) > len(actual):
                                shape_mismatch_tensor_names.add(tname)
                                continue

                            # Unify prefix
                            for (edim, adim) in zip(prefix, actual[:len(prefix)]):
                                unify_dim(edim, adim)

                            # The middle is unconstrained, so skip

                            # Unify suffix
                            # e.g. if suffix = (4, 'N'), and prefix = (3,), then
                            # actual[:len(prefix)] = actual[:1], actual[-len(suffix):] = actual[-2:]
                            if len(suffix) > 0:
                                for (edim, adim) in zip(
                                    suffix[::-1],
                                    actual[len(actual)-len(suffix):][::-1],
                                ):
                                    unify_dim(edim, adim)

                    except ValueError as e:
                        raise e  # Reraise for clarity

                # Now check that all wildcards are consistent (each name must unify to exactly one set dimension)
                inconsistent_wildcards = {key for key, value in wildcards.items() if len(value) != 1}

                if not exacts_good or shape_mismatch_tensor_names or inconsistent_wildcards:
                    # Build a message for debugging
                    tensor_strings = []
                    for tensor_name, (expected, actual) in collected_tensors.items():
                        prefix_str = f"{tensor_name}: ("
                        parts = []
                        # We'll do a second pass, but highlighting mistakes
                        #   so we parse again. This is purely for a nicer error message:
                        if ... not in expected:
                            # exact match
                            for (edim, adim) in zip(expected, actual) if len(expected)==len(actual) else []:
                                match edim:
                                    case int(exact_dim):
                                        if exact_dim != adim:
                                            parts.append(f"{exact_dim}\033[0;31m={adim}\033[0m")
                                        else:
                                            parts.append(str(adim))
                                    case str(wildcard_label):
                                        # Check if inconsistent
                                        if wildcard_label in inconsistent_wildcards:
                                            parts.append(f"{wildcard_label}\033[0;31m={adim}\033[0m")
                                        else:
                                            parts.append(f"{wildcard_label}")
                                    case _:
                                        parts.append(f"{edim}?")
                            # If length mismatch, we highlight all
                            if len(expected) != len(actual):
                                mismatch_str = f"Expected {len(expected)} dims, got {len(actual)} dims."
                                parts = [mismatch_str]
                        else:
                            # has ellipsis
                            # prefix + suffix
                            ellipsis_index = expected.index(...)
                            prefix = expected[:ellipsis_index]
                            suffix = expected[ellipsis_index+1:]
                            # highlight prefix
                            for (edim, adim) in zip(prefix, actual[:len(prefix)]):
                                match edim:
                                    case int(exact_dim):
                                        if exact_dim != adim:
                                            parts.append(f"{exact_dim}\033[0;31m={adim}\033[0m")
                                        else:
                                            parts.append(str(adim))
                                    case str(wildcard_label):
                                        if wildcard_label in inconsistent_wildcards:
                                            parts.append(f"{wildcard_label}\033[0;31m={adim}\033[0m")
                                        else:
                                            parts.append(f"{wildcard_label}")
                                    case _:
                                        parts.append(f"{edim}?")

                            parts.append("...")  # literal ellipsis in error message

                            if len(prefix) + len(suffix) > len(actual):
                                # mismatch, can't unify suffix
                                parts.append(f"\033[0;31m(Suffix mismatch?)\033[0m")
                            else:
                                # highlight suffix
                                tail_actual = actual[len(actual)-len(suffix):]
                                for (edim, adim) in zip(suffix, tail_actual):
                                    match edim:
                                        case int(exact_dim):
                                            if exact_dim != adim:
                                                parts.append(f"{exact_dim}\033[0;31m={adim}\033[0m")
                                            else:
                                                parts.append(str(adim))
                                        case str(wildcard_label):
                                            if wildcard_label in inconsistent_wildcards:
                                                parts.append(f"{wildcard_label}\033[0;31m={adim}\033[0m")
                                            else:
                                                parts.append(f"{wildcard_label}")
                                        case _:
                                            parts.append(f"{edim}?")

                        tensor_strings.append(prefix_str + ",".join(parts) + ")")

                    error_msg = []
                    if shape_mismatch_tensor_names:
                        error_msg.append(
                            "Tensor(s) had different # of dims than expected: "
                            + ", ".join(shape_mismatch_tensor_names)
                        )
                    if inconsistent_wildcards:
                        error_msg.append(
                            "Inconsistent wildcard(s): " + ", ".join(inconsistent_wildcards)
                        )
                    if not exacts_good:
                        error_msg.append("Static dims do not match.")

                    final_error_msg = (
                        "@sizes: " + "; ".join(error_msg) + "\n  " + "\n  ".join(tensor_strings)
                    )
                    raise ValueError(final_error_msg)

                return tensor_func(*actual_func_bindings.args, **actual_func_bindings.kwargs)

            except ValueError as e:
                # Strip this decorator's traceback so user sees their own code line
                tb = e.__traceback__
                while tb is not None:
                    code_name = tb.tb_frame.f_code.co_name
                    if code_name != 'sizes_wrapper':
                        break
                    tb = tb.tb_next
                raise e.with_traceback(tb)

        return sizes_wrapper

    return sizes_decorator

shapes = sizes
"""An alias for the `@sizes` decorator."""


if __name__ == "__main__":
    from torch import rand, Tensor

    @sizes(a=(3,'N'), b=(4,4,'N'))
    def tensorfunc(a:Tensor, b:Tensor):
        print(a.size())
        print(b.size())

    tensorfunc(rand(3,3), rand(4,4,3)) # Works
    try:
        tensorfunc(rand(3,3), rand(3,4,4)) # Does not work
    except ValueError as e:
        print(e)

    @sizes(x=(4,...))
    def ends_with_4(x: Tensor):
        print("Shape of x:", x.shape)

    ends_with_4(rand(4,1,2,3,4))     # OK
    ends_with_4(rand(4))         # OK
    ends_with_4(rand(3,4))       # fails (first dim not 4)

#---------------------------------------------------------------------------------------------#
#   This was my ORIGINAL code to implement the @sizes decorator which I implemented myself.   #
#---------------------------------------------------------------------------------------------#
# def __sizes_deprecated(**expected_shapes: Tuple[int|str, ...]):
#     """
#     Runtime assertion that Tensors match their expected sizes.
#     Example: `@sizes(tensor_a=(3,'N',3), tensor_b=('B','N'))`
#     """

#     def sizes_decorator(tensor_func):
#         actual_func_sig = inspect.signature(tensor_func)

#         @functools.wraps(tensor_func)
#         def sizes_wrapper(*args, **kwargs):
#             try:
#                 actual_func_bindings = actual_func_sig.bind(*args, **kwargs)
#                 actual_func_bindings.apply_defaults()

#                 collected_tensors : Dict[str,Tuple[Tuple[int|str,...],Tuple[int,...]]] = {}
#                 # Maps the tensor name to its (expected,actual) shapes.

#                 for expected_tensor_name, expected_shape in expected_shapes.items():
#                     # Ensure the tensor belongs to the function signature
#                     if expected_tensor_name not in actual_func_sig.parameters:
#                         raise ValueError(F"@sizes: Function is missing Tensor argument {expected_tensor_name}.")

#                     # Ensure the argument is tensor-like
#                     maybe_tensor = actual_func_bindings.arguments[expected_tensor_name]
#                     if not hasattr(maybe_tensor, "shape"):
#                         raise ValueError(F"@sizes: Expected tensor-like object, but {expected_tensor_name} has no shape attribute.")
#                     actual_tensor = maybe_tensor

#                     # Ensure tensor dimensions match expected dimensions
#                     actual_shape = tuple(actual_tensor.shape)
#                     if len(actual_shape) != len(expected_shape):
#                         raise ValueError(F"@sizes: Tensor {expected_tensor_name} with shape {actual_shape} cannot match {expected_shape}.")
                    
#                     collected_tensors[expected_tensor_name] = (expected_shape, actual_shape)

#                 exacts_good : bool = True
#                 # Keeps track if all static/non-wildcard dimensions are valid.

#                 wildcards : Dict[str,Set[int]] = {}
#                 # Keeps track of the possible values wildcards can take.

#                 for _, (expected, actual) in collected_tensors.items():
#                     for (expected_dim, actual_dim) in zip(expected, actual):
#                         match expected_dim:
#                             case int(exact_dim):
#                                 exacts_good &= (actual_dim == exact_dim)
#                             case str(wildcard_label):
#                                 wildcards.setdefault(wildcard_label,set()).add(actual_dim)
#                             case _ as impossible:
#                                 assert_never(impossible)

#                 inconsistent_wildcards = {key for key, value in wildcards.items() if len(value) != 1}
                
#                 # Put together error message if shape errors exist
#                 if not exacts_good or inconsistent_wildcards:
#                     tensor_strings = []
#                     for tensor_name, (expected, actual) in collected_tensors.items():
#                         tensor_tuple_strings = []
#                         for (expected_dim, actual_dim) in zip(expected, actual):
#                             match expected_dim:
#                                 case int(exact_dim):
#                                     if actual_dim != exact_dim:
#                                         tensor_tuple_strings.append(F"{exact_dim}\033[0;31m={actual_dim}\033[0m")
#                                     else:
#                                         pass
#                                         tensor_tuple_strings.append(F"{actual_dim}")
#                                 case str(wildcard_label):
#                                     if wildcard_label in inconsistent_wildcards:
#                                         tensor_tuple_strings.append(F"{wildcard_label}\033[0;31m={actual_dim}\033[0m")
#                                     else:
#                                         tensor_tuple_strings.append(F"{wildcard_label}")
#                         tensor_strings.append(F"{tensor_name}: ({','.join(tensor_tuple_strings)})")

#                     # Filter out correct-shaped tensors by looking for escape character
#                     tensor_strings = [s for s in tensor_strings if s.find('\033') != -1]

#                     error_msg = None
#                     if exacts_good and inconsistent_wildcards:
#                         error_msg = F"@sizes: Tensor wildcard(s) were {inconsistent_wildcards} in the Tensor(s):"
#                     elif not exacts_good and not inconsistent_wildcards:
#                         error_msg = F"@sizes: Tensor dimensions were inconsistent in the Tensor(s):"
#                     else:
#                         error_msg = F"@sizes: Tensor dimensions and wildcards {', '.join(inconsistent_wildcards)} were inconsistent in Tensor(s):"
                    
#                     raise ValueError(error_msg + '\n  ' + ", ".join(tensor_strings))
                
#                 return tensor_func(*actual_func_bindings.args, **actual_func_bindings.kwargs)
            
#             except ValueError as e:
#                 tb = e.__traceback__
#                 while tb is not None:
#                     code_name = tb.tb_frame.f_code.co_name
#                     if code_name != 'sizes_wrapper':
#                         break
#                     tb = tb.tb_next
#                 raise e.with_traceback(tb) ### NOTE: Error caused by invalid tensor shapes coming from prior stack frame. ###

#         return sizes_wrapper

#     return sizes_decorator

