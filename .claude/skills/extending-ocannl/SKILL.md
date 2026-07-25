---
name: extending-ocannl
description: Touch-lists for common OCANNL extension tasks — adding a primitive operation, adding or extending a backend, extending shape inference, and diagnosing output differences between backends. Use when adding an op to ops.ml, implementing a new backend, changing projection/constraint logic, or when the same computation gives different results on cc vs cuda vs metal.
---

# Extending OCANNL

## Adding New Primitive Operations

1. Add primitive operation to `arrayjit/lib/ops.ml`
2. Implement interpretation in the same file
3. Add syntax support in `tensor/ppx_*.ml` if needed
4. Add high-level wrappers in `tensor/operation.ml`
5. For neural network blocks, see `lib/nn_blocks.ml` for patterns

## Debugging Backend Discrepancies

When outputs differ between backends:

1. Compare runtime logs in `<backend>-<device>-<stream>.log` files (might require minimizing test tensors)
2. Check generated code in `build_files/<exe-name>/*.c` vs `*.cu` / `*.metal` for differences
3. Common issues:
   - Incorrect type conversion in `convert_precision` overrides
   - Different numerical precision between CPU and GPU operations

## Backend Extensions

1. Implement device-specific module following `Backend_impl` signatures
2. Add compilation logic in `arrayjit/lib/backends.ml`
3. Handle memory management and synchronization
4. Add configuration options in `ocannl_config.reference`

## Shape Inference Extensions

1. Modify projection logic in `arrayjit/lib/indexing.ml`
2. Update shape constraint generation in `tensor/shape.ml`
3. Test with various einsum patterns in e.g. `test/einsum_trivia.ml`
