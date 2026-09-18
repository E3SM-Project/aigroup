# taig — toys

Reusable neural blocks and architectures for testing concepts.

**Skeleton in 0.1.0.** Requires the `taig` extra (torch) once implemented.

## Rules

- Nothing else in `xaig` may import `taig`, and `taig` may not import `caig` or `daig`.
  They are siblings; the coupling would be one-way and would not stay that way.
- Import torch lazily, inside functions or behind the extra — never at package import
  time, or the base tier stops being light.
- Blocks should be framework-plain: take and return tensors, no config objects, no
  dependency on any training harness.
