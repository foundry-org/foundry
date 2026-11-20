# RNG in captured graphs

A graph that draws from torch's default CUDA generator of the capturing device
(`torch.rand`, `dropout`, `bernoulli_`, `multinomial`, ...) is saved with the addresses of
its seed/offset buffers and its offset increment. On LOAD each replay takes (seed, offset)
from that generator and advances it by the increment, as torch's own replay does, so
`torch.manual_seed` and eager RNG in the LOAD process apply to loaded graphs too.

Refused at SAVE (`foundry capture: unsupported: ...`):

- RNG from a `torch.Generator` other than the device's default generator
- RNG drawn on the capturing stream from a thread other than the capturing one
- allocations on a non-capturing stream inside the capture body
- a CUDA allocator backend other than `native`
- a torch whose `memory_snapshot()` lacks `segment_pool_id` / `requested_size`, or whose
  capture RNG state fails the SAVE-time self-test
- graphs that use RNG state allocated in a capture that was refused or raised

torch refuses captures on the default stream.
