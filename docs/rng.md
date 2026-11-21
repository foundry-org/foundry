# RNG in captured graphs

A graph that draws from torch's default CUDA generator of the capturing device
(`torch.rand`, `dropout`, `bernoulli_`, `multinomial`, ...) is saved with the addresses of
its seed/offset buffers and its offset increment. On LOAD each replay takes (seed, offset)
from that generator and advances it by the increment, as torch's own replay does, so
`torch.manual_seed` and eager RNG in the LOAD process apply to loaded graphs too.

Refused at SAVE (`foundry capture: unsupported: ...`):

- RNG from a `torch.Generator` other than the device's default generator
- RNG drawn on the capturing stream from a thread other than the capturing one (torch 2.14 /
  2.15; on 2.16 such a draw takes a pair torch pre-allocated and is saved)
- allocations on a non-capturing stream inside the capture body
- a CUDA allocator backend other than `native`
- a torch whose `memory_snapshot()` lacks `segment_pool_id` / `requested_size`, or whose
  capture RNG state fails the SAVE-time self-test
- any capture after a capture that failed (was refused, or its body raised) once its record
  window had allocated memory: LOAD cannot reproduce that allocation
- saving an RNG graph before the graph whose capture allocated Foundry's RNG pool segment

torch refuses captures on the default stream.

Unsupported and not detected:

- RNG from a non-default `torch.Generator` drawn on the capturing stream by another thread
  whose state takes an address freed inside the same capture (torch 2.14 / 2.15)
- swapping the default generator's state (`graphsafe_set_state`) after a capture: torch's
  replay keeps the captured state, a loaded graph uses the generator's current one
