/*


Q: what is uncoalesced r/w ? bank conflict ?
A: see learn cuda ch2
uncoalesced == multiple threads from the same warp accessing same bank
resulting in bank conflict


Q: what is warp ? what is the significance of "warp size in GPU is 32" ?
A:
1 lane == 1 thread within 1 warp
1 warp == 32 lanes (fixed)
1 block == arbitrary warps 
1 grid == arbitrary blocks 

significance of warp:
1 warp of 32 threads is GUARANTEED to execute concurrently by GPU streaming multiprocessor 
warps within 1 block or blocks within 1 grid is ARBITRARILY ordered


warp is responsible for populating the x and y variables within an index that has a range of
64 KB, that is, warp 1 is responsible for the first 64 KB, while warp 2 is responsible for the
elements in the next 64 KB. Each thread in a warp loops (the innermost for loop) to
populate the index within the same 64 KB


warp per page concept, which basically means that each warp will access elements that
are in the same pages. This requires additional effort from the developer.

The CUDA streaming multiprocessor controls CUDA threads in groups of 32. A group is
called a warp. In this manner, one or multiple warps configures a CUDA thread block.


Q: what is warp divergence ? 
A: whenever theres if else 
no divergence when the stride size is greater than the warp size




Q: 
A: 



Q: global vs shared vs pinned memory ? why would u not use shared memory ?
A: 
global == worried about non sequential access, cache miss
shared == worried about accessing same bank, x wait 1 more cycle
each bank has a bandwidth of 4 bytes per 2 clock cycles




*/