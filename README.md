CUDA Samples
============

All samples make use of the driver API.

deviceQuery
-----------

This sample enumerates the properties of the CUDA devices present in the system.

bandwidthTest
-------------

This sample measures host to device and device to host copy bandwidth for pageable, page-locked and write-combined memory of transfer sizes 3KB, 15KB, 15MB and 100MB, and outputs them in CSV format.

jit
---

This sample jit-in-time compiles a .ptx and outputs error log and info log.

zeroCopy
--------

This sample uses zero copy to map a host pointer to a device pointer so that kernels can read and write directly to pinned system memory.

vectorAdd
---------

This sample uses async API, dynamic ptx version selection, and constant and shared memory to add two vectors of float.

hyperQ
------

This sample uses multiple streams to exploit the HyperQ technology.
