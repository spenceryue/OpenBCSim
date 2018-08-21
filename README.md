# PyTorch Interface for OpenBCSim
This project has diverged significantly from the original OpenBCSim. The aims are to provide:
 - A CUDA implementation of the Spatial Impulse Response algorithm ([as used by Field II](https://field-ii.dk/?background.html), see also [here](https://field-ii.dk/documents/jaj_springer_2002.pdf)).
 - A Python interface with PyTorch tensors to manage GPU memory (project is built as a [PyTorch extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)).
 - A planar wave beamforming reference implementation.

The new code for this project is under [`pytorch_extension/openbcsim`](pytorch_extension/openbcsim).

---

### _(Original README of)_ OpenBCSim
This project is a fast C++/CUDA open-source implementation of an ultrasound simulator based on the COLE algorithm as published by Gao et al. in "A fast convolution-based methodology to simulate
2-D/3-D cardiac ultrasound images.", IEEE TUFFC 2009.

The algorithm has been extended to optionally use B-splines for representing dynamic point scatterers.
### Features:
- Python scripts for generating point-scatterer phantoms
- Supports both fixed and dynamic (B-spline based) point scatterers
- Multicore CPU implementation (OpenMP based)
- GPU implementation (using NVIDIA CUDA)
- Python interface using Boost.Python and numpy-boost
- Qt5-based interactive GUI front-end
- The output data type is complex demodulated IQ data (w/optional radial decimation)
- Cross-platform code. Successfully built on Linux (Ubuntu 15.04) and Windows 7

This code is still experimental. More documentation, examples, and instructions on how to compile the code will be added soon.
