## TODO
- In `Simulator.py`, separate `simulate()` into `project()` and `convolve()` steps.
  - Currently, we only have `project()` being launched. `convolve()`'s job is to convolve
  the projected scatterer time points with the excitation signal.
- Write tests calling `test_openbcsim.exe` with various arguments.
  - Currently, getting `Exit code 0xc0000135` error. Why aren't the runtime libraries being found?
