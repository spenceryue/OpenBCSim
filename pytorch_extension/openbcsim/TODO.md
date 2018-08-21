## TODO
- [ ] In `Simulator.py`, separate `simulate()` into `project()` and `convolve()` steps.
  - [ ] Currently, we only have `project()` being launched. `convolve()`'s job is to convolve
  the projected scatterer time points with the excitation signal.
- [ ] Configure CTest to call `test_openbcsim.exe` with various arguments.
  - [x] Currently, getting `Exit code 0xc0000135` error. Why aren't the runtime libraries being found?
    - [x] Because semicolons were expanding `PATH` into a list in `set_test_properties()`
  - [ ] [Debug] Why isn't the `result` Tensor printing in `with_pytorch()`?
    - [ ] Check allocating output_buffer in `openbcsim_module.cpp:launch()`?
    - Note that casting DeviceProperties() also crashes if do not first import openbcsim module. What if the DLLs of PyTorch used
      are different leading to this crash? (i.e. the copied ones are different from the linked ones.)
      <STOPPED HERE 11:55 8/20/18>
  - [ ] [Debug] Why is `without_pytorch()` test also failing?
- [x] Does anything work from Python side?
  - [x] Yes, seems able to launch and produce results.
- [ ] Implement `__getitem__` for `DeviceProperties`.
