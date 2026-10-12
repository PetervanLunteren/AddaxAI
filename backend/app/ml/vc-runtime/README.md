# Bundled C++ runtime for micromamba on Windows

Microsoft's Visual C++ runtime, copied next to `micromamba.exe` on Windows
(`_install_vc_runtime` in `../environment_manager.py`). Not used on macOS or
Linux.

| File | sha256 |
|---|---|
| `msvcp140.dll` | `8f141b4454fa78db34bc1f28c571b4da0e00cd2c43f7ad0e282f313036826aae` |
| `vcruntime140.dll` | `0205071c36c17f1efbd70178c852cb7d49985c484202752b8704b7ac6b184e60` |
| `vcruntime140_1.dll` | `963e45edd064545962e216c12d68071ced94dc8e11862a18f07f14eb2690a57c` |

All three are version 14.44.35208.0, taken unmodified from conda-forge
`vc14_runtime-14.44.35208-h818238b_41.conda` (package sha256
`ec706fe9368b837fbd05af4851f5ebef12bef8c471a273f4805f7af788407317`). They
import nothing but each other and `kernel32.dll`, so the three are the whole
set.

**Why.** The Windows micromamba is not self-contained: it imports these three
DLLs and is built with toolset 14.44. Code built with 14.40 or newer crashes in
`std::mutex` when it loads an older `msvcp140.dll`, which happens on machines
whose Visual C++ Redistributable is out of date. micromamba then dies at
start-up with exit 0xC0000005 and prints nothing. Users on 7.11.0 to 7.13.0 hit
exactly that at the first setup step (October 2026). Our backend ships
`vcruntime140.dll` but no `msvcp140.dll`, so the machine's copy decided. Windows
searches the program's own directory first, so these copies win over anything
else on the machine.

**Why bundled instead of downloaded.** conda-forge publishes this runtime only
as `.conda` (zstd), which the backend cannot unpack without another
dependency, and a download is one more thing a locked-down network can block.

**Licence.** Microsoft permits redistributing these files with an application
(the Visual Studio "Distributable Code" terms, app-local deployment).
conda-forge redistributes them the same way.

**Replacing them.** Only when micromamba is rebuilt with a newer toolset: read
its linker version (`MajorLinkerVersion.MinorLinkerVersion` in the PE header),
take the matching `vc14_runtime` from conda-forge, and update the hashes here
and in `tests/ml/test_vc_runtime.py`.
