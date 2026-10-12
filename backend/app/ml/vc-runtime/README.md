# Bundled C++ runtime for micromamba on Windows

Microsoft's Visual C++ runtime, copied next to `micromamba.exe` on Windows
(`_install_vc_runtime` in `../environment_manager.py`). Not used on macOS or
Linux.

`msvcp140.dll`, `vcruntime140.dll` and `vcruntime140_1.dll`, all version
14.44.35208.0, taken unmodified from conda-forge
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

**We own their updates.** Windows Update services the runtime in System32, not
copies next to a program, which is why Microsoft prefers the system-wide
installer. That needs admin rights, and AddaxAI installs without them. The
exposure is small: these DLLs only load into micromamba while an environment
is built.

**Replacing them.** When micromamba is bumped, or for a Microsoft security fix
in the runtime. Read micromamba's linker version
(`MajorLinkerVersion.MinorLinkerVersion` in the PE header), take a
`vc14_runtime` from conda-forge at least that new, and update the hashes in
`tests/ml/test_vc_runtime.py`, the one record of what ships here.
