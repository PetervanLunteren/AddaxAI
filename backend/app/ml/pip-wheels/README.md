# Bundled pip wheels

Wheels shipped inside the app because the environment build must not depend on
reaching them over the network.

## ultralytics_yolov5-0.1.1-py3-none-any.whl

| | |
|---|---|
| Size | 862,408 bytes |
| sha256 | `d532e62d6d7cd6c7381e4453810f31f10fd269cf9f237494bc570e74168dc2e1` |
| Upstream | https://github.com/ultralytics/yolov5 |
| Also hosted at | https://huggingface.co/Addax-Data-Science/pip-wheels |

**Why a wheel at all.** `megadetector` depends on `ultralytics-yolov5==0.1.1`,
whose PyPI release is sdist-only. That sdist's `setup.py` downloads a README
from GitHub at build time, which crashes on machines where Python cannot load
the Windows certificate store (ssl ASN1 error, beta report 2026-06-10).
Installing a wheel skips `setup.py` entirely.

**Why bundled instead of downloaded.** The env YAMLs pin it by direct URL, and
pip has no index setting that can redirect a direct-URL requirement:
`--index-url`, `--extra-index-url` and `--find-links` only affect index
resolution. On a network that blocks the host, that single line fails the whole
environment build and no user configuration can fix it. This is what stopped
setup in mainland China (beta report 2026-08-13), where hf-mirror.com answers
`/resolve/` with a 308 redirect back to the blocked huggingface.co.

`substitute_bundled_wheels` in `../environment_manager.py` rewrites the URL to
this directory when it copies the YAML for micromamba. The YAMLs keep the URL,
so they still record where the file came from and pip still verifies the
`#sha256=` fragment against this copy.

**Licence.** The package metadata declares `License: Apache` and the Apache
classifier, but the `LICENSE` file inside the wheel is the GNU General Public
License v3. Upstream YOLOv5 was GPL-3.0 at the time of this release and is
AGPL-3.0 today. We redistribute the wheel unmodified, its licence text travels
inside it, and the corresponding source is at the upstream repository above and
on PyPI. It is installed into a separate analysis environment and run as a
separate program; no AddaxAI code links against it.

**Replacing this file.** `tests/ml/test_bundled_wheels.py` derives the expected
filename and sha256 from the env YAMLs, so change the YAML pin first and the
test will tell you exactly which file to put here.
