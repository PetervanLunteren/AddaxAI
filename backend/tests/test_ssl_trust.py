"""enable_os_trust_store() injects truststore and never bricks startup.

The frozen backend has no CA bundle wired into OpenSSL, so this routes TLS
through the OS trust store. The call runs once at import of app.main, so a
failure here must degrade (log + continue), not crash the process.
"""

import truststore

from app.core import ssl_trust


def test_injects_truststore(monkeypatch):
    calls: list[bool] = []
    monkeypatch.setattr(
        truststore, "inject_into_ssl", lambda: calls.append(True)
    )

    ssl_trust.enable_os_trust_store()

    assert calls == [True]


def test_failure_does_not_raise(monkeypatch):
    def boom() -> None:
        raise RuntimeError("no OS trust store here")

    monkeypatch.setattr(truststore, "inject_into_ssl", boom)

    # Must swallow the error so app startup continues on the default context.
    ssl_trust.enable_os_trust_store()
