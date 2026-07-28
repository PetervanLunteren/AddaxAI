"""Route TLS verification through the operating system trust store.

The frozen (PyInstaller) backend ships its own Python with no system CA
bundle wired into OpenSSL, so a plain ``urllib.request.urlopen`` fails on
first launch with ``CERTIFICATE_VERIFY_FAILED`` while downloading micromamba
(``requests`` / ``httpx`` escape this only because they carry certifi).

``truststore`` makes every ``ssl.SSLContext`` defer to the OS trust store
(macOS Keychain, Windows certificate store). That fixes both the
missing-bundle case and lab / institutional networks that do TLS inspection
with a locally installed corporate root CA, which a fixed certifi bundle
would never trust.
"""

from app.core.logging_config import get_logger

logger = get_logger(__name__)


def enable_os_trust_store() -> None:
    """Inject truststore so all TLS uses the OS trust store.

    Call once, as early as possible, before any network request. Failure is
    logged but not fatal: the process falls back to OpenSSL's default
    context, which is no worse than before this call existed.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
        logger.info("TLS verification routed through the OS trust store")
    except Exception:
        logger.error("Could not enable OS trust store for TLS", exc_info=True)
