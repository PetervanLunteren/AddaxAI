#!/bin/bash

# Custom replacement for electron-builder's default after-install.tpl.
# Same content, except the chrome-sandbox SUID bit is set unconditionally.
# The default template only sets it when user namespaces look unavailable,
# but that test runs as root during package install and root always has
# user namespaces. On Ubuntu 23.10+ AppArmor denies unprivileged user
# namespaces to the app at runtime, Chromium then falls back to the SUID
# helper, and without the SUID bit the app aborts on launch. Unconditional
# SUID (the same thing Google Chrome's deb does) keeps the sandbox working
# everywhere.
#
# Note: this file is run through electron-builder's template engine, so
# ${...} is reserved for its macros. Plain $VAR bash syntax is safe.

if type update-alternatives 2>/dev/null >&1; then
    # Remove previous link if it doesn't use update-alternatives
    if [ -L '/usr/bin/${executable}' -a -e '/usr/bin/${executable}' -a "`readlink '/usr/bin/${executable}'`" != '/etc/alternatives/${executable}' ]; then
        rm -f '/usr/bin/${executable}'
    fi
    update-alternatives --install '/usr/bin/${executable}' '${executable}' '/opt/${sanitizedProductName}/${executable}' 100 || ln -sf '/opt/${sanitizedProductName}/${executable}' '/usr/bin/${executable}'
else
    ln -sf '/opt/${sanitizedProductName}/${executable}' '/usr/bin/${executable}'
fi

chmod 4755 '/opt/${sanitizedProductName}/chrome-sandbox' || true

if hash update-mime-database 2>/dev/null; then
    update-mime-database /usr/share/mime || true
fi

if hash update-desktop-database 2>/dev/null; then
    update-desktop-database /usr/share/applications || true
fi
