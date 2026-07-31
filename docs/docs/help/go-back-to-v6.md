---
sidebar_position: 4
title: Go back to version 6
---

# Go back to version 6

Version 6 is the legacy AddaxAI, and the latest release is 6.37. If you would rather use it than version 7, here is how to install it. This page covers Windows and macOS. Linux is left out, because version 6 was buggy on Linux and never worked well there.

Version 6 is not maintained. New models and bug fixes only go into the latest version, so version 6 stays as it is.

<div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-start'}}>
  <div style={{flex: '0.888 1 0', minWidth: '220px'}}>
    <img src="/img/v6-simple-mode.webp" alt="AddaxAI version 6, simple mode" style={{width: '100%', height: 'auto', display: 'block'}} />
  </div>
  <div style={{flex: '1.389 1 0', minWidth: '220px'}}>
    <img src="/img/v6-advanced-mode.webp" alt="AddaxAI version 6, advanced mode" style={{width: '100%', height: 'auto', display: 'block'}} />
  </div>
</div>

## Version 6 and version 7 are separate

Version 6 and version 7 do not share data, and they install in different places. If you prefer version 6 over version 7, you'll need to uninstall version 7 manually. The steps are [here](../start-here/install.mdx).

## Windows

1. Download [windows-installer.exe](https://github.com/PetervanLunteren/AddaxAI/releases/download/v6.37/windows-installer.exe).
2. Run it. Windows may warn that the publisher is unknown. Choose More info, then Run anyway.

## macOS

1. Download [macos-installer.zip](https://github.com/PetervanLunteren/AddaxAI/releases/download/v6.37/macos-installer.zip) and unzip it. You get an installer app.
2. Open the installer app. Right-click it and choose Open.
3. macOS often blocks unsigned apps and says it cannot check the developer for malware. If the dialog has no Open button, allow it by hand:
   - Open System Settings, then Privacy and Security.
   - Scroll down to the Security section, where a line says the installer was blocked.
   - Click Open Anyway, and confirm with your password or Touch ID.
   - Open the installer again, and click Open on the last dialog.
4. The installer then downloads and sets up version 6.

<img src="/img/v6-macos-installer.webp" alt="The version 6 installer downloading on macOS" style={{maxWidth: '420px', width: '100%', display: 'block'}} />
