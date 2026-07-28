; Custom NSIS hooks for AddaxAI: Timelapse Analyser integration +
; uninstaller.
;
; Three responsibilities:
;
; 1. Publish where AddaxAI.exe is, for Timelapse to read. See the
;    "Timelapse locator key" section below.
;
; 2. Drop a Timelapse Analyser launcher shim. Saul Greenberg's command
;    in Timelapse (TimelapseMenuRecognitions.cs, MenuItemAddaxAIRun_Click)
;    runs four candidates in order, first success wins:
;
;      (cd /d %ProgramFiles% &&
;        "%ProgramFiles%\AddaxAI_files\AddaxAI\open.bat" timelapse <dir>
;      ) || (cd /d %USERPROFILE% &&
;        "%USERPROFILE%\AddaxAI_files\AddaxAI\open.bat" timelapse <dir>
;      ) || ( ...the same two paths again for legacy EcoAssist_files... )
;
;    The `||` operator short-circuits: as soon as one command succeeds,
;    the rest are skipped.
;
;    NOTE the order: $PROGRAMFILES is tried FIRST, $PROFILE second.
;    AddaxAI's electron-builder NSIS install is per-user by default
;    (no UAC), so writes to $PROFILE always succeed and writes to
;    $PROGRAMFILES silently fail. We attempt BOTH anyway:
;
;    - $PROGRAMFILES\AddaxAI_files\AddaxAI\open.bat — Saul's first-try
;      path, but only writable when the install is elevated
;      (per-machine config, future change). Harmless no-op on per-user
;      installs. Useful in case someone runs the installer with
;      elevation, or if this codebase ever switches to perMachine: true.
;    - $PROFILE\AddaxAI_files\AddaxAI\open.bat — the typical landing
;      spot for a per-user install, and Saul's second try.
;
;    Consequence of that ordering: on a machine where a LEGACY AddaxAI
;    or EcoAssist was installed elevated into $PROGRAMFILES, its
;    open.bat wins over our per-user shim and Timelapse launches the
;    legacy app. We cannot overwrite it without elevation. The locator
;    key in (1) is the way out of this: once Timelapse prefers the
;    registry, the shim search order stops mattering.
;
;    Multi-user note: Timelapse's own install scope (per-user vs
;    per-machine) does NOT affect this. Saul's command resolves
;    %USERPROFILE% / %ProgramFiles% at runtime to the running user's
;    paths regardless of where Timelapse itself lives. The integration
;    works for whichever user has an AddaxAI shim under their profile.
;    For multi-user machines where every user needs the integration,
;    switch AddaxAI to a per-machine install (`perMachine: true` in
;    electron/package.json) so the $PROGRAMFILES write succeeds and
;    Saul's `||` fallback covers all users.
;
;    The shim translates the legacy invocation
;    (`open.bat timelapse <folder>`) into the new exe's CLI flag
;    (`AddaxAI.exe --timelapse "<folder>"`). It finds the exe through
;    the same locator key from (1), so a custom install directory works
;    on current Timelapse versions without Saul changing anything.
;
;    No legacy-detect prompt: if a legacy AddaxAI install is sitting
;    at one of these paths, we just overwrite its `open.bat`. The
;    explanatory dialog this used to show was confusing jargon for
;    most users, and the worst case (user reinstalls legacy later) is
;    self-healing — they reinstall AddaxAI and the shim wins again.
;
; 3. Offer to clear the per-user data folder on uninstall (existing
;    behavior, see customUnInstall below).
;
; ---------------------------------------------------------------------
; Timelapse locator key
; ---------------------------------------------------------------------
;
; The shim in (2) exists because Timelapse hardcodes a path to a
; launcher script. That is fragile: this installer lets the user change
; the install directory, so no hardcoded exe path is reliable, and the
; legacy-shim ordering problem above has no fix on our side alone.
;
; So we publish where we actually landed. Timelapse (or anything else)
; reads ExePath and runs `AddaxAI.exe --timelapse "<folder>"` directly,
; no shim, no cmd.exe, no guessing:
;
;   Software\AddaxAI
;     ExePath     full path to AddaxAI.exe
;     InstallDir  the install directory
;     Version     the version that wrote these values
;
; SHELL_CONTEXT is an NSIS built-in that follows SetShellVarContext, so
; this lands in HKCU for a per-user install and HKLM for a per-machine
; one, matching where the app itself went. Consumers should read HKCU
; first, then HKLM, and still File.Exists the result: an app folder
; deleted by hand rather than uninstalled leaves the key behind.
;
; This is a published contract. Renaming the key or its values breaks
; every consumer silently, so treat it as append-only.

!define TL_LOCATOR_KEY "Software\AddaxAI"

; Per-user path (typical for the current per-user installer)
!define TL_SHIM_DIR_PU "$PROFILE\AddaxAI_files\AddaxAI"
!define TL_SHIM_PATH_PU "$PROFILE\AddaxAI_files\AddaxAI\open.bat"

; Per-machine path (only writable when the installer is elevated)
!define TL_SHIM_DIR_PM "$PROGRAMFILES\AddaxAI_files\AddaxAI"
!define TL_SHIM_PATH_PM "$PROGRAMFILES\AddaxAI_files\AddaxAI\open.bat"

; Body of the shim. Resolves AddaxAI.exe at run time, in this order:
;
;   1. ${TL_LOCATOR_KEY} ExePath in HKCU, then HKLM. This is the only
;      branch that survives a custom install directory, which the
;      installer allows (allowToChangeInstallationDirectory: true).
;   2. The two default install locations, per-user then per-machine.
;      Covers an install whose locator key was removed by hand, and
;      keeps the shim working if the key is ever dropped.
;
; No `start` around the exe call: the batch file must return the exe's
; own exit code so a failure falls through Timelapse's `||` chain to the
; next candidate. `start` always returns 0 and would end the chain on a
; shim that resolved nothing.
!macro WriteTimelapseShim DIR PATH
  CreateDirectory "${DIR}"
  FileOpen $0 "${PATH}" w
  ; FileOpen on a forbidden directory returns an empty handle. Subsequent
  ; FileWrite calls on an empty handle silently no-op. NSIS does not
  ; raise. This is intentional: per-user installs cannot write to
  ; $PROGRAMFILES, and we accept that as a no-op without aborting.
  FileWrite $0 "@echo off$\r$\n"
  FileWrite $0 "setlocal$\r$\n"
  FileWrite $0 "set $\"ADDAXAI_EXE=$\"$\r$\n"
  ; `tokens=2,*` splits the reg query line "ExePath REG_SZ <path>" into
  ; %%A=REG_SZ and %%B=<path>, so a path containing spaces stays intact.
  FileWrite $0 "for /f $\"tokens=2,*$\" %%A in ('reg query $\"HKCU\${TL_LOCATOR_KEY}$\" /v ExePath 2^>nul ^| find $\"ExePath$\"') do set $\"ADDAXAI_EXE=%%B$\"$\r$\n"
  FileWrite $0 "if not exist $\"%ADDAXAI_EXE%$\" for /f $\"tokens=2,*$\" %%A in ('reg query $\"HKLM\${TL_LOCATOR_KEY}$\" /v ExePath 2^>nul ^| find $\"ExePath$\"') do set $\"ADDAXAI_EXE=%%B$\"$\r$\n"
  ; An unset ADDAXAI_EXE makes `if not exist ""` true, so both fallbacks
  ; are reached when neither registry read produced anything.
  FileWrite $0 "if not exist $\"%ADDAXAI_EXE%$\" set $\"ADDAXAI_EXE=%LOCALAPPDATA%\Programs\AddaxAI\AddaxAI.exe$\"$\r$\n"
  FileWrite $0 "if not exist $\"%ADDAXAI_EXE%$\" set $\"ADDAXAI_EXE=%ProgramFiles%\AddaxAI\AddaxAI.exe$\"$\r$\n"
  FileWrite $0 "if $\"%~1$\"==$\"timelapse$\" ($\r$\n"
  FileWrite $0 "  $\"%ADDAXAI_EXE%$\" --timelapse $\"%~2$\"$\r$\n"
  FileWrite $0 ") else ($\r$\n"
  FileWrite $0 "  $\"%ADDAXAI_EXE%$\"$\r$\n"
  FileWrite $0 ")$\r$\n"
  FileClose $0
!macroend

!macro customInstall
  ; The locator key: the path we want consumers to use going forward.
  WriteRegStr SHELL_CONTEXT "${TL_LOCATOR_KEY}" "ExePath" "$INSTDIR\AddaxAI.exe"
  WriteRegStr SHELL_CONTEXT "${TL_LOCATOR_KEY}" "InstallDir" "$INSTDIR"
  WriteRegStr SHELL_CONTEXT "${TL_LOCATOR_KEY}" "Version" "${VERSION}"

  ; The shims: kept for every Timelapse version that still looks for
  ; open.bat, which is all of them today.
  ; $PROFILE (Saul's second-try path; this one always succeeds).
  !insertmacro WriteTimelapseShim "${TL_SHIM_DIR_PU}" "${TL_SHIM_PATH_PU}"
  ; $PROGRAMFILES (Saul's first-try path; silently no-ops on per-user
  ; installs but populates correctly when the installer is elevated).
  !insertmacro WriteTimelapseShim "${TL_SHIM_DIR_PM}" "${TL_SHIM_PATH_PM}"
!macroend

!macro customUnInstall
  ; Drop the locator key so a consumer never launches an exe we just
  ; deleted. Same SHELL_CONTEXT the install wrote it under.
  DeleteRegKey SHELL_CONTEXT "${TL_LOCATOR_KEY}"

  ; Remove our Timelapse shims if we own them. Non-recursive RMDir so
  ; a legacy AddaxAI sitting next to ours stays intact: it only succeeds
  ; when the directory is empty.
  IfFileExists "${TL_SHIM_PATH_PU}" 0 +2
    Delete "${TL_SHIM_PATH_PU}"
  RMDir "${TL_SHIM_DIR_PU}"
  RMDir "$PROFILE\AddaxAI_files"

  IfFileExists "${TL_SHIM_PATH_PM}" 0 +2
    Delete "${TL_SHIM_PATH_PM}"
  RMDir "${TL_SHIM_DIR_PM}"
  RMDir "$PROGRAMFILES\AddaxAI_files"

  ; Skip the user-data prompt during silent uninstalls (CI, scripted removal).
  IfSilent skip_userdata_removal

  ; Nothing to ask about if the folder isn't present.
  IfFileExists "$PROFILE\AddaxAI\*.*" 0 skip_userdata_removal

  MessageBox MB_YESNO|MB_ICONQUESTION|MB_DEFBUTTON2 \
    "Also delete your AddaxAI user data?$\r$\n$\r$\n$PROFILE\AddaxAI$\r$\n$\r$\nThis includes your projects database, models, environments, and logs. Removing it cannot be undone.$\r$\n$\r$\nYour original images and videos are never touched.$\r$\n$\r$\nChoose No to keep your data so a future reinstall picks up where you left off." \
    /SD IDNO IDNO skip_userdata_removal

  RMDir /r "$PROFILE\AddaxAI"

  skip_userdata_removal:
!macroend
