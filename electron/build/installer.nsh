; Custom NSIS hooks for AddaxAI: Timelapse Analyser shim + uninstaller.
;
; Two responsibilities:
;
; 1. Drop a Timelapse Analyser launcher shim. Saul Greenberg's command
;    in Timelapse runs:
;
;      (cd /d %homedrive%%homepath% &&
;        "%homedrive%%homepath%\AddaxAI_files\AddaxAI\open.bat" timelapse <dir>
;      ) || (cd /d %ProgramFiles% &&
;        "%ProgramFiles%\AddaxAI_files\AddaxAI\open.bat" timelapse <dir>
;      )
;
;    The `||` operator short-circuits: as soon as the first command
;    succeeds, the second is skipped.
;
;    AddaxAI's electron-builder NSIS install is per-user by default
;    (no UAC), so writes to $PROFILE always succeed and writes to
;    $PROGRAMFILES silently fail. We attempt BOTH anyway:
;
;    - $PROFILE\AddaxAI_files\AddaxAI\open.bat — the typical landing
;      spot. Saul's command tries this first and finds it.
;    - $PROGRAMFILES\AddaxAI_files\AddaxAI\open.bat — only writable
;      when the install is elevated (per-machine config, future
;      change). Harmless no-op on per-user installs. Useful in case
;      someone runs the installer with elevation, or if this codebase
;      ever switches to perMachine: true.
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
;    (`AddaxAI.exe --timelapse "<folder>"`).
;
;    No legacy-detect prompt: if a legacy AddaxAI install is sitting
;    at one of these paths, we just overwrite its `open.bat`. The
;    explanatory dialog this used to show was confusing jargon for
;    most users, and the worst case (user reinstalls legacy later) is
;    self-healing — they reinstall AddaxAI and the shim wins again.
;
; 2. Offer to clear the per-user data folder on uninstall (existing
;    behavior, see customUnInstall below).

; Per-user path (typical for the current per-user installer)
!define TL_SHIM_DIR_PU "$PROFILE\AddaxAI_files\AddaxAI"
!define TL_SHIM_PATH_PU "$PROFILE\AddaxAI_files\AddaxAI\open.bat"

; Per-machine path (only writable when the installer is elevated)
!define TL_SHIM_DIR_PM "$PROGRAMFILES\AddaxAI_files\AddaxAI"
!define TL_SHIM_PATH_PM "$PROGRAMFILES\AddaxAI_files\AddaxAI\open.bat"

; Body of the shim. Probes both possible new-AddaxAI install locations
; at run time so it works regardless of which install scope was used.
!macro WriteTimelapseShim DIR PATH
  CreateDirectory "${DIR}"
  FileOpen $0 "${PATH}" w
  ; FileOpen on a forbidden directory returns an empty handle. Subsequent
  ; FileWrite calls on an empty handle silently no-op. NSIS does not
  ; raise. This is intentional: per-user installs cannot write to
  ; $PROGRAMFILES, and we accept that as a no-op without aborting.
  FileWrite $0 "@echo off$\r$\n"
  FileWrite $0 "setlocal$\r$\n"
  FileWrite $0 "set $\"ADDAXAI_EXE=%LOCALAPPDATA%\Programs\AddaxAI\AddaxAI.exe$\"$\r$\n"
  FileWrite $0 "if not exist $\"%ADDAXAI_EXE%$\" set $\"ADDAXAI_EXE=%ProgramFiles%\AddaxAI\AddaxAI.exe$\"$\r$\n"
  FileWrite $0 "if $\"%~1$\"==$\"timelapse$\" ($\r$\n"
  FileWrite $0 "  $\"%ADDAXAI_EXE%$\" --timelapse $\"%~2$\"$\r$\n"
  FileWrite $0 ") else ($\r$\n"
  FileWrite $0 "  $\"%ADDAXAI_EXE%$\"$\r$\n"
  FileWrite $0 ")$\r$\n"
  FileClose $0
!macroend

!macro customInstall
  ; $PROFILE first (Saul's first-try path; this one always succeeds).
  !insertmacro WriteTimelapseShim "${TL_SHIM_DIR_PU}" "${TL_SHIM_PATH_PU}"
  ; $PROGRAMFILES second (Saul's fallback; silently no-ops on per-user
  ; installs but populates correctly when the installer is elevated).
  !insertmacro WriteTimelapseShim "${TL_SHIM_DIR_PM}" "${TL_SHIM_PATH_PM}"
!macroend

!macro customUnInstall
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
