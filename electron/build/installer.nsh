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
;    The `||` fallback only fires if the first command fails, so the
;    user-profile path is tried FIRST. If we only drop a shim at
;    Program Files, a per-user legacy install at $PROFILE wins and our
;    new app never gets called. That is why we drop the shim at BOTH
;    `$PROFILE\AddaxAI_files\AddaxAI\open.bat` (Saul's first try) and
;    `$PROGRAMFILES\AddaxAI_files\AddaxAI\open.bat` (the fallback).
;
;    The shim translates the legacy invocation
;    (`open.bat timelapse <folder>`) into the new exe's CLI flag
;    (`AddaxAI.exe --timelapse "<folder>"`).
;
;    If a legacy AddaxAI install is found at either location, prompt
;    the user before overwriting their existing open.bat. Default
;    answer is No so we never silently break a working legacy setup.
;
; 2. Offer to clear the per-user data folder on uninstall (existing
;    behavior, see customUnInstall below).

; Per-machine path (Saul's fallback target)
!define TL_SHIM_DIR_PM "$PROGRAMFILES\AddaxAI_files\AddaxAI"
!define TL_SHIM_PATH_PM "$PROGRAMFILES\AddaxAI_files\AddaxAI\open.bat"
!define TL_LEGACY_MARKER_PM "$PROGRAMFILES\AddaxAI_files\AddaxAI\AddaxAI_GUI.py"

; Per-user path (Saul's first-try target)
!define TL_SHIM_DIR_PU "$PROFILE\AddaxAI_files\AddaxAI"
!define TL_SHIM_PATH_PU "$PROFILE\AddaxAI_files\AddaxAI\open.bat"
!define TL_LEGACY_MARKER_PU "$PROFILE\AddaxAI_files\AddaxAI\AddaxAI_GUI.py"

; Body of the shim. Probes both per-user and per-machine new-AddaxAI
; install locations so a single shim works regardless of which install
; scope was used.
!macro WriteTimelapseShim DIR PATH
  CreateDirectory "${DIR}"
  FileOpen $0 "${PATH}" w
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
  ; Collect legacy install paths. $1 holds the bullet list shown in the
  ; prompt; empty means no legacy installs anywhere.
  StrCpy $1 ""
  IfFileExists "${TL_LEGACY_MARKER_PU}" 0 +2
    StrCpy $1 "$1$\r$\n  ${TL_SHIM_DIR_PU}"
  IfFileExists "${TL_LEGACY_MARKER_PM}" 0 +2
    StrCpy $1 "$1$\r$\n  ${TL_SHIM_DIR_PM}"

  StrCmp $1 "" write_shims
    IfSilent write_shims  ; CI / scripted installs: leave legacy alone.
    MessageBox MB_YESNOCANCEL|MB_ICONQUESTION|MB_DEFBUTTON2 \
      "A legacy AddaxAI install was found at:$1$\r$\n$\r$\nThe new AddaxAI can replace its Timelapse Analyser launcher (open.bat) so Timelapse calls the new app instead.$\r$\n$\r$\nYes  - replace the legacy launcher (recommended)$\r$\nNo   - keep legacy launcher; Timelapse will keep using legacy AddaxAI$\r$\nCancel - abort install" \
      /SD IDNO IDYES write_shims IDNO skip_shims
    Abort  ; Cancel pressed.

  write_shims:
    ; Drop at the per-user path FIRST since Saul's command tries that
    ; one first. Order does not matter for the install, only for the
    ; reading of this file by future maintainers.
    !insertmacro WriteTimelapseShim "${TL_SHIM_DIR_PU}" "${TL_SHIM_PATH_PU}"
    !insertmacro WriteTimelapseShim "${TL_SHIM_DIR_PM}" "${TL_SHIM_PATH_PM}"
  skip_shims:
!macroend

!macro customUnInstall
  ; Remove the Timelapse shims if we own them. We do not unconditionally
  ; nuke the directories because a legacy AddaxAI install may still live
  ; there; only the open.bat files we wrote are ours to remove. The
  ; RMDir calls are non-recursive and only succeed when the directory
  ; is empty, which keeps legacy installs intact.
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
    "Also delete your AddaxAI user data?$\r$\n$\r$\n$PROFILE\AddaxAI$\r$\n$\r$\nThis includes your projects database, downloaded model weights, the analysis environment (about 2 GB), logs, and thumbnails. Removing it cannot be undone.$\r$\n$\r$\nChoose No to keep your data so a future reinstall picks up where you left off." \
    /SD IDNO IDNO skip_userdata_removal

  RMDir /r "$PROFILE\AddaxAI"

  skip_userdata_removal:
!macroend
