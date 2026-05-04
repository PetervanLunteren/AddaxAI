; Custom NSIS hooks for AddaxAI: Timelapse Analyser shim + uninstaller.
;
; Two responsibilities:
;
; 1. Drop a Timelapse Analyser launcher shim at
;    %ProgramFiles%\AddaxAI_files\AddaxAI\open.bat. Saul Greenberg's
;    Timelapse Analyser hardcodes that path in its "Recognition > Use
;    AddaxAI" command. The shim translates the legacy invocation
;    (`open.bat timelapse <folder>`) into the new exe's CLI flag
;    (`AddaxAI.exe --timelapse "<folder>"`). Without this shim, users
;    invoking AddaxAI from Timelapse would have to wait until Saul
;    updates Timelapse's command list.
;
;    If a legacy AddaxAI install is found at the same path, prompt the
;    user to remove it before dropping the shim. The legacy install
;    also has an open.bat there, and we do not want to overwrite a
;    working legacy setup without consent.
;
; 2. Offer to clear the per-user data folder on uninstall (existing
;    behavior, see customUnInstall below).

!define TL_SHIM_DIR "$PROGRAMFILES\AddaxAI_files\AddaxAI"
!define TL_SHIM_PATH "$PROGRAMFILES\AddaxAI_files\AddaxAI\open.bat"
!define TL_LEGACY_MARKER "$PROGRAMFILES\AddaxAI_files\AddaxAI\AddaxAI_GUI.py"

!macro customInstall
  ; Detect a legacy AddaxAI install. If present, ask the user before we
  ; touch the directory. Default answer is No (skip shim) so we never
  ; silently break a working legacy setup.
  IfFileExists "${TL_LEGACY_MARKER}" 0 write_shim
    IfSilent write_shim  ; CI / scripted installs: leave legacy alone.
    MessageBox MB_YESNOCANCEL|MB_ICONQUESTION|MB_DEFBUTTON2 \
      "A legacy AddaxAI install was found at:$\r$\n$\r$\n${TL_SHIM_DIR}$\r$\n$\r$\nThe new AddaxAI can replace its Timelapse Analyser launcher (open.bat) so Timelapse calls the new app instead.$\r$\n$\r$\nYes  - replace the legacy launcher (recommended)$\r$\nNo   - keep legacy launcher; Timelapse will keep using legacy AddaxAI$\r$\nCancel - abort install" \
      /SD IDNO IDYES write_shim IDNO skip_shim
    Abort  ; Cancel pressed.

  write_shim:
    CreateDirectory "${TL_SHIM_DIR}"
    ; Resolve the new AddaxAI exe. Per-user installs land in
    ; %LOCALAPPDATA%\Programs\AddaxAI; per-machine installs land in
    ; %ProgramFiles%\AddaxAI. The shim probes both at run time so it
    ; works regardless of which install scope was used.
    FileOpen $0 "${TL_SHIM_PATH}" w
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
  skip_shim:
!macroend

!macro customUnInstall
  ; Remove the Timelapse shim if we own it. We do not unconditionally
  ; nuke the directory because a legacy AddaxAI install may still live
  ; there; only the open.bat we wrote is ours to remove.
  IfFileExists "${TL_SHIM_PATH}" 0 +2
    Delete "${TL_SHIM_PATH}"
  ; Remove the parent directory only if it is empty (i.e. no legacy
  ; AddaxAI files remain).
  RMDir "${TL_SHIM_DIR}"
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
