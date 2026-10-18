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
; 4. Keep the app out of the data folder. See "The install folder never
;    overlaps the data folder" below.
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

; ---------------------------------------------------------------------
; The install folder never overlaps the data folder
; ---------------------------------------------------------------------
;
; The data folder (database, backups, models, envs, logs) is
; %ADDAXAI_USER_DATA_DIR% when set, else $PROFILE\AddaxAI, the same
; rule as resolveUserDataDir in src/main.ts. The installer lets the user
; pick any install folder, and electron-builder's uninstaller ends with
; `RMDir /r $INSTDIR`. An update runs the OLD version's uninstaller
; first, so an app installed in the data folder deletes all of it on the
; next update. That happened on 2026-10-13: 7.8 to 7.14, one user lost
; every project and every backup.
;
; One rule: the install folder and the data folder are never the same,
; and neither lies inside the other. AddaxAIOverlapsDataDir is the only
; implementation, checked in three places:
;
;   - customInit, for an install that is already in the data folder:
;     its uninstall registry entry is removed before electron-builder's
;     uninstallOldVersion runs, so that uninstaller is never called. Its
;     own "Uninstall AddaxAI.exe" goes too, since that file alone could
;     still wipe the folder later. The other old app files stay; they
;     are harmless. An interactive run says so and proposes the
;     default folder on the folder page.
;   - customInit again, for a silent install into the data folder:
;     refused with exit code 3 and a reason in
;     %TEMP%\AddaxAI-Setup-error.txt. Never redirected somewhere else.
;   - the guard page after the folder page, for an interactive install:
;     the user is sent back to pick another folder.
;
; No customRemoveFiles: with these checks, `RMDir /r $INSTDIR` only ever
; runs on a folder that holds no data.

!ifndef BUILD_UNINSTALLER
  !include LogicLib.nsh
  !include nsDialogs.nsh

  ; "1" when customInit found the previous install inside the data folder.
  Var AddaxAIPrevInDataDir
!endif

; The functions live in customHeader rather than at the top of this file:
; electron-builder writes this file into the script header, possibly
; before its plugin directory line, and System::Call needs that line.
; customHeader is inserted after every include.
!macro customHeader
!ifndef BUILD_UNINSTALLER
; Push PATH, Call, Pop the full path without a trailing backslash.
; kernel32's GetFullPathNameW, because NSIS's own GetFullPathName fails
; on a folder that does not exist yet, which every new install folder is.
Function AddaxAINormalizePath
  Exch $0
  Push $1
  Push $2
  System::Call 'kernel32::GetFullPathNameW(w r0, i ${NSIS_MAX_STRLEN}, w .r1, p 0) i .r2'
  ${If} $2 <> 0
    StrCpy $0 $1
  ${EndIf}
  StrCpy $1 $0 1 -1
  ${If} $1 == "\"
    StrCpy $0 $0 -1
  ${EndIf}
  Pop $2
  Pop $1
  Exch $0
FunctionEnd

; Push CHILD, push PARENT, Call, Pop "1" when CHILD is PARENT or lies
; inside it. Both normalised. LogicLib's == is case-insensitive, like
; Windows paths. The backslash check keeps C:\AddaxAI2 out of C:\AddaxAI.
Function AddaxAIIsSameOrInside
  Exch $1
  Exch
  Exch $0
  Push $2
  Push $3
  StrLen $2 $1
  StrCpy $3 $0 $2
  ${If} $3 == $1
    StrCpy $3 $0 1 $2
    ${If} $3 == ""
    ${OrIf} $3 == "\"
      StrCpy $0 "1"
    ${Else}
      StrCpy $0 ""
    ${EndIf}
  ${Else}
    StrCpy $0 ""
  ${EndIf}
  Pop $3
  Pop $2
  Exch $0
  Exch
  Pop $1
FunctionEnd

; Push PATH, Call, Pop "1" when PATH and the data folder overlap.
Function AddaxAIOverlapsDataDir
  Exch $0
  Push $1
  Push $2
  Push $0
  Call AddaxAINormalizePath
  Pop $0

  ; The data folder. An ADDAXAI_USER_DATA_DIR that is not absolute
  ; (no "X:\" and no leading slash) is ignored, as resolveUserDataDir
  ; ignores it.
  ReadEnvStr $1 ADDAXAI_USER_DATA_DIR
  StrCpy $2 $1 1 1
  ${If} $2 == ":"
    StrCpy $2 $1 1 2
  ${Else}
    StrCpy $2 $1 1
  ${EndIf}
  ${If} $2 != "\"
  ${AndIf} $2 != "/"
    StrCpy $1 "$PROFILE\AddaxAI"
  ${EndIf}
  Push $1
  Call AddaxAINormalizePath
  Pop $1

  Push $0
  Push $1
  Call AddaxAIIsSameOrInside
  Pop $2
  ${If} $2 != "1"
    Push $1
    Push $0
    Call AddaxAIIsSameOrInside
    Pop $2
  ${EndIf}

  StrCpy $0 $2
  Pop $2
  Pop $1
  Exch $0
FunctionEnd
!endif
!macroend

; Forget a previous install that sits in the data folder, so
; uninstallOldVersion finds no UninstallString and runs nothing. The
; InstallLocation value goes too, or the install-mode page proposes the
; data folder again and the next run treats it as an existing install.
!macro AddaxAIReleasePreviousInstall ROOT
  Push $0
  Push $1
  ReadRegStr $0 ${ROOT} "${INSTALL_REGISTRY_KEY}" InstallLocation
  ${If} $0 != ""
    Push $0
    Call AddaxAIOverlapsDataDir
    Pop $1
    ${If} $1 == "1"
      DeleteRegKey ${ROOT} "${UNINSTALL_REGISTRY_KEY}"
      !ifdef UNINSTALL_REGISTRY_KEY_2
        DeleteRegKey ${ROOT} "${UNINSTALL_REGISTRY_KEY_2}"
      !endif
      DeleteRegValue ${ROOT} "${INSTALL_REGISTRY_KEY}" InstallLocation
      Delete "$0\${UNINSTALL_FILENAME}"
      StrCpy $AddaxAIPrevInDataDir "1"
    ${EndIf}
  ${EndIf}
  Pop $1
  Pop $0
!macroend

!macro customInit
  ; A silent install has no folder page to send the user back to, so it
  ; stops here, before anything is changed. Running it again without /D
  ; fails the same way; it never lands somewhere the caller did not ask.
  ${If} ${Silent}
    Push $INSTDIR
    Call AddaxAIOverlapsDataDir
    Pop $0
    ${If} $0 == "1"
      FileOpen $0 "$TEMP\AddaxAI-Setup-error.txt" w
      FileWrite $0 "AddaxAI was not installed. The install folder $INSTDIR is the AddaxAI data folder or overlaps it. That folder holds the database, models and backups, and an update or uninstall would delete it. Run the installer again with /D set to another folder, for example /D=$LOCALAPPDATA\Programs\AddaxAI$\r$\n"
      FileClose $0
      SetErrorLevel 3
      Quit
    ${EndIf}
  ${EndIf}

  ; HKCU only in the outer instance: an elevated inner instance runs
  ; .onInit again, possibly as another account. HKLM only when we may
  ; write it.
  ${IfNot} ${UAC_IsInnerInstance}
    !insertmacro AddaxAIReleasePreviousInstall HKCU
  ${EndIf}
  ${If} ${UAC_IsAdmin}
    !insertmacro AddaxAIReleasePreviousInstall HKLM
  ${EndIf}

  ; Tell the user why the folder changes, once (not again in an elevated
  ; inner instance), then propose the default folder on the folder page,
  ; where they still see it and can change it.
  ${If} $AddaxAIPrevInDataDir == "1"
  ${AndIfNot} ${Silent}
  ${AndIfNot} ${UAC_IsInnerInstance}
    Push $INSTDIR
    Call AddaxAIOverlapsDataDir
    Pop $0
    ${If} $0 == "1"
      StrCpy $INSTDIR "$LOCALAPPDATA\Programs\${APP_FILENAME}"
    ${EndIf}
    MessageBox MB_OK|MB_ICONINFORMATION "AddaxAI is installed in your data folder, next to your projects, models and backups. Updating it there would delete them, so AddaxAI moves to a new folder:$\r$\n$\r$\n$INSTDIR$\r$\n$\r$\nThe installer has already selected that folder. Click Next to accept it, or pick another one. Your data is safe and stays where it is."
  ${EndIf}
!macroend

; The guard page, right after the folder page. A plain `Page custom`, not
; a MUI_PAGE_* macro: those would take electron-builder's pending
; MUI_PAGE_CUSTOMFUNCTION_PRE, which belongs to the install page.
; instFilesPre is electron-builder's own "append \AddaxAI" rule, called
; here so the check sees the folder that will really be used; it is
; idempotent, and the install page calls it again.
;
; Next is disabled and the leave function refuses, so Back to the folder
; page is the only way on.
!macro customPageAfterChangeDir
  Page custom AddaxAIGuardCreate AddaxAIGuardLeave

  Function AddaxAIGuardCreate
    Call instFilesPre
    Push $INSTDIR
    Call AddaxAIOverlapsDataDir
    Pop $0
    ${If} $0 != "1"
      Abort
    ${EndIf}

    !insertmacro MUI_HEADER_TEXT "Choose another folder" "$INSTDIR"
    nsDialogs::Create 1018
    Pop $1
    ${If} $1 == error
      Abort
    ${EndIf}
    ${NSD_CreateLabel} 0 0 100% 48u "This folder holds your AddaxAI data: projects, models and backups. AddaxAI cannot be installed there, because an update or uninstall would delete that data.$\r$\n$\r$\nGo back and choose another folder."
    Pop $1
    GetDlgItem $1 $HWNDPARENT 1
    EnableWindow $1 0
    nsDialogs::Show
  FunctionEnd

  Function AddaxAIGuardLeave
    Abort
  FunctionEnd
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
