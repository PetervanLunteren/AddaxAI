; Custom NSIS hook for AddaxAI uninstaller.
;
; The default uninstaller removes the application from Program Files but
; leaves %USERPROFILE%\AddaxAI alone. That folder holds the analysis
; environment (~2 GB), downloaded model weights, the SQLite project DB,
; logs, and thumbnails. Surviving across uninstalls is the right default
; for users who plan to reinstall, but users who actually want a clean
; removal currently have to find that folder by hand.
;
; This macro asks once, after the normal uninstall has run, whether to
; also delete that folder. Default answer is No so reinstall flows keep
; their projects and avoid the 2 GB env redownload.

!macro customUnInstall
  ; Skip the prompt during silent uninstalls (CI, scripted removal).
  IfSilent skip_userdata_removal

  ; Nothing to ask about if the folder isn't present.
  IfFileExists "$PROFILE\AddaxAI\*.*" 0 skip_userdata_removal

  MessageBox MB_YESNO|MB_ICONQUESTION|MB_DEFBUTTON2 \
    "Also delete your AddaxAI user data?$\r$\n$\r$\n$PROFILE\AddaxAI$\r$\n$\r$\nThis includes your projects database, downloaded model weights, the analysis environment (about 2 GB), logs, and thumbnails. Removing it cannot be undone.$\r$\n$\r$\nChoose No to keep your data so a future reinstall picks up where you left off." \
    /SD IDNO IDNO skip_userdata_removal

  RMDir /r "$PROFILE\AddaxAI"

  skip_userdata_removal:
!macroend
