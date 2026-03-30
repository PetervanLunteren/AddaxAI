
# Manually download environments (Windows)

Below are instructions for Windows users who are unable to download virtual environments through AddaxAI due to network restrictions (e.g., university or government networks). This should be plan B, as the automatic download is much quicker. If you haven't tried the automatic download yet, try that first and check your firewall, proxy, and VPN settings.

## Which environment do you need?

AddaxAI uses separate virtual environments for different model types. The error window will tell you which environment failed to download. Here is a list of the available environments:

| Environment | Used by |
|---|---|
| `pytorch` | Most classification models (e.g., AI Pua&#699;a, AWC135, DeepFaune) |
| `tensorflow-v1` | Older TensorFlow-based classifiers (e.g., TropiCam-AI) |
| `tensorflow-v2` | Newer TensorFlow/Keras classifiers |
| `speciesnet` | Google SpeciesNet |

## Step-by-step instructions

In the instructions below, replace `{name}` with the environment name from the table above (e.g., `pytorch`, `speciesnet`).

1. Go to the AddaxAI releases page: [https://github.com/PetervanLunteren/AddaxAI/releases/latest](https://github.com/PetervanLunteren/AddaxAI/releases/latest)

2. Scroll down to the **Assets** section at the bottom of the page.

3. Download all files that start with `windows-env-{name}`. Depending on the file size, the environment may be a single file or split into multiple parts:
   - Single file: `windows-env-{name}.zip`
   - Split files: `windows-env-{name}.zip.001`, `windows-env-{name}.zip.002`, etc.

   Download all of them.

4. **If you downloaded multiple parts**, you need to combine them into one file first. Open a Command Prompt window (click the Start menu, type `cmd`, and press Enter) and run the following command. Replace `{name}` with your environment name:
   ```
   copy /b "%USERPROFILE%\Downloads\windows-env-{name}.zip.001"+"%USERPROFILE%\Downloads\windows-env-{name}.zip.002" "%USERPROFILE%\Downloads\windows-env-{name}.zip"
   ```
   If there are 3 or more parts, add them all with `+` signs. For example, with 3 parts:
   ```
   copy /b "%USERPROFILE%\Downloads\windows-env-{name}.zip.001"+"%USERPROFILE%\Downloads\windows-env-{name}.zip.002"+"%USERPROFILE%\Downloads\windows-env-{name}.zip.003" "%USERPROFILE%\Downloads\windows-env-{name}.zip"
   ```
   If you downloaded a single `.zip` file (not split), skip this step.

5. Navigate to your `AddaxAI_files` folder. On Windows, this is usually located at `C:\Users\<username>\AddaxAI_files`. See [this page](https://github.com/PetervanLunteren/AddaxAI/blob/main/markdown/AddaxAI_files_location.md) if you're not sure where it is.

6. Open the `envs` subfolder inside `AddaxAI_files`. If it doesn't exist, create it.

7. Right-click the `.zip` file in your Downloads folder > **Extract All...** > set the destination to your `envs` folder, e.g.:
   ```
   C:\Users\<username>\AddaxAI_files\envs
   ```

8. Verify that the folder structure is correct. You should see:
   ```
   C:\Users\<username>\AddaxAI_files\envs\env-{name}\python.exe
   ```
   For example, if you downloaded the `pytorch` environment:
   ```
   C:\Users\<username>\AddaxAI_files\envs\env-pytorch\python.exe
   ```
   If `python.exe` is nested inside an extra subfolder (e.g., `envs\env-pytorch\env-pytorch\python.exe`), move the inner folder's contents up one level so `python.exe` is directly inside `env-{name}\`.

9. Close AddaxAI completely and reopen it.

_______________________________________________________________________
<details>
<summary><b>Tip: download on another computer</b></summary>

<br>

If your network blocks GitHub downloads entirely, you can download the files on another computer (e.g., at home, on a mobile hotspot, or on a less restricted network) and transfer them to your work computer using a USB drive or other file transfer method.

1. Download the environment file(s) on the other computer.
2. Copy them to a USB drive.
3. On your work computer, follow steps 4-9 above.

</details>

<details>
<summary><b>Alternative: download using the command line</b></summary>

<br>

If your internet connection is unstable and the browser download keeps failing, you can use the command line instead. This supports resuming interrupted downloads. Open a Command Prompt and run the following commands, replacing `{name}` with your environment name (e.g., `pytorch`).

If the environment is a single file:
```
curl --retry 5 --retry-delay 10 --continue-at - -L -o "%USERPROFILE%\Downloads\windows-env-{name}.zip" "https://github.com/PetervanLunteren/AddaxAI/releases/latest/download/windows-env-{name}.zip"
```

If the environment is split into parts (check the releases page to see which files are available):
```
curl --retry 5 --retry-delay 10 --continue-at - -L -o "%USERPROFILE%\Downloads\windows-env-{name}.zip.001" "https://github.com/PetervanLunteren/AddaxAI/releases/latest/download/windows-env-{name}.zip.001"
curl --retry 5 --retry-delay 10 --continue-at - -L -o "%USERPROFILE%\Downloads\windows-env-{name}.zip.002" "https://github.com/PetervanLunteren/AddaxAI/releases/latest/download/windows-env-{name}.zip.002"
copy /b "%USERPROFILE%\Downloads\windows-env-{name}.zip.001"+"%USERPROFILE%\Downloads\windows-env-{name}.zip.002" "%USERPROFILE%\Downloads\windows-env-{name}.zip"
```

If the download is interrupted, you can resume it by running the same `curl` command again. After downloading, continue with step 5 above.

</details>
