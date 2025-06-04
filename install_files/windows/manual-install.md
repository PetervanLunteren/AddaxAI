
# Steps to manually install AddaxAI

Below are instructions for AddaxAI users who are unable to execute the normal install due to unstable internet, security issues, or something else. Installing AddaxAI should be plan B, as the normal installation is much quicker, performs additional checks, and automates the whole process. If you haven't tried the [normal install](https://addaxdatascience.com/AddaxAI-windows/) yet, please try that first. 

**Follow the steps below to install AddaxAI from a ZIP file.**
1. Download the compressed files using [this link](https://addaxaipremiumstorage.blob.core.windows.net/github-zips/latest/windows/base-install.7z). It might take a while, as it is 4.4 GB. If the internet connection is too unstable or slow for this to work, see the 'Download using the command line for a resume option in unstable internet' alternative at the bottom of this page.
<div align="center"><img width="400" alt="Screenshot 2025-01-26 at 11 28 56" src="https://github.com/user-attachments/assets/e18ab79e-e955-426e-ac99-6cdc3e6a7d80"></div>
<br>
<br>
  
2. Create an `AddaxAI_files` subfolder in your user folder: `C:\Users\<your_username>\AddaxAI_files`

3. Download 7zip via [this link](https://www.7-zip.org/a/7z2408-x64.exe), double-click the `7z2408-x64.exe` file in your Downloads folder, leave the destination folder as it is, and press 'Install'. After a few seconds, it should be installed, and the window can be closed by clicking 'Close'.
<div align="center">
  <img width="400" alt="Screenshot 2024-08-14 at 08 14 04" src="https://github.com/user-attachments/assets/f6ad90e6-1ce3-4b1c-a2fd-d22139408c22">
  <img width="400" alt="Screenshot 2025-01-26 at 11 34 55" src="https://github.com/user-attachments/assets/66155752-863f-4382-a969-97c5bbe907dc">
</div>
<br>
<br>

4. Navigate to the `windows-latest.7z` file. It will be in your Downloads folder.
5. Right-click > (Optionally: 'Show more options') > 7-zip > Extract files...
<div align="center"><img width="800" alt="Screenshot 2024-08-14 at 07 31 51" src="https://github.com/user-attachments/assets/921a6f79-bf9a-4a7c-8137-df339eb9694e"></div>
<br>
<br>

6. Click the `...` button to browse your user folder. The location is important. Choose `C:\Users\<your_username>\AddaxAI_files`.
7. Make sure the checkbox under the 'Extract to' path is unchecked.
8. Choose 'Overwrite without prompt' as 'Overwrite mode'. 
9. Click OK. This process will take about 1–5 minutes. 
<div align="center"><img width="800" alt="Screenshot 2024-08-14 at 07 41 37" src="https://github.com/user-attachments/assets/e040a7b2-8da8-4b33-b347-1e27187ee3a2"></div>
<br>
<br>

10. When the files are extracted, check if AddaxAI works by double-clicking the `C:\Users\<your_username>\AddaxAI_files\AddaxAI\open.bat` file.
<div align="center"><img width="600" alt="Screenshot 2024-08-14 at 07 23 35" src="https://github.com/user-attachments/assets/410577b9-a722-4c1c-9217-3bd0d773aa6f"></div>

<br>
<br>

11. You can create a shortcut file so you don't always have to navigate to the `open.bat` file. If you still have your shortcut file left from the normal install, that will work fine. If not, you can create one by right-clicking on `open.bat` > (Optionally: 'Show more options') > Create shortcut. Then drag and drop that shortcut file to your desktop (or wherever you want to place it) and rename it to 'AddaxAI'.
<br>
<br>

_______________________________________________________________________
<details>
<summary><b>Download using the command line for a resume option in unstable internet</b></summary>

  <br>

Instead of using the web browser to download the files at step 1, you can also use the command line to download the files to your Downloads folder.

<img width="900" alt="Screenshot 2025-01-26 at 12 12 08" src="https://github.com/user-attachments/assets/ef8a286a-ff2f-4a69-8a57-40e4bc4918e2" />

Open up a command prompt window (you might have to search for it via 'Start') and execute the following command. If the download is interrupted, you can resume the download by executing the same command again. If the download has finished, you can continue the install process at step 2 at the top of this page.

  <br>

```
curl --retry 5 --retry-delay 10 --retry-max-time 300 --connect-timeout 20 --continue-at - https://storage.googleapis.com/github-release-files-storage/latest/windows-latest.7z -o "%USERPROFILE%\Downloads\windows-latest.7z"
```

</details>
