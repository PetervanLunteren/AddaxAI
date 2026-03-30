
# Manually download model weights

Below are instructions for users who are unable to download model weight files through AddaxAI due to network restrictions (e.g., university or government networks). This should be plan B, as the automatic download is much quicker. If you haven't tried the automatic download yet, try that first and check your firewall, proxy, and VPN settings.

## MegaDetector models

MegaDetector is the detection model that finds animals, people, and vehicles in camera trap images. The weight files are hosted on GitHub, which may be blocked by some restricted networks. Here are the available models:

| Model | Download link | File name | Target folder |
|---|---|---|---|
| MegaDetector 1000 Redwood | [Download](https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-redwood.pt) | `md_v1000.0.0-redwood.pt` | `models/det/MegaDetector 1000 Redwood/` |
| MegaDetector 1000 Spruce | [Download](https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-spruce.pt) | `md_v1000.0.0-spruce.pt` | `models/det/MegaDetector 1000 Spruce/` |
| MegaDetector 5a | [Download](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt) | `md_v5a.0.0.pt` | `models/det/MegaDetector 5a/` |
| MegaDetector 5b | [Download](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5b.0.0.pt) | `md_v5b.0.0.pt` | `models/det/MegaDetector 5b/` |

Note: MegaDetector 5a is pre-installed with AddaxAI. If it is missing, reinstalling AddaxAI may be easier than a manual download.

## Step-by-step instructions

1. Click the download link for the model you need from the table above. The model you need is displayed in the AddaxAI error window.

2. Navigate to your `AddaxAI_files` folder. The location depends on your operating system, see [this page](https://github.com/PetervanLunteren/AddaxAI/blob/main/markdown/AddaxAI_files_location.md) for details.

3. Open the `models/det/` subfolder inside `AddaxAI_files`. If a folder with the exact model name does not already exist, create it. The folder name must match exactly, including spaces. For example: `MegaDetector 1000 Redwood`.

4. Move or copy the downloaded `.pt` file into the model folder. The file name must remain unchanged. For example:
   ```
   AddaxAI_files/models/det/MegaDetector 1000 Redwood/md_v1000.0.0-redwood.pt
   ```

5. Close AddaxAI completely and reopen it. The model should now be available.

_______________________________________________________________________
<details>
<summary><b>Tip: download on another computer</b></summary>

<br>

If your network blocks GitHub downloads entirely, you can download the file on another computer (e.g., at home, on a mobile hotspot, or on a less restricted network) and transfer it to your work computer using a USB drive or other file transfer method.

1. Download the `.pt` file on the other computer.
2. Copy it to a USB drive.
3. On your work computer, follow steps 2-5 above to place it in the correct folder.

</details>

<details>
<summary><b>Alternative: download using the command line</b></summary>

<br>

If your internet connection is unstable and the browser download keeps failing, you can use the command line instead. This supports resuming interrupted downloads. Open a terminal and run the command for the model you need.

**MegaDetector 1000 Redwood:**
```
curl --retry 5 --retry-delay 10 --continue-at - -L -o "md_v1000.0.0-redwood.pt" "https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-redwood.pt"
```

**MegaDetector 1000 Spruce:**
```
curl --retry 5 --retry-delay 10 --continue-at - -L -o "md_v1000.0.0-spruce.pt" "https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-spruce.pt"
```

**MegaDetector 5a:**
```
curl --retry 5 --retry-delay 10 --continue-at - -L -o "md_v5a.0.0.pt" "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"
```

**MegaDetector 5b:**
```
curl --retry 5 --retry-delay 10 --continue-at - -L -o "md_v5b.0.0.pt" "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5b.0.0.pt"
```

If the download is interrupted, run the same command again to resume. After downloading, move the file to the correct model folder as described in steps 2-5 above.

</details>

## Other models

Classification models (for species identification) are hosted on HuggingFace, which is usually not affected by the same network restrictions as GitHub. If a classification model download fails, AddaxAI's error window will show manual download instructions with direct links to the files on HuggingFace. If you still have trouble, feel free to reach out at [peter@addaxdatascience.com](mailto:peter@addaxdatascience.com).
