Below you can find some common error messages and their potential solutions. If you can't figure it out yourself, raise an error in this repository or [email me](mailto:petervanlunteren@hotmail.com) the logfiles, a detailed explanation of what's going on and a way to recreate the error. 

## Ask for debug mode
There must be something going wrong in the backend. Can you run AddaxAI in debug mode and try again? 

How to run in debug mode: https://github.com/PetervanLunteren/AddaxAI/blob/main/markdown/debug_mode.md

It won't solve the issue, but it will print more information about what is going on. If you could then copy-paste the full console text and send it to me - I'll investigate!

Kind regards,

## How to run AddaxAI in debug mode?
It is possible to run AddaxAI in debug mode, where it'll print its output in a console window. That should point us in the right direction if there is an error. How to run it in debug mode depends on your operating system.
<details>
<summary><b>Windows</b></summary>
<br>
 
Open a fresh window of the command prompt and run the following.
```
( "%homedrive%%homepath%\AddaxAI_files\AddaxAI\open.bat" debug ) || ( "%ProgramFiles%\AddaxAI_files\AddaxAI\open.bat" debug ) || ( "%homedrive%\AddaxAI_files\AddaxAI\open.bat" debug )
```
Now try to recreate the error and check the console output for its message.
</details>
<details>
<summary><b>macOS</b></summary>
<br>
 
1. Open your `/Applications/AddaxAI.command` (or wherever you placed it) with a text editor like TextEdit or Visual Studio Code;
2. Outcomment these lines like so:
```bash
# exec 1> $LOCATION_ADDAXAI_FILES/AddaxAI/logfiles/stdout.txt
# exec 2> $LOCATION_ADDAXAI_FILES/AddaxAI/logfiles/stderr.txt
 ```
3. Close and save the file;
4. Start AddaxAI as you would do normally and recreate the error.
</details>
<details>
<summary><b>Linux</b></summary>
<br>
 
1. Open the file `/home/<username>/.AddaxAI_files/AddaxAI/open.command` with a text editor like TextEdit or Visual Studio Code;
2. Outcomment these lines like so:
```bash
# exec 1> $LOCATION_ADDAXAI_FILES/AddaxAI/logfiles/stdout.txt
# exec 2> $LOCATION_ADDAXAI_FILES/AddaxAI/logfiles/stderr.txt
 ```
3. Close and save the file;
4. Start AddaxAI as you would do normally and recreate the error.
</details>

This will show you AddaxAI's output during runtime. If there are any errors, they will most probably show up here. Please copy-paste the entire console output and [email](mailto:petervanlunteren@hotmail.com) it to me, or [raise an issue](https://github.com/PetervanLunteren/AddaxAI/issues/new).

## Where to find the AddaxAI installation files?
All files are located in one folder, called `AddaxAI_files`. Please be aware that it's hidden, so you'll probably have to adjust your settings before you can see it (find out how to: [macOS](https://www.sonarworks.com/support/sonarworks/360003040160-Troubleshooting/360003204140-FAQ/5005750481554-How-to-show-hidden-files-Mac-and-Windows-), [Windows](https://support.microsoft.com/en-us/windows/view-hidden-files-and-folders-in-windows-97fbc472-c603-9d90-91d0-1166d1d9f4b5#WindowsVersion=Windows_11), [Linux](https://askubuntu.com/questions/232649/how-to-show-or-hide-a-hidden-file)).

<details>
<summary>Location on Windows</summary>
<br>
  
```r
# All users
─── 📁Program Files
    └── 📁AddaxAI_files

# Single user
─── 📁Users
    └── 📁<username>
        └── 📁AddaxAI_files
```
</details>

<details>
<summary>Location on macOS</summary>
<br>
  
```r
─── 📁Applications
    └── 📁.AddaxAI_files
```
</details>

<details>
<summary>Location on Linux</summary>
<br>
  
```r
─── 📁home
    └── 📁<username>
        └── 📁.AddaxAI_files
```
</details>

## `Local variable 'elapsed time' referenced before assignment`
This error message can be thrown when trying to deploy a model. It basically means that the model outputs some unexpected text. First of all, try if the model deploys succesfully over the test images supplied [here](https://github.com/PetervanLunteren/AddaxAI#test-your-installation). Check if your preloaded models are correctly downloaded. Sometimes protection software prevent the download of the actual model files. There should be two files named `md_v5a.0.0.pt` and `md_v5b.0.0.pt` in `\AddaxAI_files\pretrained_models\`. You can find the location of the AddaxAI_files [here](https://github.com/PetervanLunteren/AddaxAI#uninstall). If that isn't the problem, the logfiles should point you in the right direction.

## ImportError: cannot import name 'generate_parser' from 'conda.cli.main'
"I figured the error was due to a conflict between mamba and conda. So after uninstalling mamba the installation proceeded without any more issues."

## `SSL certificate problem: unable to get local issuer certificate` -> git
It seems there is something going on with your SSL certificates. Most probably that is due to some security setting (VPN, firewall, proxy, etc.) which prevents it from downloading the specific package. Most of the time this can be fixed by simply disabling the VPN/firewall/proxy and reinstalling AddaxAI. If that doesn't work, try switching internet networks, from WiFi to ethernet, or vice versa. If all that doesn't work, you can disable SSL verification temporarily. If you want that, open a new console window and execute

`PATH_TO_GIT\Git\cmd\git.exe config --global http.sslVerify false`

then drag and drop the install.bat file in the same window and press enter. Please note that this solution opens you to attacks like man-in-the-middle attacks. Therefore turn on verification again after installing AddaxAI, by executing

`PATH_TO_GIT\Git\cmd\git.exe config --global http.sslVerify true`

Hope this helps!

## `CondaSSLError: OpenSSL appears to be unavailable on this machine.` -> conda
If you see this message popping up somewhere, there is something going on with your SSL certificates. Most probably that is due to some security setting (VPN, firewall, proxy, etc.) which prevents it from downloading the specific package. Most of the time this can be fixed by simply disabling the VPN/firewall/proxy and reinstalling AddaxAI. If that doesn't work, try switching internet networks, from WiFi to ethernet, or vice versa. If this doesn't work, it would probably be better to have somebody from IT look at the situation. It's possible to disable SSL verification temporarily. If you want that, open a new console window and execute

`conda config --set ssl_verify false`

then drag and drop the install.bat file in the same window and press enter. Please note that this solution opens you to attacks like man-in-the-middle attacks. Therefore turn on verification again after installing AddaxAI, by executing

`conda config --set ssl_verify true`

If you see `'conda' is not recognized as an internal or external command`, you'll have to explicitly tell you computer where it can find the `conda` executable, by changing the commands to

`path\to\conda\installation\Scripts\conda.exe config --set ssl_verify true`

Hope this helps!

Cheers,

## `ValueError: path is on mount '...', start on mount '...'`
This is a Windows error message and means that your training data is located on a different drive than AddaxAI. For example, it would say `ValueError: path is on mount 'C:', start on mount 'F:'`, which means that the training data is located on your `F:` drive, while AddaxAI is located on the `C:` drive. The sollution is to get everything on the same drive. So either install AddaxAI on the `F:` drive, or move the training data to the `C:` drive. 

## `PackagesNotFoundError: The following packages are not available from current channels`
It looks like it crashes because it can't access certain package repositories. I'm not sure, but my guess is that some kind of protection software (firewall, VPN, antivirus, proxy settings, etc.) might be blocking conda from accessing the required channels. Company computers often have protection software like this enabled. Could this be the case? If possible, try the AddaxAI installation again with the protection software (temporarily) disabled. Another possible solution is to run the conda installation with administrator privileges (if possible). Right-click on the installation script and select "Run as administrator."

Can you check if you have access to the conda website? You can do so by visiting https://conda.anaconda.org/pytorch/win-64/repodata_from_packages.json. If you have access, you should see a white screen with text like so:

```
{
  "info": {
    "subdir": "win-64"
  },
  "packages": {
    "cuda100-1.0-0.tar.bz2": {
      "build": "0",
      "build_number": 0,
      "depends": [],
      "md5": "f0f11be74e3d0a57b23b5e066e20155e",
      "name": "cuda100",
      "sha256": "79997bad0cc6ca7c2ec7c6f214d8a8361d4bcefdbc81114a75084b7b7d928bf8",
      "size": 1943,
      "subdir": "win-64",
      "timestamp": 1544163846173,
      "track_features": "cuda100",
      "version": "1.0"
    },
```

Let me know how that goes!


## `Exception: '>=' not supported between instances of 'RuntimeError' and 'int'`
Solution coming up...

## Example email for faulty installation

It seems there was an issue during the installation. In most cases, such issues are caused by security settings blocking certain downloads. This is often due to security software such as VPNs, proxy servers, firewalls, or antivirus applications, which are particularly strict on company-managed computers. Could you please check if any of these security settings are enabled on your computer? Additionally, the security settings might be active on the internet connection you're using. If your connection is managed by a university, government, or other organization, it could be the source of the issue. If possible, try switching to a different network (e.g., use your home network instead), or switch between WiFi and ethernet to see if this resolves the problem.

In order to debug, could you please:
1. Reboot your computer.
2. If possible, temporarily disable the protection software.
3. Repeat the AddaxAI installation using the latest installation file: https://addaxdatascience.com/addaxai/#install.
4. Copy-paste the entire console output and save it to a text file. You might need to send it to me, if it turns out AddaxAI won’t function properly.
5. Double-click the shortcut file to open AddaxAI. Have some patience. The first time opening might take about a minute due to script compiling.
6. If that doesn’t work, try opening it in debug mode: https://github.com/PetervanLunteren/AddaxAI/blob/main/markdown/errors.md#how-to-run-addaxai-in-debug-mode
7. Now you should see some output in the console. Again, copy-paste the entire console output, add it to your log file, and send it to me.

I’ll take a look and help you to get AddaxAI running!

## `ModuleNotFoundError: No module named 'git'`
So, I managed to get it to install. Not entirely sure what it was but it is now working. I think what I did was uninstall everything and manually delete everything to do with miniforge, anaconda etc. that wasn’t removed during uninstall. As this is a work laptop I was unable to disable security. (see email Sam, Apr 4, 2024, 10:54 PM)

-----
I'm not sure why, but it seems like it doesn't recognise your git command. You have provided the folder 'C:\Users\mlc662\AppData\Local\Programs\Git' as your git installation, but for some reason that doesn't work. 

I think the easiest would be to add an extra installation of git via this link and provide that folder when installing AddaxAI. Chances are that if you accept all default options, AddaxAI will locate it automatically. 

## `ImportError: DLL load failed while importing _ctypes: The specified module could not be found.`
### Option 1
It looks like your miniforge installation was corrupted in one way or another. Perhaps due to an unstable internet connection, but who knows. A reinstall with stable internet should do the trick, but please note that all virtual environments created through miniforge will be lost. That is, if you installed miniforge just for AddaxAI, there is nothing to worry about. I'm just saying that if you're using miniforge for other applications too, you should double check before uninstalling it. 

### Option 2
It looks like you have two conda distributions on your machine, namely "C:\ProgramData\anaconda3" and "C:\ProgramData\miniforge3". These two might be interfering with each other. Try uninstalling one of them and retry the AddaxAI installation. I'm guessing that you installed miniforge3 especially for AddaxAI, right? Then it is safe to uninstall it again. Follow the steps below.
1. Double click the file "C:\ProgramData\miniforge3\Uninstall-Miniforge3.exe";
2. Follow the steps provided to uninstall;Leave Anaconda3 as it is and do not reinstall Miniforge3;
3. Repeat the AddaxAI installation with the latest install file: https://addaxdatascience.com/addaxai/#install
4. It will probably find your anaconda3 installation automatically, but perhaps you'll have to provide the path when prompted: "C:\ProgramData\anaconda3".

Hope this works! Let me know how it goes.

## `PermissionError: [Errno 13] Permission denied: 'C:\\Users\\<>username\\AppData\\Local\\conda\\conda\\Cache\\notices\\notices.cache'`

This looks like conda can't access a certain file, which might be due to the fact that it is being used by another program. Could you restart your computer and try again? Make sure you are executing the latest install file: https://addaxdatascience.com/addaxai/#install

## `RuntimeError: CUDA error: no kernel image is available for execution on the device CUDA kernel errors might be asynchronously reported at some other API call so the stacktrace below might be incorrect.`
Not sure what the solution is yet, but it looks like there might be a package conflict with CUDA and TORCH. See: https://stackoverflow.com/questions/69968477/runtimeerror-cuda-error-no-kernel-image-is-available-for-execution-on-the-devi

Possible solution is running the commands below in a command prompt window:
1. `<path to conda dir>\Scripts\conda --version`
2. `<path to conda dir>\Scripts\conda activate addaxaicondaenv-base`
3. `pip install --upgrade torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html`

Or the following in an anaconda or miniforge prompt:
1. `conda --version`
2. `conda activate addaxaicondaenv-base`
3. `pip install --upgrade torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html`

## `undefined symbol: iJIT_NotifyEvent`
This is a bug that Dan Morris warned me about. See email from May 7th. This is what he wrote about it:

_[This recent PyTorch bug](https://github.com/pytorch/pytorch/issues/123097) may impact you, even if you don't change anything about your dependencies.  Worse, it may not impact you on any machines you regularly test on, but still impact user machines.  It appears that some packages (e.g. numpy and pytorch) take an unpinned dependency on mkl, so even if you pin every package (like the MD conda env file does), on a new machine, you might get a new version of the mkl package, which suddenly became incompatible with even older versions of PyTorch, when certain functions are called.  The solution, if it comes up, is to pin mkl <= 2024.0 at the beginning of any environment file; I've made this change to the MD environment file [here](https://github.com/agentmorris/MegaDetector/blob/main/envs/environment-detector.yml#L13).  I don't know that you need to rush to do anything, but if users report "undefined symbol: iJIT_NotifyEvent" errors, this is the fix._

## `Runtimetrror: CODA error: no kernel image is available for execution on the device CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorred.`
This has to do with CUDA not being compatible with PyTorch in some mystical way. There are a few things we can try to dial in on the problem. 

1. Does AddaxAI work when you disable the GPU (see attached screenshot)?
2. Perhaps this has to do with your GPU version or driver. What kind of GPU do you have? Make sure you have a recent driver installed, then reboot.
3. Does it work on a different computer?

Let me know how that goes!

<img width="1306" alt="Screenshot 2024-05-09 at 09 08 27" src="https://github.com/PetervanLunteren/AddaxAI/assets/85185478/993dff48-f90c-4eb0-9590-0a3c930d8536">

## CondaSSLError: OpenSSL appears to be unavailable on this machine. OpenSSL is required to download and install packages.

There seems to be some issue with SSL certificates. Let's see if we can fix it by installing openSSL. Feel free to do some research online, but I think that this install (Win64 OpenSSL v3.3.1) would be good:

https://slproweb.com/download/Win64OpenSSL-3_3_1.exe

After installing that, reboot your computer and try to install AddaxAI again. If that doesn't work right away, try opening an "anaconda prompt", "miniforge prompt", or "miniconda prompt" (search for "anaconda prompt", "miniforge prompt", or "miniconda prompt" (depending on which conda distribution you have installed on you machine) - this is not the same as the "command prompt") and execute the following command:
```
conda update openssl
```
Then reboot, and try to install AddaxAI again.

Let me know how that goes!

_A user reinstalled Anaconda and it worked_

## `The system cannot find the batch label specified <name_of_label>`
As it turn out, if the end-of-line break types are not set to CRLF, the batch files can have unexpected behaviours. See https://stackoverflow.com/questions/232651/why-the-system-cannot-find-the-batch-label-specified-is-thrown-even-if-label-e

Double check if the eol type is set to CRLF:  https://superuser.com/a/790319/1741678

## `LibMambaUnsatisfiableError: Encountered problems while solving: package cudatoolkit-11.3.1-h280eb24_10 has constraint __cuda >=11 conflicting with __cuda-8.0-0`
There seems to be an issue with the Cuda version on our computer. Can you try the following:
1. Disable VPN, firewall, proxy, etc if possible.
2. Install Cuda 11.3 from https://developer.nvidia.com/cuda-11.3.0-download-archive
3. Make sure you have a recent driver installed: https://www.nvidia.com/download/index.aspx
4. Reboot your computer
5. Download latest AddaxAI install file from https://addaxdatascience.com/addaxai/
6. Execute the install script with admin access if possible


## GPU out of memory

The error message is saying that the GPU is out of memory. In other words, it can't handle the workload. Apart from buying new hardware, there are a few things you can do.
1. Make sure you have a recent driver installed, then reboot: https://www.nvidia.com/download/index.aspx
2. Increase the paging file size. See FAQ 6 - point 1: https://github.com/PetervanLunteren/AddaxAI/blob/main/markdown/FAQ.md#faq-6-what-should-i-do-when-i-run-out-ofmemory. Please note that the rest of these FAQs are for model training, so are not necessarily applicable for your situation when deploying an existing model. 
3. Decrease the image size from 1280 to 640 at the 'Use custom image size' option in advanced mode. Please note that this will drastically decrease accuracy. This is generally not advised.
4. Select the 'Disable GPU processing' option in advanced mode. Processing will go slow, but at least it won't crash.

## New install method

This error usually occurs when the internet signal speed is too slow or unstable. Are you by chance on a weak wifi network? If possible, try the installation again on a fibre internet connection, or perhaps on a different, stronger, wifi network. If you're using a VPN, try disconnecting from it.

Otherwise it might be due to security-related issues (firewall, VPN, antivirus, proxy settings, etc.), which are common when working on government, university, or company computers. Could this be the case? If possible, try the AddaxAI installation again with the protection software (temporarily) disabled.

If that doesn't work or is not possible, I’ve updated the installation method that should bypass these restrictions. Please note that this is still in the Beta phase, so any feedback you can provide would be greatly appreciated. You can find the new installation guide here: https://addaxdatascience.com/addaxai-windows/

## Download from ZIP file

This error usually occurs when the internet signal speed is too slow or unstable. Are you by chance on a weak wifi network? If possible, try the installation again on a fibre internet connection, or perhaps on a different, stronger, wifi network. If you're using a VPN, try disconnecting from it.

Otherwise it might be due to security-related issues (firewall, VPN, antivirus, proxy settings, etc.), which are common when working on government, university, or company computers. Could this be the case? If possible, try the AddaxAI installation again with the protection software (temporarily) disabled. 

If that doesn't work or is not possible, I’ve prepared an alternative installation method that should bypass these restrictions. Please note that this is still in the Beta phase, so any feedback you can provide would be greatly appreciated.

You can find the alternative installation guide here:
https://github.com/PetervanLunteren/AddaxAI/blob/main/markdown/install-from-zip/windows.md

Let me know how it works for you, and feel free to reach out if you have any questions or encounter any issues.

Best regards,

## Train your own model inquiry

Glad to hear AddaxAI can help you in your work!

I used to have a train feature inside AddaxAI, but I've removed that some time ago. Although it was technically working, the results weren't very good. Science has since moved away from training detection models to identify species, to training image classifiers to be used in conjunction with detection models. The old AddaxAI way is training detection models. That being said, you can still do that if you want. A tutorial can be found here (https://drive.google.com/drive/folders/12Qn05KE7TXYhDKLHGSKz1uG-toep9sqx?usp=sharing), and you can downgrade your AddaxAI version to 4.3 via the link in the FAQ section (https://addaxdatascience.com/addaxai/#FAQ).

That being said, the better way is to train an image classifier. For this I've started a collaboration with some folks from the University of Tasmania. They provide a pipeline on training species identification models, which then can be loaded into AddaxAI. I'll admit that it can be a bit daunting, since you'll need to at least have an understanding of Python and Docker. If you want to go down this path, I advise you to first do the Quick Start Vignette (https://github.com/zaandahl/mewc/blob/master/vignette.md).
1. How to train: https://github.com/zaandahl/mewc
2. How to load: https://github.com/PetervanLunteren/AddaxAI/blob/main/markdown/MEWC_integration.md

If you prefer, you can also outsource it and let Addax Data Science develop a species identification model for you. These models will be deployable via AddaxAI. More information here: https://addaxdatascience.com/species-recognition-models/ We'll work together on what the best way is to annotate the images and divide them into classes. I can also pull camera trap images from public repositories of comparable ecological studies if we need additional images to strengthen the model. 

Here are some similar projects I've done in the past, which might give you an idea.
1. Namibia: https://addaxdatascience.com/projects/2023-01-dlc/
2. New Zealand: https://addaxdatascience.com/projects/#2024-06-NZF
3. Warning system for wolves: https://addaxdatascience.com/projects/#project-wolf

I’m also working on species identification models for Hawaii (for the USDA), Arizona (for the University of Arizona), British Columbia, and Scotland, though these are still in development.

Anyway, hope this helps!

Kind regards,

## Update miniforge

It seems that the problem might be due to the miniforge installation. A simple reinstall might do the trick!

Could you please:

1. Reboot your computer.
2. If possible, temporarily disable the protection software.
3. Uninstall existing miniforge by checking if there’s an entry in the Windows "Add or Remove Programs" section (use the search bar at 'Start'). If it appears there, uninstall it using the standard process.
4. Delete the miniforge folder if there are still files inside. Search your window explorer for 'miniforge3' (it is usually located at 'C:\Users\<username>\miniforge3\' or 'C:\ProgramData\miniforge3') and manually delete if there are still residual files left over.
5. Download miniforge using this installer: https://github.com/conda-forge/miniforge/releases/download/24.9.0-0/Miniforge3-Windows-x86_64.exe
6. Execute the installer and follow the steps. You can leave all settings as the default values. If you see a 'Windows protected your PC' warning, you may need to click 'More info' and 'Run anyway'.
7. Reboot your computer after it finishes successfully.
    Repeat the AddaxAI installation using the latest installation file: https://addaxdatascience.com/addaxai/#install.
8. Copy-paste the entire console output and save it to a text file. You might need to send it to me, if it turns out AddaxAI won’t function properly.
9. Double-click the shortcut file to open AddaxAI. Have some patience. The first time opening might take about a minute due to script compiling.
10. If that doesn’t work, try opening it in debug mode: https://github.com/PetervanLunteren/AddaxAI/blob/main/markdown/errors.md#how-to-run-addaxai-in-debug-mode
11. Now you should see some output in the console. Again, copy-paste the entire console output, add it to your log file, and send it to me.

I’ll take a look and help you to get AddaxAI running!

# Manually remove all addaxai files
I'm not sure what is going on, but I believe a hard reset might do the trick. 
1. Reboot your computer.
2. Open your C: drive in file explorer and search for "AddaxAI_files" and manually remove this folder. Make sure hidden items are shown.

<img width="764" alt="Screenshot 2024-10-30 at 08 37 36" src="https://github.com/user-attachments/assets/80bc8f34-e3d2-4495-9885-ab532944aaa7">

4. Now use the search term "addaxaicondaenv-" and remove all those folders.

<img width="766" alt="Screenshot 2024-10-30 at 09 41 20" src="https://github.com/user-attachments/assets/e04cd86b-2e59-4fa1-a5da-53cb95642952">

6. Reboot your computer again.
7. If possible, temporarily disable the protection software.
8. Repeat the AddaxAI installation using the latest installation file: https://addaxdatascience.com/addaxai/#install.
9. Copy-paste the entire console output and save it to a text file. You might need to send it to me, if it turns out AddaxAI won’t function properly.
10. Double-click the shortcut file to open AddaxAI. Have some patience. The first time opening might take about a minute due to script compiling.
11. If that doesn’t work, try opening it in debug mode: https://github.com/PetervanLunteren/AddaxAI/blob/main/markdown/errors.md#how-to-run-addaxai-in-debug-mode
12. Now you should see some output in the console. Again, copy-paste the entire console output, add it to your log file, and send it to me.

I’ll take a look and help you to get AddaxAI running!

# Timelapse command to run AddaxAI in Timelapse mode
```
(cd /d %homedrive%%homepath% && "%homedrive%%homepath%\AddaxAI_files\AddaxAI\open.bat" timelapse "C:\Users\smart\Desktop\test_images") || (cd /d %ProgramFiles% && "%ProgramFiles%\AddaxAI_files\AddaxAI\open.bat" timelapse "C:\Users\smart\Desktop\test_images")
```

```
(cd /d %homedrive%%homepath% && "%homedrive%%homepath%\EcoAssist_files\EcoAssist\open.bat" timelapse "C:\Users\smart\Desktop\test_images") || (cd /d %ProgramFiles% && "%ProgramFiles%\EcoAssist_files\EcoAssist\open.bat" timelapse "C:\Users\smart\Desktop\test_images")
```

```
(cd /d %homedrive%%homepath% && "%homedrive%%homepath%\AddaxAI_files\AddaxAI\open.bat" timelapse "C:\Users\smart\Desktop\test_images") || (cd /d %ProgramFiles% && "%ProgramFiles%\AddaxAI_files\AddaxAI\open.bat" timelapse "C:\Users\smart\Desktop\test_images") ||  (cd /d %homedrive%%homepath% && "%homedrive%%homepath%\EcoAssist_files\EcoAssist\open.bat" timelapse "C:\Users\smart\Desktop\test_images") || (cd /d %ProgramFiles% && "%ProgramFiles%\EcoAssist_files\EcoAssist\open.bat" timelapse "C:\Users\smart\Desktop\test_images")
```
