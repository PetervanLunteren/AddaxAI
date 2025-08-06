#!/usr/bin/env bash

### Linux install commands 
### The the linux install was too big for the free GitHub runners (as opposed to windows and macos), so
### the linux install is still with the old install method
### Peter van Lunteren, 15 Jan 2025 (latest edit)

# timestamp the start of installation
START_DATE=`date`

# set location var
LOCATION_ADDAXAI_FILES="$HOME/.AddaxAI_files"

# delete previous installation of AddaxAI if present so that it can update
rm -rf $LOCATION_ADDAXAI_FILES && echo "Removed dir '${LOCATION_ADDAXAI_FILES}'"

# early exit if the folder still exists
if [ -d $LOCATION_ADDAXAI_FILES ]; then
    echo "Error: Folder $LOCATION_ADDAXAI_FILES could not be removed. Exiting script...."
    echo "Copy-paste all text in this console window and send it to peter@addaxdatascience.com for further support."
    exit 1
fi

# make dir and change into
mkdir -p $LOCATION_ADDAXAI_FILES
cd $LOCATION_ADDAXAI_FILES || { echo "Could not change directory to ${LOCATION_ADDAXAI_FILES}. Command could not be run. Copy-paste all text in this console window and send it to peter@addaxdatascience.com for further support."; exit 1; }

# log the start
echo "This installation started at: $START_DATE"

# log the platform
echo "This installation is using platform: $PLATFORM"

# log system information
UNAME_A=`uname -a`
if [ "$PLATFORM" = "Apple Silicon Mac" ] || [ "$PLATFORM" = "Intel Mac" ]; then
  MACHINE_INFO=`system_profiler SPSoftwareDataType SPHardwareDataType SPMemoryDataType SPStorageDataType`
elif [ "$PLATFORM" = "Linux" ]; then
  PATH=$PATH:/usr/sbin
  MACHINE_INFO_1=`lscpu`
  MACHINE_INFO_2=`dmidecode`
  MACHINE_INFO_3=""
  SYSTEM_INFO_DIRECTORY="/sys/devices/virtual/dmi/id/"
  if [ -d "$SYSTEM_INFO_DIRECTORY" ]; then
    echo "$SYSTEM_INFO_DIRECTORY does exist."
    cd $SYSTEM_INFO_DIRECTORY
    for f in *; do
      MACHINE_INFO_3+="
      $f = `cat $f 2>/dev/null || echo "***_Unavailable_***"`"
    done
  fi
  MACHINE_INFO="${MACHINE_INFO_1}
  ${MACHINE_INFO_2}
  ${MACHINE_INFO_3}"
fi
echo "uname -a:" 
echo "$UNAME_A" 
echo "" 
echo "System information:" 
echo "$MACHINE_INFO" 
echo "" 

### create folder structure
mkdir -p "${LOCATION_ADDAXAI_FILES}"
mkdir -p "${LOCATION_ADDAXAI_FILES}/envs"
mkdir -p "${LOCATION_ADDAXAI_FILES}/models"
mkdir -p "${LOCATION_ADDAXAI_FILES}/models/det"
mkdir -p "${LOCATION_ADDAXAI_FILES}/models/cls"
mkdir -p "${LOCATION_ADDAXAI_FILES}/models/det/MegaDetector 5a"
mkdir -p "${LOCATION_ADDAXAI_FILES}/yolov5_versions/yolov5_old"
mkdir -p "${LOCATION_ADDAXAI_FILES}/yolov5_versions/yolov5_new"
echo "Hello world!" >> "${LOCATION_ADDAXAI_FILES}/first-startup.txt"

## clone repositories
git clone --depth 1 https://github.com/PetervanLunteren/AddaxAI.git "${LOCATION_ADDAXAI_FILES}/AddaxAI"
rm -rf "${LOCATION_ADDAXAI_FILES}/AddaxAI/.git"
FILE="$LOCATION_ADDAXAI_FILES/AddaxAI/Linux_open_AddaxAI_shortcut.desktop" # create shortcut file

# create desktop shortcut and launcher script
LAUNCHER_SCRIPT="${LOCATION_ADDAXAI_FILES}/AddaxAI/launch_addaxai.sh"

cat << 'EOF' > "$LAUNCHER_SCRIPT"
#!/usr/bin/env bash
CMD_TO_RUN="bash -c '$HOME/.AddaxAI_files/envs/env-base/bin/python $HOME/.AddaxAI_files/AddaxAI/AddaxAI_GUI.py; exec \$SHELL'"

# Check for popular terminals
if command -v konsole >/dev/null 2>&1; then
  konsole -e "$CMD_TO_RUN"
elif command -v gnome-terminal >/dev/null 2>&1; then
  gnome-terminal -- "$CMD_TO_RUN"
elif command -v xfce4-terminal >/dev/null 2>&1; then
  xfce4-terminal --command="$CMD_TO_RUN"
elif command -v alacritty >/dev/null 2>&1; then
  alacritty -e bash -c "'$HOME/.AddaxAI_files/envs/env-base/bin/python $HOME/.AddaxAI_files/AddaxAI/AddaxAI_GUI.py; exec \$SHELL'"
elif command -v kitty >/dev/null 2>&1; then
  kitty --hold bash -c "$HOME/.AddaxAI_files/envs/env-base/bin/python $HOME/.AddaxAI_files/AddaxAI/AddaxAI_GUI.py"
elif command -v xterm >/dev/null 2>&1; then
  # xterm is a common fallback
  xterm -e "$CMD_TO_RUN"
else
  ERROR_MSG="Could not find a known terminal, use your own terminal and run the following command:\n\n$HOME/.AddaxAI_files/envs/env-base/bin/python $HOME/.AddaxAI_files/AddaxAI/AddaxAI_GUI.py"
  
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --text="$ERROR_MSG" --width=400
  elif command -v kdialog >/dev/null 2>&1; then
    kdialog --error "$ERROR_MSG"
  elif command -v xterm >/dev/null 2>&1; then
    xterm -T "AddaxAI - Terminal Not Found" -e "printf '%b' \"$ERROR_MSG\n\n(Press Enter to close this window)\"; read" &
  else
    # This will only be seen if the script is run from an existing terminal, but it's a final fallback.
    printf '%b' "$ERROR_MSG\n" >&2
  fi
  exit 1
fi
EOF

chmod +x "$LAUNCHER_SCRIPT"
echo "Launcher script created."


# create the .desktop file
DESKTOP_FILE_SOURCE="${LOCATION_ADDAXAI_FILES}/AddaxAI/Linux_open_AddaxAI_shortcut.desktop"

cat << EOF > "$DESKTOP_FILE_SOURCE"
[Desktop Entry]
Type=Application
Terminal=false
Name=AddaxAI
Icon=logo_small_bg
Exec=$LAUNCHER_SCRIPT
Categories=Application;
EOF

# Move the final .desktop file to the user's Desktop
mv -f "$DESKTOP_FILE_SOURCE" "$HOME/Desktop/Linux_open_AddaxAI_shortcut.desktop"
# and give it an icon
SOURCE="$LOCATION_ADDAXAI_FILES/AddaxAI/imgs/logo_small_bg.png"
DEST="$HOME/.icons/logo_small_bg.png"
mkdir -p "$HOME/.icons" # create location if not already present
cp $SOURCE $DEST # copy icon to proper location
FILE="$LOCATION_ADDAXAI_FILES/AddaxAI/Linux_open_AddaxAI_shortcut.desktop"
mv -f $FILE "$HOME/Desktop/Linux_open_AddaxAI_shortcut.desktop" # move file and replace
echo "AddaxAI cloned"

git clone https://github.com/agentmorris/MegaDetector.git "${LOCATION_ADDAXAI_FILES}/MegaDetector" || {
      # if this git repo fails to clone, chances are the conda environments will give problems too
      # so better already set the conda settings to accommodate slow internet speeds
      export CONDA_REMOTE_READ_TIMEOUT_SECS=120
      export CONDA_REMOTE_CONNECTIONS=1
      export CONDA_REMOTE_MAX_RETRIES=20
      # some users experience timeout issues due to the large size of this repository
      # if it fails here, we'll try again with a larger timeout value and less checks during cloning
      echo "First attempt failed. Retrying with extended timeout..."
      GIT_HTTP_POSTBUFFER=524288000 git clone --progress \
          --config transfer.fsckObjects=false \
          --config receive.fsckObjects=false \
          --config fetch.fsckObjects=false \
          https://github.com/agentmorris/MegaDetector.git "${LOCATION_ADDAXAI_FILES}/MegaDetector"
}
git -C "${LOCATION_ADDAXAI_FILES}/MegaDetector" checkout a64a8f8c467ae6c87b5ca09f54b1abe502168b50
rm -rf "${LOCATION_ADDAXAI_FILES}/MegaDetector/.git"
mv "${LOCATION_ADDAXAI_FILES}/MegaDetector" "${LOCATION_ADDAXAI_FILES}/cameratraps"
echo "MegaDetector cloned"

git clone https://github.com/ultralytics/yolov5.git "${LOCATION_ADDAXAI_FILES}/yolov5_versions/yolov5_old/yolov5"
git -C "${LOCATION_ADDAXAI_FILES}/yolov5_versions/yolov5_old/yolov5" checkout 868c0e9bbb45b031e7bfd73c6d3983bcce07b9c1
rm -rf "${LOCATION_ADDAXAI_FILES}/yolov5_versions/yolov5_old/yolov5/.git"
echo "yolov5 old version cloned"

git clone https://github.com/ultralytics/yolov5.git "${LOCATION_ADDAXAI_FILES}/yolov5_versions/yolov5_new/yolov5"
git -C "${LOCATION_ADDAXAI_FILES}/yolov5_versions/yolov5_new/yolov5" checkout 3e55763d45f9c5f8217e4dad5ba1e6c1f42e3bf8
rm -rf "${LOCATION_ADDAXAI_FILES}/yolov5_versions/yolov5_new/yolov5/.git"
echo "yolov5 new version cloned"

git clone --branch pyside6 --depth 1 https://github.com/PetervanLunteren/Human-in-the-loop.git "${LOCATION_ADDAXAI_FILES}/Human-in-the-loop"
rm -rf "${LOCATION_ADDAXAI_FILES}/Human-in-the-loop/.git"
echo "Human-in-the-loop cloned"

git clone --depth 1 https://github.com/PetervanLunteren/visualise_detection.git "${LOCATION_ADDAXAI_FILES}/visualise_detection"
rm -rf "${LOCATION_ADDAXAI_FILES}/visualise_detection/.git"
echo "visualise_detection cloned"

# check if curl is installed
if command -v curl &> /dev/null
then
    echo "curl is already installed."
else
    echo "curl is not installed. Trying to installing curl..."
  { 
    sudo apt update
    sudo apt install -y curl
  } || { 
    echo "Curl could not be installed. Copy-paste all text in this console window and send it to peter@addaxdatascience.com for further support."; exit 1;
    }
  echo "curl successfully installed."
fi

### download megadetector 
curl -L https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt -o "${LOCATION_ADDAXAI_FILES}/models/det/MegaDetector 5a/md_v5a.0.0.pt"

# install miniforge
curl --keepalive -L -o "${LOCATION_ADDAXAI_FILES}/Miniforge3.sh" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash "${LOCATION_ADDAXAI_FILES}/Miniforge3.sh" -b -p "${LOCATION_ADDAXAI_FILES}/miniforge"
rm "${LOCATION_ADDAXAI_FILES}/Miniforge3.sh"

### source conda 
source "${LOCATION_ADDAXAI_FILES}/miniforge/etc/profile.d/conda.sh"
source "${LOCATION_ADDAXAI_FILES}/miniforge/bin/activate"
conda_exe="${LOCATION_ADDAXAI_FILES}/miniforge/bin/conda"

### install mamba
$conda_exe install mamba -n base -c conda-forge -y
conda_exe="${LOCATION_ADDAXAI_FILES}/miniforge/bin/mamba"

### install env-base
$conda_exe env create --file="${LOCATION_ADDAXAI_FILES}/cameratraps/envs/environment-detector.yml" -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" -y
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install RangeSlider
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install gpsphoto
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install exifread
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install piexif
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install openpyxl
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install customtkinter
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install CTkTable
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install folium
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install plotly
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install "gitpython>=3.1.30"
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install "tensorboard>=2.4.1"
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install "thop>=0.1.1"
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install "protobuf<=3.20.1"
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install "setuptools>=65.5.1"
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install PySide6
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-base" pip install "lxml==4.6.3"
"${LOCATION_ADDAXAI_FILES}/envs/env-base/bin/pyside6-rcc" -o "${LOCATION_ADDAXAI_FILES}/Human-in-the-loop/libs/resources.py" "${LOCATION_ADDAXAI_FILES}/Human-in-the-loop/resources.qrc"

### clean
$conda_exe clean --all --yes --force-pkgs-dirs
$conda_exe clean --all --yes

# requirements for Human-in-the-loop
echo "We need to install libxcb-cursor-dev (https://packages.debian.org/sid/libxcb-cursor-dev) and libxcb-cursor0 (https://packages.debian.org/sid/libxcb-cursor0). If you don't have root privileges you might be prompted for a password. Press CONTROL+D to skip authentication and not install these packages. AddaxAI will still work fine without it but you might have problems with the Human-in-the-loop software."
{ # first try without sudo
  add-apt-repository universe
  apt-get update
  apt-get install libxcb-cursor-dev
  apt-get install libxcb-cursor0
} || { # otherwise with sudo
  sudo add-apt-repository universe
  sudo apt-get update
  sudo apt-get install libxcb-cursor-dev
  sudo apt-get install libxcb-cursor0
  }

### install env-tensorflow
$conda_exe env create --file="${LOCATION_ADDAXAI_FILES}/AddaxAI/classification_utils/envs/tensorflow-linux-windows.yml" -p "${LOCATION_ADDAXAI_FILES}/envs/env-tensorflow" -y

### clean
$conda_exe clean --all --yes --force-pkgs-dirs
$conda_exe clean --all --yes

### install env-pytorch
$conda_exe create -p "${LOCATION_ADDAXAI_FILES}/envs/env-pytorch" python=3.8 -y
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-pytorch" pip install torch==2.0.1 torchvision==0.15.2
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-pytorch" pip install "ultralytics==8.0.191"
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-pytorch" pip install "numpy==1.24.1"
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-pytorch" pip install "humanfriendly==10.0"
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-pytorch" pip install "jsonpickle==3.0.2"
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-pytorch" pip install timm
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-pytorch" pip install dill
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-pytorch" pip install omegaconf
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-pytorch" pip install "pytorch-lightning==2.0.6"
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-pytorch" pip install "albumentations==1.3.1"

### clean
$conda_exe clean --all --yes --force-pkgs-dirs
$conda_exe clean --all --yes

### install env-pywildlife
$conda_exe create -p "${LOCATION_ADDAXAI_FILES}/envs/env-pywildlife" python=3.8 -y
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-pywildlife" pip install pytorchwildlife
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-pywildlife" pip install "setuptools<70"
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-pywildlife" pip install jsonpickle

### clean
$conda_exe clean --all --yes --force-pkgs-dirs
$conda_exe clean --all --yes

### install env-speciesnet
$conda_exe create -p "${LOCATION_ADDAXAI_FILES}/envs/env-speciesnet" python=3.11 -y
$conda_exe run -p "${LOCATION_ADDAXAI_FILES}/envs/env-speciesnet" pip install speciesnet

### clean
$conda_exe clean --all --yes --force-pkgs-dirs
$conda_exe clean --all --yes

# timestamp the end of installation
END_DATE=`date`
echo "This installation ended at: $END_DATE"

# message for the user
echo ""
echo "THE INSTALLATION IS DONE! You can close this window now and proceed to open AddaxAI by double clicking the shortcut on your desktop."
echo ""
