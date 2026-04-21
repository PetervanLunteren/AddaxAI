
# Manually download model weights

Below are instructions for users who are unable to download model weight files through AddaxAI due to network restrictions (e.g., university or government networks). This should be plan B, as the automatic download is much quicker. If you haven't tried the automatic download yet, try that first and check your firewall, proxy, and VPN settings. If you still have trouble, feel free to reach out at [peter@addaxdatascience.com](mailto:peter@addaxdatascience.com).

Some models consist of a single file, others of several (model weights, class list, taxonomic mapping, sometimes a pretrained backbone). You need to download **all listed files** for a given model and place them together in the same folder.

## Step-by-step instructions

1. Find the model you need in the list below. The model name shown matches the name displayed in AddaxAI's error window.

2. Click each download link and save the file with the **exact filename** shown. Filenames must remain unchanged.

3. Navigate to your `AddaxAI_files` folder. The location depends on your operating system, see [this page](https://github.com/PetervanLunteren/AddaxAI/blob/main/markdown/AddaxAI_files_location.md) for details.

4. Open the target folder shown for your model inside `AddaxAI_files`. If the folder does not already exist, create it. The folder name must match exactly, including spaces, punctuation, and special characters (e.g., `Hawaiʻi, USA - AI Puaʻa v1.0`).

5. Move all downloaded files into that model folder. For example:
   ```
   AddaxAI_files/models/cls/Namibian Desert - Addax Data Science/namib_desert_v1.pt
   ```

6. Close AddaxAI completely and reopen it. The model should now be available.

_______________________________________________________________________
<details>
<summary><b>Tip: download on another computer</b></summary>

<br>

If your network blocks downloads entirely, you can download the files on another computer (e.g., at home, on a mobile hotspot, or on a less restricted network) and transfer them to your work computer using a USB drive or other file transfer method.

</details>

<details>
<summary><b>Alternative: download using the command line</b></summary>

<br>

If your internet connection is unstable and the browser download keeps failing, you can use the command line instead. This supports resuming interrupted downloads. Open a terminal and run the following command for each file you need, replacing `<filename>` and `<url>` with the values from the model section below:

```
curl --retry 5 --retry-delay 10 --continue-at - -L -o "<filename>" "<url>"
```

If the download is interrupted, run the same command again to resume. Files download into your current working directory, so move them to the correct model folder afterwards.

</details>

## Available models

### MegaDetector 1000 Redwood

- **Target folder:** `models/det/MegaDetector 1000 Redwood/`
- **Size:** 268 MB
- **Files:**
  - [`md_v1000.0.0-redwood.pt`](https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-redwood.pt)

### MegaDetector 1000 Spruce

- **Target folder:** `models/det/MegaDetector 1000 Spruce/`
- **Size:** 14 MB
- **Files:**
  - [`md_v1000.0.0-spruce.pt`](https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-spruce.pt)

### MegaDetector 5a

- **Target folder:** `models/det/MegaDetector 5a/`
- **Size:** 281 MB
- **Files:**
  - [`md_v5a.0.0.pt`](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt)

Note: MegaDetector 5a is pre-installed with AddaxAI. If it is missing, reinstalling AddaxAI may be easier than a manual download.

### MegaDetector 5b

- **Target folder:** `models/det/MegaDetector 5b/`
- **Size:** 281 MB
- **Files:**
  - [`md_v5b.0.0.pt`](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5b.0.0.pt)

### Hawaiʻi, USA - AI Puaʻa v1.0

- **Target folder:** `models/cls/Hawaiʻi, USA - AI Puaʻa v1.0/`
- **Files:**
  - [`always_crop_99710272_22x8_v12_epoch_00148.pt`](https://huggingface.co/Addax-Data-Science/HWI-ADS-v1/resolve/main/always_crop_99710272_22x8_v12_epoch_00148.pt?download=true)
  - [`final-20260317.pt`](https://huggingface.co/Addax-Data-Science/HWI-ADS-v1/resolve/main/final-20260317.pt?download=true)
  - [`taxon-mapping.csv`](https://huggingface.co/Addax-Data-Science/HWI-ADS-v1/resolve/main/taxon-mapping.csv?download=true)

### Neotropical region - TropiCam-AI v1.0

- **Target folder:** `models/cls/Neotropical region - TropiCam-AI v1.0/`
- **Size:** 360 MB
- **Files:**
  - [`TropiCam_AI_TensorFlow_model.h5`](https://huggingface.co/Addax-Data-Science/NEO-MNCN-v1-0/resolve/main/TropiCam_AI_TensorFlow_model.h5?download=true)
  - [`taxon-mapping.csv`](https://huggingface.co/Addax-Data-Science/NEO-MNCN-v1-0/resolve/main/taxon-mapping.csv?download=true)

### Victoria, Australia - Parks Victoria - Addax Data Science

- **Target folder:** `models/cls/Victoria, Australia - Parks Victoria - Addax Data Science/`
- **Size:** 500 MB
- **Files:**
  - [`always_crop_99710272_22x8_v12_epoch_00148.pt`](https://huggingface.co/Addax-Data-Science/VIC-ADS-v1/resolve/main/always_crop_99710272_22x8_v12_epoch_00148.pt?download=true)
  - [`final-20251221.pt`](https://huggingface.co/Addax-Data-Science/VIC-ADS-v1/resolve/main/final-20251221.pt?download=true)
  - [`taxon-mapping.csv`](https://huggingface.co/Addax-Data-Science/VIC-ADS-v1/resolve/main/taxon-mapping.csv?download=true)

### Australian Wildlife Classifier - AWC135

- **Target folder:** `models/cls/Australian Wildlife Classifier - AWC135/`
- **Size:** 115 MB
- **Files:**
  - [`awc-135-v1.pth`](https://huggingface.co/Addax-Data-Science/AWC135-AWC-v1/resolve/main/awc-135-v1.pth?download=true)
  - [`taxon-mapping.csv`](https://huggingface.co/Addax-Data-Science/AWC135-AWC-v1/resolve/main/taxon-mapping.csv?download=true)

### Southwestern Borderlands USA

- **Target folder:** `models/cls/Southwestern Borderlands USA/`
- **Size:** 450 MB
- **Files:**
  - [`final-20260109.pt`](https://huggingface.co/Addax-Data-Science/SBUSA-ADS-v1/resolve/main/final-20260109.pt?download=true)
  - [`always_crop_99710272_22x8_v12_epoch_00148.pt`](https://huggingface.co/Addax-Data-Science/SBUSA-ADS-v1/resolve/main/always_crop_99710272_22x8_v12_epoch_00148.pt?download=true)

### AHDriFT-ID (Midwest US) v1.0

- **Target folder:** `models/cls/AHDriFT-ID (Midwest US) v1.0/`
- **Files:**
  - [`final-20251223.pt`](https://huggingface.co/Addax-Data-Science/AHDRIFT-v1/resolve/main/final-20251223.pt?download=true)
  - [`full_image_88545560_22x8_v12_epoch_00153.pt`](https://huggingface.co/Addax-Data-Science/AHDRIFT-v1/resolve/main/full_image_88545560_22x8_v12_epoch_00153.pt?download=true)
  - [`taxon-mapping.csv`](https://huggingface.co/Addax-Data-Science/AHDRIFT-v1/resolve/main/taxon-mapping.csv?download=true)

### Queensland Wet Tropics - WildObs

- **Target folder:** `models/cls/Queensland Wet Tropics - WildObs/`
- **Files:**
  - [`wildobs_QLD_WetTropics.pt`](https://huggingface.co/Addax-Data-Science/WetTropics_WildObs/resolve/main/wildobs_QLD_WetTropics.pt?download=true)
  - [`taxon-mapping.csv`](https://huggingface.co/Addax-Data-Science/WetTropics_WildObs/resolve/main/taxon-mapping.csv?download=true)

### African tropical forests - DeepForestVision

- **Target folder:** `models/cls/African tropical forests - DeepForestVision/`
- **Size:** 1.2 GB
- **Files:**
  - [`DFV.pt`](https://huggingface.co/Addax-Data-Science/AFR-DFV-v1/resolve/main/DFV.pt?download=true)
  - [`taxon-mapping.csv`](https://huggingface.co/Addax-Data-Science/AFR-DFV-v1/resolve/main/taxon-mapping.csv?download=true)

### Europe - DeepFaune v1.4

- **Target folder:** `models/cls/Europe - DeepFaune v1.4/`
- **Size:** 1.1 GB
- **Files:**
  - [`deepfaune-vit_large_patch14_dinov2.lvd142m.v4.pt`](https://huggingface.co/Addax-Data-Science/Deepfaune_v1.4/resolve/main/deepfaune-vit_large_patch14_dinov2.lvd142m.v4.pt?download=true)

### New Zealand Species v3.03 - DOC NZ - wekaResearch

- **Target folder:** `models/cls/New Zealand Species v3.03 - DOC NZ - wekaResearch/`
- **Size:** 85 MB
- **Files:**
  - [`Exp_60_run_01_best_weights.pt`](https://huggingface.co/Addax-Data-Science/NZS-WEK-v3-03/resolve/main/Exp_60_run_01_best_weights.pt?download=true)
  - [`taxon-mapping.csv`](https://huggingface.co/Addax-Data-Science/NZS-WEK-v3-03/resolve/main/taxon-mapping.csv?download=true)

### Gifu region Japan - Gifu University

- **Target folder:** `models/cls/Gifu region Japan - Gifu University/`
- **Size:** 386 MB
- **Files:**
  - [`gifu-wildlife_cls_resnet50_v0.2.1.pth`](https://huggingface.co/Addax-Data-Science/Japan_Gifu_v0.2/resolve/main/gifu-wildlife_cls_resnet50_v0.2.1.pth?download=true)
  - [`resnet50-11ad3fa6.pth`](https://huggingface.co/Addax-Data-Science/Japan_Gifu_v0.2/resolve/main/resnet50-11ad3fa6.pth?download=true)
  - [`classes.csv`](https://huggingface.co/Addax-Data-Science/Japan_Gifu_v0.2/resolve/main/classes.csv?download=true)
  - [`taxon-mapping.csv`](https://huggingface.co/Addax-Data-Science/Japan_Gifu_v0.2/resolve/main/taxon-mapping.csv?download=true)

### Sub-Saharan Drylands - Addax Data Science

- **Target folder:** `models/cls/Sub-Saharan Drylands - Addax Data Science/`
- **Size:** 215 MB
- **Files:**
  - [`sub_saharan_drylands_v1.pt`](https://huggingface.co/Addax-Data-Science/sub_saharan_drylands_v1.pt/resolve/main/sub_saharan_drylands_v1.pt?download=true)
  - [`taxon-mapping.csv`](https://huggingface.co/Addax-Data-Science/sub_saharan_drylands_v1.pt/resolve/main/taxon-mapping.csv?download=true)

### Terai region Nepal - Alexander Merdian-Tarko

- **Target folder:** `models/cls/Terai region Nepal - Alexander Merdian-Tarko/`
- **Size:** 215 MB
- **Files:**
  - [`model.keras`](https://huggingface.co/alexvmt/TeraiNet/resolve/main/model.keras?download=true)
  - [`class_list.yaml`](https://huggingface.co/alexvmt/TeraiNet/resolve/main/class_list.yaml?download=true)

### Tasmanian vertebrates

- **Target folder:** `models/cls/Tasmanian vertebrates/`
- **Size:** 86 MB
- **Files:**
  - [`tas_ens_mewc.keras`](https://huggingface.co/Addax-Data-Science/Tasmanian_vertebrates/resolve/main/tas_ens_mewc.keras?download=true)
  - [`class_list.yaml`](https://huggingface.co/Addax-Data-Science/Tasmanian_vertebrates/resolve/main/tas_class_map.yaml?download=true) — the URL filename is `tas_class_map.yaml`, but you must save it locally as `class_list.yaml`.

### Namibian Desert - Addax Data Science

- **Target folder:** `models/cls/Namibian Desert - Addax Data Science/`
- **Size:** 107 MB
- **Files:**
  - [`namib_desert_v1.pt`](https://huggingface.co/Addax-Data-Science/Namib-Desert-v1/resolve/main/namib_desert_v1.pt?download=true)

### New Zealand Invasives - DOC NZ - Addax Data science

- **Target folder:** `models/cls/New Zealand Invasives - DOC NZ - Addax Data science/`
- **Size:** 32 MB
- **Files:**
  - [`new_zealand_v1.pt`](https://huggingface.co/Addax-Data-Science/New_Zealand_v1/resolve/main/new_zealand_v1.pt?download=true)

### Colombian Amazon - AI for Good Lab, Microsoft

- **Target folder:** `models/cls/Colombian Amazon - AI for Good Lab, Microsoft/`
- **Size:** 197 MB
- **Files:**
  - [`AI4GAmazonClassification_v0.0.0.ckpt`](https://zenodo.org/records/10041983/files/AI4GAmazonClassification_v0.0.0.ckpt?download=1)

### Peruvian Amazon - San Diego Zoo Wildlife Alliance

- **Target folder:** `models/cls/Peruvian Amazon - San Diego Zoo Wildlife Alliance/`
- **Size:** 247 MB
- **Files:**
  - [`Peru-Amazon_0.86.h5`](https://huggingface.co/Addax-Data-Science/Peruvian_Amazon/resolve/main/Peru-Amazon_0.86.h5?download=true)
  - [`Peru-Amazon_0.86.txt`](https://huggingface.co/Addax-Data-Science/Peruvian_Amazon/resolve/main/Peru-Amazon_0.86.txt?download=true)

### Perivuan Andes - San Diego Zoo Wildlife Alliance

- **Target folder:** `models/cls/Perivuan Andes - San Diego Zoo Wildlife Alliance/` (folder name uses the existing spelling `Perivuan`)
- **Size:** 431 MB
- **Files:**
  - [`andes_v1.pt`](https://huggingface.co/Addax-Data-Science/Peruvian_Andes/resolve/main/andes_v1.pt?download=true)
  - [`efficientnet_v2_m-dc08266a.pth`](https://huggingface.co/Addax-Data-Science/Peruvian_Andes/resolve/main/efficientnet_v2_m-dc08266a.pth?download=true)
  - [`classes.csv`](https://huggingface.co/Addax-Data-Science/Peruvian_Andes/resolve/main/classes.csv?download=true)

### Iran - Addax Data Science

- **Target folder:** `models/cls/Iran - Addax Data Science/`
- **Size:** 32 MB
- **Files:**
  - [`iran_v1.pt`](https://huggingface.co/Addax-Data-Science/Iran_v1/resolve/main/iran_v1.pt?download=true)

### Kirghizistan - Manas v1 - OSI-Panthera - Hex Data

- **Target folder:** `models/cls/Kirghizistan - Manas v1 - OSI-Panthera - Hex Data/`
- **Size:** 472 MB
- **Files:**
  - [`best_model_Fri_Sep__1_18_50_55_2023.pt`](https://huggingface.co/Hex-Data/Panthera/resolve/main/best_model_Fri_Sep__1_18_50_55_2023.pt?download=true)
  - [`classes_Fri_Sep__1_18_50_55_2023.pickle`](https://huggingface.co/Hex-Data/Panthera/resolve/main/classes_Fri_Sep__1_18_50_55_2023.pickle?download=true)

### Southwest USA v3 - San Diego Zoo Wildlife Alliance

- **Target folder:** `models/cls/Southwest USA v3 - San Diego Zoo Wildlife Alliance/`
- **Size:** 431 MB
- **Files:**
  - [`southwest_v3.pt`](https://huggingface.co/Addax-Data-Science/Southwest_USA_v3/resolve/main/southwest_v3.pt?download=true)
  - [`efficientnet_v2_m-dc08266a.pth`](https://huggingface.co/Addax-Data-Science/Southwest_USA_v3/resolve/main/efficientnet_v2_m-dc08266a.pth?download=true)
  - [`classes.csv`](https://huggingface.co/Addax-Data-Science/Southwest_USA_v3/resolve/main/classes.csv?download=true)
