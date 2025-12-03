# Quick test
This document explains how to run a small test on your own data to get an initial sense of how a model behaves. A fast way to explore this is to run the model on a few camera deployments, one at the time.

## Run the model
1. Make sure you are running the latest version of AddaxAI to avoid bugs that may have been fixed already: https://addaxdatascience.com/addaxai/#install
2. Open AddaxAI and switch to `Advanced mode` using the button in the top left corner.

<img width="416" height="103" alt="Screenshot 2025-12-03 at 09 34 38" src="https://github.com/user-attachments/assets/bb1d034c-436e-463d-94f1-d589a538e620" />

<br/>
<br/>

3. In `Step 1: Select folder`, select a fresh camera deployment. Make sure this deployment was not part of the model training data, otherwise you will get unrealistically good results. Choose a folder with a few hundred to a few thousand images. It is easiest to do this with images rather than videos.
4. For `Model to detect animals, vehicles, and persons`, select `MegaDetector 5a`.
5. For `Model to identify animals`, select the model you want to test.
6. Select the species present in your project area at `Select species`. This may take a moment if you have many classes, but it will be saved for future sessions. Selecting only the relevant species prevents predictions for species that do not occur in your area.
7. Leave the detection and classification thresholds at their defaults. Typical values are 0.30 for detection and 0.50 for classification.
8. Enable `Smooth confidence scores per sequence`.
9. Enable `Taxonomic confidence aggregation`.
10. For `Prediction level`, choose `Let the model decide`.
11. Enable `Process images, if present`.
12. Click `Start processing`.

<img width="1306" height="952" alt="Screenshot 2025-12-03 at 10 12 56" src="https://github.com/user-attachments/assets/6a414fc0-2fe4-4220-b463-4faa0ed1151d" />

You will see two progress bars. One belongs to the detection model (`Locating animals...`) and one to the species identification model (`Identifying animals...`). When both finish, `Step 2: Analysis` will show a green checkmark, indicating that processing completed successfully.

<img width="440" height="424" alt="Screenshot 2025-12-03 at 09 45 50" src="https://github.com/user-attachments/assets/39bcb654-9d8c-418f-b657-ff2102d4aa98" />

## Visualise the results
To inspect predictions in a simple way, skip `Step 3: Annotation (optional)` and go straight to `Step 4: Post-processing (optional)`.

1. Choose a destination folder for the output. This can be temporary, for example `/Users/Peter/Desktop/temp-01`.
2. Enable `Visualise detections and blur people`.
3. Enable `Draw bounding boxes and confidences`.
4. Set `Line width and font size` to `Medium`.
5. You may choose to blur people or not. It does not affect the test.
6. Disable all other post-processing options.
7. Set the `Confidence threshold` to `0.30`.
8. Click `Start processing`.

<img width="1306" height="952" alt="Screenshot 2025-12-03 at 10 22 58" src="https://github.com/user-attachments/assets/64fa8589-4ce7-4bfd-a006-20355b9f9a8c" />

After the progress bar completes, the destination folder will contain copies of the images with bounding boxes and confidence scores drawn on them. This makes it easy to review how the model is performing. Browse through the images and take note of both good and bad predictions.

<img width="1434" height="820" alt="Screenshot 2025-12-03 at 10 16 56" src="https://github.com/user-attachments/assets/b1142feb-a397-4051-ad6e-dfdbd1258653" />

## Adjust settings

Once you have some initial results, you can repeat the workflow above and experiment with different settings.

#### Detection confidence threshold

The detection model finds animals in the image. This threshold controls which detections are passed on for species identification. Lower values allow more detections through, including uncertain ones. Higher values filter out weaker detections.

#### Classification confidence threshold

The classification model identifies the species. This threshold determines which predictions will be accepted. Lower values allow more uncertain predictions, while higher values restrict results to high-confidence classifications.

#### Smooth confidence scores per sequence

Sequence smoothing averages confidence scores across a sequence of images. This reduces noise and provides more stable predictions when multiple images show the same event. It assumes one species per sequence, so do not use it if multispecies sequences are common. It does not affect the detection of vehicles or people that appear alongside animals.

#### Taxonomic confidence aggregation

When enabled, the model will fall back to a higher taxonomic level (such as genus or family) if it is uncertain at the species level. This can improve overall accuracy by avoiding low-confidence species predictions. Some classes may only have higher-level labels available. You can also experiment with fixing the prediction level to a specific taxonomic level, like family. 


## Test on new locations

After you have reviewed results for your first deployment and you feel like experimenting more, you can try the model on a few different deployments, ideally from different locations. This helps reveal how well the model handles different backgrounds, lighting conditions and environments.



