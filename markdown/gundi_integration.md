## Send AddaxAI detections to EarthRanger via Gundi
Below is a step-by-step tutorial on how to upload wildlife detections from AddaxAI to [Gundi](https://gundiservice.org), which routes them onwards to [EarthRanger](https://www.earthranger.com/) (or any other destination Gundi supports, such as SMART or Movebank). Each detection is sent as an event with the species, confidence, camera metadata, and the original photo attached, so it appears in your EarthRanger map like any other field report.

Learn more about the AddaxAI software: https://addaxdatascience.com/addaxai/

Learn more about Gundi: https://gundiservice.org

If anything is unclear, let me know: peter@addaxdatascience.com

### I - What you need
1. **A Gundi account with a Connection configured.** A Connection tells Gundi where your data should go (e.g. your EarthRanger site). Without one, uploads have no destination. See section II.
2. **A destination that accepts the events.** For EarthRanger, your site must have the `wildlife_observation` event type available. Ask your EarthRanger administrator if you are not sure.
3. **Images with GPS coordinates** in their EXIF metadata. Gundi events require a location, so images without GPS coordinates are skipped (AddaxAI will tell you how many).

### II - Set up the Gundi Connection and get your API key
1. Log in to the [Gundi Portal](https://sensors.api.gundiservice.org/).
2. Create a Connection with your data provider as the source and your EarthRanger site (or other platform) as the destination. The EarthRanger team can help you set this up — see their [Gundi documentation](https://support.earthranger.com/developer_docs/gundi-api) or contact support.
3. Copy the **API key** from your Connection. This is the only credential AddaxAI needs.

### III - Enable the upload in AddaxAI
1. Open AddaxAI in **advanced mode** and run your images through a detection model as usual (steps 1 and 2).
2. In **Step 4: Post-processing**, tick **"Upload events to Gundi"**. The Gundi options panel will unfold.
3. Paste your API key into the **API key** field. It is remembered for next time — AddaxAI stores it in a private file (`gundi-api-key.txt` next to the `AddaxAI_files` folder), never in the shared settings file.
4. Set the **confidence threshold** as usual — only detections at or above the threshold are uploaded.
5. Click **Start post-processing**.

### IV - What happens during the upload
1. AddaxAI first scans your images and tells you how many detections will be uploaded, and how many images will be skipped (no GPS coordinates, or missing from disk).
2. The upload runs with a progress bar, elapsed/remaining time, and a working cancel button. The upload happens *before* any file separation, so it also works when you move files into subdirectories.
3. When everything succeeds you get a confirmation with the number of uploaded events. If any events fail, the details are written to `gundi_upload_errors.txt` in your destination folder and you get a warning.

### V - Verify the events arrived
Check your **destination platform** (e.g. the EarthRanger events feed) — each event carries the species, confidence, and the photo as an attachment.

Note: photo attachments do **not** show up as separate entries in Gundi's own Activity Log — that is normal. The attachment travels with the event to the destination, so always verify in EarthRanger itself.

### VI - What gets sent
Each detection above the threshold becomes one `wildlife_observation` event with:

| Field | Content |
|---|---|
| `title` | e.g. `Panthera leo (lion) detected (93% confidence) (verified)` |
| `recorded_at` | capture time from the image EXIF (falls back to filename patterns or file date) |
| `location` | GPS latitude/longitude from the image EXIF |
| `event_details` | species, confidence, camera make/model, detection & classification model, human-verified flag, bounding box, altitude, AddaxAI version, image filename |
| attachment | the original photo |

The human-verified flag reflects AddaxAI's human-in-the-loop verification status, so reviewed detections are distinguishable in EarthRanger.

### VII - Troubleshooting
- **"Gundi API key is required"** — enter your API key in the Gundi options panel (section III).
- **"No detections could be uploaded"** — your images lack GPS coordinates, or the files were not found on disk. The message shows both counts.
- **Events don't appear in EarthRanger** — check that your Gundi Connection's destination is your EarthRanger site, and that the site has the `wildlife_observation` event type. Also note that re-uploading an event with identical data is discarded as a duplicate by EarthRanger.
- **Some events failed** — see `gundi_upload_errors.txt` in your destination folder for the per-image server responses.
- **For developers:** set the environment variable `ADDAXAI_GUNDI_ENV=stage` before launching to test against Gundi's staging API instead of production. Unknown values refuse to start rather than silently using production.
