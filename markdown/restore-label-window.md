
# Restore the species label window

If the species label panel on the right side of the verification window has disappeared, you can restore it by deleting a settings file. This is a labelImg quirk. The label tool that AddaxAI uses for manual verification stores its window layout in a hidden file. Deleting it resets the layout to its default state.

## The fix

1. Make sure you can see hidden files in File Explorer.
   - **Windows 11**: Open File Explorer > click **View** in the toolbar > **Show** > check **Hidden items**.
   - **Windows 10**: Open File Explorer > click the **View** tab > check the **Hidden items** checkbox.

2. Navigate to your user folder: `C:\Users\<username>\`.

3. Find and delete the file `.labelImgSettings.pkl`.

4. Close AddaxAI completely and reopen it.

5. Start the verification again. The species label window should be restored.
