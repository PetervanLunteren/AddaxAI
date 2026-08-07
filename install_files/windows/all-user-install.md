# AddaxAI install for all users

The standard installation of AddaxAI is for the current user only. This avoids the need for admin rights, which can be restrictive for users on locked-down machines, such as those in universities and government agencies. However, it is possible to install AddaxAI for multiple users by following the steps below.

The software works if the files (i.e., the "AddaxAI_files" folder) are located in either one of these locations:
* `C:\Users\<username>\AddaxAI_files\`
* `C:\Program Files\AddaxAI_files`

That means we _just_ need to move the files to the other location (and some other things):

1. Install AddaxAI version 6 using the normal install: https://docs.addaxai.com/docs/help/go-back-to-v6. That will install everything at `C:\Users\<username>\AddaxAI_files\`. If the installer is blocked on your machine, use the [manual install](manual-install.md) instead. 
2. Now manually move the entire folder from `C:\Users\<username>\AddaxAI_files\` to `C:\Program Files\AddaxAI_files`. You might be prompted to fill in your admin password. This can take some time since it contains many files.

<img width="449" alt="Screenshot 2025-02-21 at 08 18 04" src="https://github.com/user-attachments/assets/49147cde-2116-4807-a01f-cd43c87a68d0" />

<br>
<br>

3. The old shortcut created by the installation can't be used anymore, since it still points towards the old location. So we need to create a new one. Navigate to `C:\Program Files\AddaxAI_files\AddaxAI\open.bat`, right-click (you might need to click 'More options'), and click 'Create shortcut'.

<img width="1289" alt="Screenshot 2025-02-21 at 08 37 10" src="https://github.com/user-attachments/assets/f3d1ea48-88bb-4d20-a7dc-fd2dc9eedd80" />

<br>
<br>


4. You might be prompted that it is not possible to create a shortcut at that location. It will propose to create one on your Desktop. Click 'yes'. You will probably have to do this for every user.

<img width="401" alt="Screenshot 2025-02-21 at 08 37 25" src="https://github.com/user-attachments/assets/1e854f34-8311-41ee-b6d0-219cc9877ce2" />

<br>
<br>


5. Now we have all the files at the right location and a shortcut to open the application. But we still can't open in without the user needing admin rights. We can avoid this by granting the users read and write access to this folder. Right-click the folder and choose properties.

<img width="1282" alt="Screenshot 2025-02-21 at 08 38 05" src="https://github.com/user-attachments/assets/d7f06570-25af-4e07-bd47-477e1f418ab5" />

<br>
<br>


6. Choose the security tab and click Edit.

<img width="363" alt="Screenshot 2025-02-21 at 08 38 51" src="https://github.com/user-attachments/assets/fe7d36eb-f8f9-466a-ad63-f9cf6d5f669f" />

<br>
<br>


7. Choose 'Users' (_Gebruikers_ in Dutch) and check the allow boxes.

<img width="362" alt="Screenshot 2025-02-21 at 09 04 29" src="https://github.com/user-attachments/assets/183c4b55-0e7a-4eaa-9932-d46cca76419d" />

<br>
<br>


8. Now it will change the permissions of the files recursively. That can take a minute.

<img width="401" alt="Screenshot 2025-02-21 at 08 41 50" src="https://github.com/user-attachments/assets/347baa53-afc1-4aa2-b104-7cf231da84c5" />

<br>
<br>


9. When that is done, you should be able to open AddaxAI using the shortcut created in step 4.

<img width="138" alt="Screenshot 2025-02-21 at 09 06 30" src="https://github.com/user-attachments/assets/571f16c8-0340-4262-a26d-4d34a2b5e58b" />

<br>
<br>



Good luck!



