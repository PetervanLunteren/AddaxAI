const { notarize } = require('@electron/notarize');
const fs = require('fs');
const path = require('path');

async function attemptNotarization(appPath, credentials, timeoutMinutes, attemptNumber) {
  const startTime = Date.now();
  const timeoutMs = timeoutMinutes * 60 * 1000;

  console.log(`\n📤 Attempt ${attemptNumber}: Submitting to Apple (timeout: ${timeoutMinutes} min)...`);

  // Progress logging
  const progressInterval = setInterval(() => {
    const elapsed = Math.round((Date.now() - startTime) / 1000);
    console.log(`⏱️  Still notarizing... (${elapsed}s elapsed)`);
  }, 60000); // Log every 60 seconds

  try {
    const notarizePromise = notarize({
      tool: 'notarytool',
      appBundleId: 'com.addaxai.cameratrap',
      appPath: appPath,
      appleId: credentials.appleId,
      appleIdPassword: credentials.password,
      teamId: credentials.teamId,
    });

    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`Timeout after ${timeoutMinutes} minutes`)), timeoutMs)
    );

    await Promise.race([notarizePromise, timeoutPromise]);

    clearInterval(progressInterval);
    const duration = Math.round((Date.now() - startTime) / 1000);
    console.log(`\n✅ Notarization complete (took ${duration}s)`);
    return true;
  } catch (error) {
    clearInterval(progressInterval);
    throw error;
  }
}

function checkNotarizationTicket(appPath) {
  // Check if the app has a notarization ticket stapled
  const { execSync } = require('child_process');
  try {
    execSync(`xcrun stapler validate "${appPath}"`, { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

exports.default = async function notarizing(context) {
  const { electronPlatformName, appOutDir } = context;

  console.log('=== Notarization Debug Info ===');
  console.log('Platform:', electronPlatformName);
  console.log('App output directory:', appOutDir);

  // Only notarize on macOS
  if (electronPlatformName !== 'darwin') {
    console.log('⏭️  Skipping notarization: not macOS');
    return;
  }

  // Check if we have the required credentials
  console.log('Checking environment variables...');
  console.log('APPLE_ID:', process.env.APPLE_ID ? '✓ Set' : '✗ Missing');
  console.log('APPLE_APP_SPECIFIC_PASSWORD:', process.env.APPLE_APP_SPECIFIC_PASSWORD ? '✓ Set' : '✗ Missing');
  console.log('APPLE_TEAM_ID:', process.env.APPLE_TEAM_ID ? '✓ Set' : '✗ Missing');

  if (!process.env.APPLE_ID || !process.env.APPLE_APP_SPECIFIC_PASSWORD || !process.env.APPLE_TEAM_ID) {
    console.log('⏭️  Skipping notarization: missing credentials');
    console.log('Set APPLE_ID, APPLE_APP_SPECIFIC_PASSWORD, and APPLE_TEAM_ID environment variables to enable notarization');
    return;
  }

  const appName = context.packager.appInfo.productFilename;
  const appPath = `${appOutDir}/${appName}.app`;

  console.log('\n🔐 Starting notarization with smart retry...');
  console.log('App name:', appName);
  console.log('App path:', appPath);
  console.log('Bundle ID: com.addaxai.cameratrap');
  console.log('Team ID:', process.env.APPLE_TEAM_ID);

  const credentials = {
    appleId: process.env.APPLE_ID,
    password: process.env.APPLE_APP_SPECIFIC_PASSWORD,
    teamId: process.env.APPLE_TEAM_ID,
  };

  try {
    // First attempt: 5 minutes
    try {
      await attemptNotarization(appPath, credentials, 5, 1);
      return; // Success!
    } catch (firstError) {
      console.log(`\n⚠️  First attempt timed out after 5 minutes`);
      console.log('Checking if notarization ticket exists...');

      // Check if the app was actually notarized despite the timeout
      if (checkNotarizationTicket(appPath)) {
        console.log('✅ Notarization ticket found! The app was notarized successfully.');
        return;
      }

      console.log('❌ No ticket found. Retrying with 10-minute timeout...');
    }

    // Second attempt: 10 minutes
    try {
      await attemptNotarization(appPath, credentials, 10, 2);
      return; // Success!
    } catch (secondError) {
      console.log(`\n⚠️  Second attempt timed out after 10 minutes`);
      console.log('Checking if notarization ticket exists...');

      // Check if the app was actually notarized despite the timeout
      if (checkNotarizationTicket(appPath)) {
        console.log('✅ Notarization ticket found! The app was notarized successfully.');
        return;
      }

      console.log('❌ No ticket found. Final attempt with 30-minute timeout...');
    }

    // Third attempt: 30 minutes
    try {
      await attemptNotarization(appPath, credentials, 30, 3);
      return; // Success!
    } catch (thirdError) {
      console.log(`\n⚠️  Third attempt timed out after 30 minutes`);
      console.log('Checking if notarization ticket exists...');

      // Final check
      if (checkNotarizationTicket(appPath)) {
        console.log('✅ Notarization ticket found! The app was notarized successfully.');
        return;
      }

      // All attempts failed
      throw new Error(`Notarization failed after 3 attempts (5min + 10min + 30min). Apple's servers may be experiencing issues. Error: ${thirdError.message}`);
    }
  } catch (error) {
    console.error('\n❌ Notarization failed!');
    console.error('Error type:', error.constructor.name);
    console.error('Error message:', error.message);
    if (error.stack) {
      console.error('Stack trace:', error.stack);
    }
    console.error('\nFull error object:', JSON.stringify(error, null, 2));
    throw error;
  }
};
