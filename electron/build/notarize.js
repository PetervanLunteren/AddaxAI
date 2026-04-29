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
    if (process.env.REQUIRE_NOTARIZATION === '1') {
      throw new Error(
        'Notarization required (REQUIRE_NOTARIZATION=1) but credentials are missing. ' +
        'Set APPLE_ID, APPLE_APP_SPECIFIC_PASSWORD, and APPLE_TEAM_ID.'
      );
    }
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

  // Some notarytool errors are persistent: retrying just wastes the budget.
  // Detect them and short-circuit. Match on the error MESSAGE (notarytool
  // surfaces these strings inside a wrapped Error from the @electron/notarize
  // promise rejection).
  const isNonRetryable = (err) => {
    const msg = (err && err.message) || '';
    return (
      /HTTP status code:\s*4\d\d/i.test(msg) ||                  // any 4xx from notarytool
      /required agreement is missing or has expired/i.test(msg) || // agreements not accepted
      /Invalid credentials/i.test(msg) ||                        // bad APPLE_ID / app-specific password
      /Could not find team/i.test(msg) ||                        // bad team ID
      /Authentication failed/i.test(msg)
    );
  };

  const attempts = [
    { minutes: 5, num: 1 },
    { minutes: 10, num: 2 },
    { minutes: 30, num: 3 },
  ];

  let lastError;
  try {
    for (let i = 0; i < attempts.length; i++) {
      const { minutes, num } = attempts[i];
      try {
        await attemptNotarization(appPath, credentials, minutes, num);
        return; // Success!
      } catch (err) {
        lastError = err;
        console.log(`\n⚠️  Attempt ${num} ended after ~${minutes} min: ${err.message}`);

        if (isNonRetryable(err)) {
          console.log('❌ Non-retryable error detected (4xx / agreement / auth). Skipping further attempts.');
          break;
        }

        // Apple is slow but the upload may still have landed: check for ticket.
        if (checkNotarizationTicket(appPath)) {
          console.log('✅ Notarization ticket found! The app was notarized successfully.');
          return;
        }

        if (i < attempts.length - 1) {
          console.log(`❌ No ticket found. Retrying with ${attempts[i + 1].minutes}-minute timeout...`);
        }
      }
    }

    // Final ticket check (in case the very last attempt timed out but
    // notarization actually completed).
    if (checkNotarizationTicket(appPath)) {
      console.log('✅ Notarization ticket found on final check.');
      return;
    }

    throw new Error(`Notarization failed. Last error: ${lastError && lastError.message}`);
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
