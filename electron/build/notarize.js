const { notarize } = require('@electron/notarize');

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

  console.log('\n🔐 Starting notarization...');
  console.log('App name:', appName);
  console.log('App path:', appPath);
  console.log('Bundle ID: com.addaxai.cameratrap');
  console.log('Team ID:', process.env.APPLE_TEAM_ID);

  try {
    const startTime = Date.now();

    await notarize({
      tool: 'notarytool',
      appBundleId: 'com.addaxai.cameratrap',
      appPath: appPath,
      appleId: process.env.APPLE_ID,
      appleIdPassword: process.env.APPLE_APP_SPECIFIC_PASSWORD,
      teamId: process.env.APPLE_TEAM_ID,
    });

    const duration = Math.round((Date.now() - startTime) / 1000);
    console.log(`\n✅ Notarization complete for ${appName} (took ${duration}s)`);
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
