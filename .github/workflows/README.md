# GitHub Actions CI/CD

This directory contains automated workflows for building and releasing AddaxAI.

## Workflows

### `build-electron.yml` - Build Electron App

Automatically builds macOS installers for the AddaxAI desktop application.

**Triggers:**
- **Manual**: Go to [Actions tab](../../actions) → "Build Electron App" → "Run workflow"
- **Automatic**: Push a version tag (e.g., `v0.1.0`)

**Platforms:**
- macOS (Intel x64 + Apple Silicon arm64)

**Build outputs:**
- `AddaxAI-{version}-arm64.dmg` - macOS disk image installer
- `AddaxAI-{version}-arm64-mac.zip` - macOS ZIP archive

**Build time:** ~10 minutes

**GitHub Actions minutes used:** ~100 minutes per build (macOS has 10x multiplier)

## Usage

### Manual Build (Testing)

1. Go to the [Actions tab](../../actions)
2. Click "Build Electron App" workflow
3. Click "Run workflow" button
4. Select branch (usually `main`)
5. Click green "Run workflow" button

The installers will be available as downloadable artifacts for 90 days.

### Release Build (Production)

Create and push a version tag:

```bash
# Make sure you're on main and everything is committed
git checkout main
git pull

# Create a version tag
git tag v0.1.0 -m "Release v0.1.0"

# Push the tag
git push origin v0.1.0
```

This will:
1. Trigger the build workflow
2. Build macOS installers
3. Create a GitHub Release
4. Upload installers to the release
5. Generate release notes from commits

### Viewing Build Results

**Manual builds:**
- Go to Actions tab → Click the workflow run
- Download artifacts at the bottom of the page

**Release builds:**
- Go to [Releases page](../../releases)
- Download installers from the release

## Code Signing (Future)

Currently, builds are **unsigned**. Users will see security warnings when opening the app.

To enable code signing:

### macOS
1. Get an Apple Developer account ($99/year)
2. Create a Developer ID Application certificate
3. Export certificate as .p12 file
4. Add GitHub secrets:
   - `MACOS_CERTIFICATE`: Base64-encoded .p12 file
   - `MACOS_CERTIFICATE_PWD`: Certificate password
   - `APPLE_ID`: Your Apple ID email
   - `APPLE_ID_PASSWORD`: App-specific password
   - `APPLE_TEAM_ID`: Your team ID

5. Update workflow to enable signing (set `CSC_IDENTITY_AUTO_DISCOVERY: true`)

## Troubleshooting

### Build fails during PyInstaller step
- Check that `backend/requirements.txt` is complete
- Verify `backend/backend.spec` is correct
- Check build logs for missing dependencies

### Build fails during Electron packaging
- Verify `electron/package.json` is correct
- Check that frontend built successfully
- Ensure backend executable exists in `backend/dist/`

### Release not created
- Verify you pushed a tag starting with `v` (e.g., `v0.1.0`)
- Check that the workflow has `GITHUB_TOKEN` permissions
- Look for errors in the "Create GitHub Release" step

### Out of GitHub Actions minutes
- Free tier: 2000 minutes/month
- macOS builds use 10x multiplier (~100 min per build)
- Can upgrade to GitHub Pro for more minutes
- Or only trigger builds on releases, not every commit

## Cost Estimate

**GitHub Free Tier:**
- 2000 minutes/month included
- macOS builds: 10x multiplier
- Per build: ~10 min actual = 100 min charged
- **~20 builds per month on free tier**

**If you need more:**
- GitHub Pro: $4/month + 3000 minutes
- Pay-as-you-go: $0.08 per minute (macOS)
