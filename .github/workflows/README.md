# GitHub Actions CI/CD

## `build-electron.yml`

Builds the AddaxAI desktop app for macOS, Linux, and Windows. macOS is the
canonical target and is signed and notarized; Linux and Windows are
unsigned and currently non-blocking.

### Triggers

| Event | What happens |
|-------|--------------|
| `release.published` | A release was created in the GitHub UI (with notes) and clicked "Publish". The workflow builds binaries and attaches them to that release. |
| `workflow_dispatch` | Manual run from the Actions tab. Produces unsigned debug builds (workflow artifacts only, kept 14 days). |

There is no `push: tags: v*` trigger. Releases are deliberately a two-step
process: write release notes in the UI first, then publish. Failed builds
do not leave orphaned releases behind, and the tag can be retyped without
re-pushing.

### Build matrix

| OS | Runner | Arch | Outputs | Blocking |
|----|--------|------|---------|----------|
| macOS | `macos-14` | arm64 | `.dmg`, `-mac.zip` | Yes |
| Linux | `ubuntu-22.04` | x64 | `.deb` | No (`continue-on-error`) |
| Windows | `windows-2022` | x64 | `.exe` (NSIS), `-win.zip` | No (`continue-on-error`) |

macOS runs on Apple Silicon only. There is no Intel x64 build. Anyone on a
2019 or older Intel Mac will not get a working binary; this is a deliberate
trade-off (Apple stopped selling Intel Macs in 2023, and PyInstaller
universal builds are awkward).

Linux and Windows are non-blocking: the workflow is marked successful even
if either of them fails. Once each is verified end-to-end, flip
`continue-on-error: false` for that matrix entry.

### Versioning

The installer version is taken from the release tag at build time.
Pushing tag `v0.2.0` and publishing the release produces
`AddaxAI-0.2.0-arm64.dmg`. The hardcoded version in
`electron/package.json` is overwritten by `npm version` inside the
workflow. The git tag is the single source of truth.

`workflow_dispatch` runs do not change the version: they use whatever is
in `electron/package.json`.

### Code signing and notarization (macOS)

Required secrets (set in repo Settings → Secrets and variables → Actions):

| Secret | Purpose |
|--------|---------|
| `MACOS_CERTIFICATE` | Base64-encoded `.p12` of the Developer ID Application certificate |
| `MACOS_CERTIFICATE_PWD` | Password for the `.p12` file |
| `APPLE_ID` | Apple ID email |
| `APPLE_ID_PASSWORD` | App-specific password (created at appleid.apple.com) |
| `APPLE_TEAM_ID` | Apple Developer team ID |

On `release` events, the workflow fails fast if any of these are missing,
**before** any build runs. On `workflow_dispatch`, missing secrets are
allowed: the build proceeds and produces an unsigned `.app` for local
testing.

`electron/build/notarize.js` does a 3-attempt retry (5 → 10 → 30 min) and
checks for a stapled ticket between attempts. When `REQUIRE_NOTARIZATION=1`
is set (it is on release events), missing credentials raise instead of
silently skipping.

### Verification (macOS)

After packaging, the workflow checks:

1. `lipo -archs` on the main electron binary and the bundled PyInstaller
   backend. Both must report `arm64` exactly.
2. `codesign --verify --deep --strict --verbose=2` on the `.app` bundle.
3. `xcrun stapler validate` on the `.app` bundle (release events only).

If any check fails, the build fails. This catches the silent-skip failure
mode where electron-builder produces an unsigned-but-named-correctly
`.dmg`.

### Linux and Windows

No signing yet. Linux uses `electron-builder --linux` (deb only; targets
Ubuntu/Debian/Mint, which is the supported Linux audience for the beta).
The deb ships a custom after-install script
(`electron/build/deb-after-install.sh`) that sets the SUID bit on
chrome-sandbox unconditionally, because electron-builder's default
template tests for user namespaces as root and therefore skips the bit
on Ubuntu 23.10+, where AppArmor blocks unprivileged user namespaces at
runtime and the app aborts on launch. An AppImage was shipped before
this; it cannot work sandboxed on Ubuntu 23.10+ (nosuid FUSE mount), so
it was dropped. Windows uses `electron-builder --win` (NSIS + zip). End
users see SmartScreen / Gatekeeper warnings on these platforms; that's
expected until signing is wired up.

### Releasing

```bash
# 1. Tag a commit on main (locally)
git tag v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0

# 2. On GitHub: Releases → Draft a new release
#    Choose the tag, write notes, click "Publish release"

# 3. The workflow runs automatically and attaches binaries
```

You can also create the tag from the GitHub UI ("Releases → Draft a new
release → Choose a tag → Create new tag on publish"). Either path works.

### Manual debug build

`Actions tab → Build Electron App → Run workflow`. Produces unsigned
binaries as workflow artifacts (14-day retention). Only the macOS arm64
artifacts are guaranteed to load on a host machine; Linux/Windows artifacts
are best-effort while those matrix entries are non-blocking.

### Cost

GitHub Free tier: 2000 minutes/month. macOS uses a 10x multiplier, Linux
1x, Windows 2x. A full release run is ~10 min on macOS (= 100 min charged)
plus ~6 min Linux plus ~10 min Windows (= 20 min charged). Roughly 125
charged minutes per release, i.e. ~16 releases/month within the free tier.

### Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `Missing required secret: ...` at the top of the macOS job | Secrets not configured in repo settings |
| `lipo` reports `x86_64` instead of `arm64` | Runner image was bumped or PyInstaller was run on a different host arch. Check `matrix.runs-on` is `macos-14` |
| `codesign --verify` fails | `remove_python_signature.py` did not strip a stale signature. Inspect `_internal/` for foreign Team IDs |
| `xcrun stapler validate` fails | Notarization completed but ticket was not stapled. Re-run; usually transient |
| Notarization stuck > 30 min | Apple servers are slow. The hook does 3 retries totalling 45 min; if all 3 time out the workflow fails. Re-run when Apple recovers |
| `npm ci` fails on Windows | Lockfile drift (someone ran `npm install` without committing the lockfile) |
| Linux/Windows job marked yellow | They are non-blocking; the workflow still succeeded. Look at the failed step to fix |
