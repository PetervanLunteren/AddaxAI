// One-line fix for a crash in electron-builder 25.1.8's NSIS template.
// Runs after every `npm install` / `npm ci` (the postinstall script).
//
// The bug: templates/nsis/multiUser.nsh asks Windows for the per-user
// programs folder (SHGetKnownFolderPath) and then copies the answer with
//
//   System::Call '*$2(&w${NSIS_MAX_STRLEN} .s)'
//
// which reads a fixed 8192 wide characters (16 KB) from a block that only
// holds the path. Usually the read runs into harmless memory and stops at
// a null. When the block sits near the end of a heap segment it hits
// unmapped memory and the installer dies: exit code -1073741819
// (0xC0000005), access violation in System.dll at offset 0x1581. It runs
// on every first per-user install of an account (an existing install in
// the registry skips it), double-clicked or silent, and crashes at random.
// Found on 2026-11-05 from a crash dump: 30 of 40 silent installs of
// 7.17.0 crashed on one test machine. The same crash is in that machine's
// event log on 2026-10-15, so every release built with 25.1.8 has it.
//
// The fix: the same copy bounded by lstrcpynW, which stops at the
// terminating null. `w .s` keeps the template's behaviour otherwise (the
// result goes to the stack, as before). Upstream fixed it the same way in
// electron-builder 26.12.0, PR #9769, "safely copy UserProgramFiles path":
// https://github.com/electron-userland/electron-builder/pull/9769
//
// Remove this script, and the postinstall line in package.json, when
// electron-builder is upgraded to 26.12.0 or newer. It refuses to run on
// such a version, so the upgrade cannot leave it behind unnoticed.

const fs = require("fs");
const path = require("path");

const lib = path.join(__dirname, "..", "node_modules", "app-builder-lib");
const version = require(path.join(lib, "package.json")).version;
const file = path.join(lib, "templates", "nsis", "multiUser.nsh");

const BROKEN = "System::Call '*$2(&w${NSIS_MAX_STRLEN} .s)'";
const FIXED = "System::Call 'KERNEL32::lstrcpynW(w .s, p r2, i ${NSIS_MAX_STRLEN})'";

if (!version.startsWith("25.")) {
  throw new Error(
    `app-builder-lib is ${version}, not 25.x. electron-builder 26.12.0 and ` +
      "newer fix this upstream: delete electron/scripts/patch-electron-builder.cjs " +
      "and its postinstall line in electron/package.json.",
  );
}

const text = fs.readFileSync(file, "utf8");
const broken = text.split(BROKEN).length - 1;
const fixed = text.split(FIXED).length - 1;

if (broken === 1 && fixed === 0) {
  fs.writeFileSync(file, text.replace(BROKEN, FIXED));
  console.log(`patched ${path.relative(process.cwd(), file)}: bounded SHGetKnownFolderPath copy`);
} else if (broken === 0 && fixed === 1) {
  console.log("multiUser.nsh already patched");
} else {
  throw new Error(
    `multiUser.nsh in app-builder-lib ${version} does not look as expected ` +
      `(${broken} broken, ${fixed} fixed copies of the line). Check the template ` +
      "by hand before building an installer.",
  );
}
