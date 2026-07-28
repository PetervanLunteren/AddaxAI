import type { ReactNode } from "react";
import styles from "./styles.module.css";

// Download UI for the install page.
// - <DownloadCards /> renders one card per OS (a picker).
// - <DownloadCards os="windows" /> renders a single button for that OS,
//   used inside the per-OS install tabs so the download follows the choice.

const RELEASE =
  "https://github.com/PetervanLunteren/AddaxAI/releases/latest/download";

type OsKey = "windows" | "macos" | "linux";

interface Platform {
  name: string;
  note: string;
  file: string;
  icon: ReactNode;
}

const WindowsIcon = (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M3 5.5 10.2 4.5v7H3V5.5Zm8.4-1.2L21 3v8.5h-9.6V4.3ZM3 12.5h7.2v7L3 18.5v-6Zm8.4 0H21V21l-9.6-1.3v-7.2Z" />
  </svg>
);

const AppleIcon = (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M16.4 12.8c0-2.3 1.9-3.4 2-3.5-1.1-1.6-2.8-1.8-3.4-1.8-1.4-.1-2.8.8-3.5.8-.7 0-1.9-.8-3.1-.8-1.6 0-3.1.9-3.9 2.4-1.7 2.9-.4 7.2 1.2 9.6.8 1.2 1.7 2.5 3 2.4 1.2 0 1.6-.8 3.1-.8 1.4 0 1.8.8 3.1.7 1.3 0 2.1-1.2 2.9-2.3.9-1.3 1.3-2.6 1.3-2.7 0 0-2.5-1-2.5-3.9Zm-2.3-7.2c.7-.8 1.1-1.9 1-3-.9 0-2.1.6-2.8 1.4-.6.7-1.1 1.8-1 2.9 1 .1 2.1-.5 2.8-1.3Z" />
  </svg>
);

const LinuxIcon = (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M12 2c-2.2 0-3.6 1.7-3.6 3.9 0 1 .1 1.7-.3 2.5-.5 1-1.6 2.2-2.3 3.7-.5 1.1-.3 2 .1 2.4.2.2.1.6-.2 1.2-.5 1-.9 2.1-.3 2.8.6.7 1.9.4 3 .8.9.3 1.6 1 2.5 1.1.6.1 1.2.1 1.8 0 .9-.1 1.6-.8 2.5-1.1 1.1-.4 2.4-.1 3-.8.6-.7.2-1.8-.3-2.8-.3-.6-.4-1-.2-1.2.4-.4.6-1.3.1-2.4-.7-1.5-1.8-2.7-2.3-3.7-.4-.8-.3-1.5-.3-2.5C15.6 3.7 14.2 2 12 2Zm-1.5 3.3c.4 0 .7.4.7.9s-.3.9-.7.9-.7-.4-.7-.9.3-.9.7-.9Zm3 0c.4 0 .7.4.7.9s-.3.9-.7.9-.7-.4-.7-.9.3-.9.7-.9ZM12 8.2c.7 0 1.5.4 1.5.8 0 .3-.7.8-1.5.8s-1.5-.5-1.5-.8c0-.4.8-.8 1.5-.8Z" />
  </svg>
);

const PLATFORMS: Record<OsKey, Platform> = {
  windows: { name: "Windows", note: "Windows 10 and 11", file: "AddaxAI-Setup.exe", icon: WindowsIcon },
  macos: { name: "macOS", note: "Apple Silicon only", file: "AddaxAI-arm64.dmg", icon: AppleIcon },
  linux: { name: "Linux", note: "Debian and Ubuntu", file: "AddaxAI-amd64.deb", icon: LinuxIcon },
};

const ORDER: OsKey[] = ["windows", "macos", "linux"];

export default function DownloadCards({ os }: { os?: OsKey }): ReactNode {
  if (os) {
    const p = PLATFORMS[os];
    return (
      <div className={styles.single}>
        <a className={styles.singleBtn} href={`${RELEASE}/${p.file}`}>
          <span className={styles.singleIcon}>{p.icon}</span>
          Download for {p.name}
        </a>
        <div className={styles.singleMeta}>
          {p.note} · <code className={styles.file}>{p.file}</code>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.row}>
      {ORDER.map((key) => {
        const p = PLATFORMS[key];
        return (
          <a key={key} className={styles.card} href={`${RELEASE}/${p.file}`}>
            <span className={styles.icon}>{p.icon}</span>
            <span className={styles.name}>{p.name}</span>
            <span className={styles.note}>{p.note}</span>
            <span className={styles.button}>Download</span>
            <code className={styles.file}>{p.file}</code>
          </a>
        );
      })}
    </div>
  );
}
