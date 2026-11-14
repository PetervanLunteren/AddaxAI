import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("sonner", () => ({
  toast: {
    loading: vi.fn(() => "toast-id"),
    dismiss: vi.fn(),
    error: vi.fn(),
  },
}));
vi.mock("../api/diagnostics", () => ({
  diagnosticsApi: { downloadDiagnosticZip: vi.fn() },
}));

import { toast } from "sonner";
import { diagnosticsApi } from "../api/diagnostics";
import { exportDiagnosticReport } from "./diagnostic-export";

const download = vi.mocked(diagnosticsApi.downloadDiagnosticZip);

beforeEach(() => {
  vi.clearAllMocks();
  download.mockResolvedValue(undefined);
});

describe("exportDiagnosticReport", () => {
  it("shows a loading toast while it builds and dismisses it when done", async () => {
    let finish!: () => void;
    download.mockReturnValue(new Promise<void>((r) => (finish = r)));

    const run = exportDiagnosticReport();
    expect(toast.loading).toHaveBeenCalledTimes(1);
    expect(toast.dismiss).not.toHaveBeenCalled();

    finish();
    await run;
    expect(toast.dismiss).toHaveBeenCalledWith("toast-id");
    expect(toast.error).not.toHaveBeenCalled();
  });

  it("turns the loading toast into the error when the build fails", async () => {
    download.mockRejectedValue(new Error("status 500"));

    await exportDiagnosticReport();

    expect(toast.error).toHaveBeenCalledWith(
      "Could not build diagnostic report: status 500",
      { id: "toast-id" },
    );
    expect(toast.dismiss).not.toHaveBeenCalled();
  });

  it("ignores a click while a build is running", async () => {
    let finish!: () => void;
    download.mockReturnValue(new Promise<void>((r) => (finish = r)));

    const first = exportDiagnosticReport();
    await exportDiagnosticReport();
    expect(download).toHaveBeenCalledTimes(1);
    expect(toast.loading).toHaveBeenCalledTimes(1);

    finish();
    await first;
  });

  it("builds again after a build has finished, also a failed one", async () => {
    download.mockRejectedValueOnce(new Error("status 500"));
    await exportDiagnosticReport();
    await exportDiagnosticReport();

    expect(download).toHaveBeenCalledTimes(2);
  });
});
