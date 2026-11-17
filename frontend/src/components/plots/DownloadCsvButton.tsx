/**
 * The "Download CSV" button a card carries in its header to save the
 * rows it shows. One component so the two Trend cards cannot drift.
 */

import { Download } from "lucide-react";

import { Button } from "../ui/button";

export function DownloadCsvButton({
  onClick,
  disabled,
}: {
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <Button variant="outline" size="sm" className="h-9" onClick={onClick} disabled={disabled}>
      <Download className="mr-2 h-4 w-4" />
      Download CSV
    </Button>
  );
}
