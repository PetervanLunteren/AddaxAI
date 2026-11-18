/**
 * The download icon in an Insights page's header. It always opens a
 * small menu, also for one item: the same arrow saves data on one page
 * and a picture on another, and the menu says which before it saves.
 *
 * A quiet icon, because few users need it and it should not compete
 * with the title: the same form as the row menus on the Sites page
 * (ghost, 32 px). One component, so every page that offers a download
 * looks and behaves the same. The rule for what a page offers is in
 * FRONTEND_CONVENTIONS.md.
 */

import { Download } from "lucide-react";

import { Button } from "../ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";

/** The two things a page can offer, worded in one place. */
export const DATA_CSV = "Data (CSV)";
export const IMAGE_PNG = "Image (PNG)";

export interface DownloadItem {
  label: string;
  onSelect: () => void;
}

/** `items` is what can be saved right now. With none, the icon is
 *  disabled: the page has nothing to save yet. */
export function DownloadMenu({ items }: { items: DownloadItem[] }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          aria-label="Download"
          title="Download"
          disabled={items.length === 0}
        >
          <Download className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {items.map((item) => (
          <DropdownMenuItem key={item.label} onClick={item.onSelect}>
            {item.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
