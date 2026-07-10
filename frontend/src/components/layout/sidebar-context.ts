/**
 * Sidebar collapse state, shared with content pages.
 *
 * AppLayout owns the collapsed flag (the content margin tracks the rail
 * width). Any content that pins itself to the sidebar edge, e.g. a
 * `fixed` full-width footer bar, reads this to stay aligned when the
 * rail narrows. Defaults to false so a page rendered outside AppLayout
 * behaves as if the full sidebar is present.
 */

import { createContext, useContext } from "react";

export const SidebarCollapseContext = createContext(false);

export const useSidebarCollapsed = (): boolean =>
  useContext(SidebarCollapseContext);
