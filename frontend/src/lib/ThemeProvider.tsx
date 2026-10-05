/**
 * The theme context provider. All logic lives in theme.ts (see the
 * fast-refresh note there); this file only exports the component.
 */

import type { ReactNode } from "react";

import { ThemeContext, useThemeState } from "./theme";

export function ThemeProvider({ children }: { children: ReactNode }) {
  const value = useThemeState();
  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}
