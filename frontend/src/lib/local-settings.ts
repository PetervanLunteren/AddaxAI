/**
 * Small things a person sets once and expects to stay put, kept in
 * localStorage: a tile size, a panel height. Not the URL, because they
 * describe how someone likes to work, not what they are looking at, so
 * they survive a fresh link and do not travel with a shared one.
 *
 * One key may hold a plain value or a JSON object of fields; a field
 * write is read-modify-write, so two settings under one key written
 * from different places cannot clobber each other. Every page store
 * (`labels-settings.ts`, the Counts modal) and the resizable panels go
 * through here, so there is one copy of the try/catch and one rule for
 * a full or blocked localStorage: it is not worth failing a render over.
 */

export function readSetting(storageKey: string, field?: string): unknown {
  try {
    const raw = localStorage.getItem(storageKey);
    if (raw == null) return undefined;
    const value: unknown = JSON.parse(raw);
    if (field === undefined) return value;
    return value && typeof value === "object"
      ? (value as Record<string, unknown>)[field]
      : undefined;
  } catch {
    return undefined;
  }
}

export function persistSetting(storageKey: string, value: unknown, field?: string): void {
  try {
    if (field === undefined) {
      localStorage.setItem(storageKey, JSON.stringify(value));
      return;
    }
    const current = readSetting(storageKey);
    const object =
      current && typeof current === "object" ? (current as Record<string, unknown>) : {};
    object[field] = value;
    localStorage.setItem(storageKey, JSON.stringify(object));
  } catch {
    /* A full or blocked localStorage is not worth failing a render over. */
  }
}
