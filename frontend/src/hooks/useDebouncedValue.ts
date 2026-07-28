/**
 * Debounce a value by `delay` ms.
 *
 * Shared by every surface that drives an expensive query from a live
 * control (confidence sliders, filter bars): the control, its readout,
 * and URL state stay live on `value`, while the returned value only
 * settles `delay` ms after the last change, so the query fires once per
 * interaction instead of once per step.
 *
 * Compares by JSON serialization, so it works for object values
 * (filter maps) as well as primitives.
 */

import { useEffect, useState } from "react";

export function useDebouncedValue<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  const serialized = JSON.stringify(value);
  useEffect(() => {
    const timer = setTimeout(
      () => setDebounced(JSON.parse(serialized)),
      delay,
    );
    return () => clearTimeout(timer);
  }, [serialized, delay]);
  return debounced;
}

/** Standard debounce for filter / slider driven queries. */
export const FILTER_DEBOUNCE_MS = 300;
