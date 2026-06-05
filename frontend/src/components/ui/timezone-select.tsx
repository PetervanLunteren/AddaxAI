/**
 * Single-select searchable IANA timezone combobox.
 *
 * Two groups:
 *   1. "Fixed offset" — Etc/GMT±N zones for cameras set to a constant
 *      offset (no daylight saving), labeled as "UTC+3 (fixed, no
 *      daylight saving)".
 *   2. "Locations" — a flat alphabetical list of every regional zone,
 *      each entry labeled "🇰🇪 Kenya, Nairobi (UTC+03:00)". The offset
 *      is computed live via Intl.DateTimeFormat so it reflects the
 *      current DST state. Country names are localized via
 *      Intl.DisplayNames and flags are derived from the ISO alpha-2
 *      code via regional indicator symbols.
 *
 * The search filter matches country name, city, and IANA value — NOT
 * the "(UTC±HH:MM)" suffix. Otherwise typing "UTC" would match every
 * option.
 *
 * The country mapping lives in `src/geodata/timezone-countries.ts`,
 * ported from IANA zone.tab (see that file's header for refresh
 * instructions).
 */

import { useEffect, useMemo, useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react";

import {
  IANA_ALIAS,
  TIMEZONE_COUNTRY,
} from "../../geodata/timezone-countries";
import { cn } from "../../lib/utils";
import { Button } from "./button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "./command";
import { Popover, PopoverContent, PopoverTrigger } from "./popover";

interface TimezoneOption {
  /** Display label with flag + country + city + offset. */
  label: string;
  /** IANA name used as the value. */
  value: string;
  /** Country name searched against (empty for fixed-offset entries). */
  country: string;
  /** City searched against (or the display UTC label for fixed offsets). */
  city: string;
}

interface TimezoneGroup {
  heading: string;
  options: TimezoneOption[];
}

interface TimezoneSelectProps {
  /** Current IANA timezone string (e.g. "Europe/Amsterdam" or "Etc/GMT-3").
   *  An empty string selects the "Auto" entry when `autoLabel` is set. */
  value: string;
  /** Called with the newly selected IANA timezone, or "" for Auto. */
  onChange: (value: string) => void;
  /** Disable the control. */
  disabled?: boolean;
  /** When set, render an "Auto" entry at the top whose value is "".
   *  Used to let a project leave its timezone unset (auto-derived from
   *  site coordinates). */
  autoLabel?: string;
}

// Module-scoped instances so they build once per page load, not per option.
const regionNames = new Intl.DisplayNames(undefined, { type: "region" });
const collator = new Intl.Collator(undefined, { sensitivity: "base" });

/** Convert an ISO 3166-1 alpha-2 code like "KE" to its flag emoji 🇰🇪. */
function countryFlag(code: string): string {
  // Regional Indicator Symbol Letter A is U+1F1E6, ASCII 'A' is U+0041.
  return code
    .toUpperCase()
    .replace(/./g, (c) =>
      String.fromCodePoint(0x1f1e6 + c.charCodeAt(0) - 0x41),
    );
}

/**
 * Resolve a timezone name to its preferred display name. Browsers
 * surface both deprecated aliases (Asia/Saigon, Europe/Kiev) and
 * canonical names (Asia/Ho_Chi_Minh, Europe/Kyiv) from
 * Intl.supportedValuesOf. We prefer whichever name has a direct entry
 * in TIMEZONE_COUNTRY; only fall back to the IANA backward map when
 * neither the input nor its canonical is a real country-bound zone.
 */
function resolveTimezone(tz: string): string {
  if (TIMEZONE_COUNTRY[tz]) return tz;
  return IANA_ALIAS[tz] ?? tz;
}

/** Current UTC offset for a zone, formatted as UTC+03:00 / UTC-08:00. */
function formatOffset(tz: string): string {
  try {
    const parts = new Intl.DateTimeFormat("en-US", {
      timeZone: tz,
      timeZoneName: "longOffset",
    }).formatToParts(new Date());
    const raw = parts.find((p) => p.type === "timeZoneName")?.value ?? "";
    // Intl emits "GMT+03:00" / "GMT-08:00" / "GMT" (for UTC itself).
    if (raw === "GMT") return "UTC+00:00";
    return raw.replace("GMT", "UTC");
  } catch {
    return "";
  }
}

/** Fixed UTC offset group (no DST). Uses Etc/GMT± zone names. */
function buildFixedOffsetGroup(): TimezoneGroup {
  const options: TimezoneOption[] = [];
  for (let i = -12; i <= 14; i++) {
    if (i === 0) {
      options.push({
        label: "UTC (fixed, no daylight saving)",
        value: "UTC",
        country: "",
        city: "UTC",
      });
      continue;
    }
    // IANA convention inverts the sign: Etc/GMT-4 == UTC+4.
    const displaySign = i > 0 ? "+" : "";
    const ianaSign = i > 0 ? "-" : "+";
    const ianaValue = `Etc/GMT${ianaSign}${Math.abs(i)}`;
    const display = `UTC${displaySign}${i}`;
    options.push({
      label: `${display} (fixed, no daylight saving)`,
      value: ianaValue,
      country: "",
      city: display,
    });
  }
  return { heading: "Fixed offset", options };
}

/** Flat alphabetical group of regional zones with flag + country + city. */
function buildLocationsGroup(): TimezoneGroup {
  const anyIntl = Intl as unknown as {
    supportedValuesOf?: (key: string) => string[];
  };
  const timezones =
    typeof anyIntl.supportedValuesOf === "function"
      ? anyIntl.supportedValuesOf("timeZone")
      : [];

  const options: TimezoneOption[] = [];
  const seen = new Set<string>();

  for (const rawTz of timezones) {
    // Etc/* and bare UTC live in the fixed-offset group above.
    if (rawTz === "UTC" || rawTz.startsWith("Etc/")) continue;

    const tz = resolveTimezone(rawTz);
    if (tz === "UTC" || tz.startsWith("Etc/")) continue;
    if (seen.has(tz)) continue;
    seen.add(tz);

    const city = tz.split("/").pop()!.replace(/_/g, " ");
    const isoCode = TIMEZONE_COUNTRY[tz];
    const countryName = isoCode ? (regionNames.of(isoCode) ?? "") : "";
    const flag = isoCode ? countryFlag(isoCode) : "";
    const offset = formatOffset(tz);

    let label: string;
    if (countryName && offset) {
      label = `${flag} ${countryName}, ${city} (${offset})`;
    } else if (countryName) {
      label = `${flag} ${countryName}, ${city}`;
    } else if (offset) {
      label = `${city} (${offset})`;
    } else {
      label = city;
    }

    options.push({ label, value: tz, country: countryName, city });
  }

  // Sort: by localized country name, then city, then IANA value.
  options.sort((a, b) => {
    const byCountry = collator.compare(a.country, b.country);
    if (byCountry !== 0) return byCountry;
    const byCity = collator.compare(a.city, b.city);
    if (byCity !== 0) return byCity;
    return collator.compare(a.value, b.value);
  });

  return { heading: "Locations", options };
}

function buildGroups(): TimezoneGroup[] {
  return [buildFixedOffsetGroup(), buildLocationsGroup()];
}

export function TimezoneSelect({
  value,
  onChange,
  disabled,
  autoLabel,
}: TimezoneSelectProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");

  useEffect(() => {
    if (!open) setSearch("");
  }, [open]);

  const groups = useMemo(() => buildGroups(), []);

  // Find the option matching the current value so the trigger can show
  // its full label. Resolve aliases first so a saved value like
  // Asia/Saigon still matches its canonical entry (Asia/Ho_Chi_Minh).
  const currentLabel = useMemo(() => {
    if (!value) return autoLabel ?? "";
    const resolved = resolveTimezone(value);
    for (const group of groups) {
      const found = group.options.find((opt) => opt.value === resolved);
      if (found) return found.label;
    }
    return value;
  }, [groups, value, autoLabel]);

  // Filter on country, city, IANA value. Skipping the offset suffix
  // keeps "UTC" from matching every zone in the world.
  const filteredGroups = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return groups;
    const result: TimezoneGroup[] = [];
    for (const group of groups) {
      const filtered = group.options.filter((opt) => {
        return (
          opt.country.toLowerCase().includes(q) ||
          opt.city.toLowerCase().includes(q) ||
          opt.value.toLowerCase().includes(q)
        );
      });
      if (filtered.length > 0) {
        result.push({ heading: group.heading, options: filtered });
      }
    }
    return result;
  }, [groups, search]);

  const totalVisible = filteredGroups.reduce(
    (sum, g) => sum + g.options.length,
    0,
  );

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          disabled={disabled}
          className="w-full justify-between h-9 text-sm font-normal"
        >
          <span className="truncate">
            {currentLabel || "Select timezone"}
          </span>
          <ChevronsUpDown className="ml-1 h-3.5 w-3.5 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[360px] p-0" align="start">
        <Command shouldFilter={false}>
          <CommandInput
            placeholder="Search country or city..."
            value={search}
            onValueChange={setSearch}
          />
          <CommandList>
            <CommandEmpty>No timezones found.</CommandEmpty>
            {autoLabel && !search.trim() && (
              <CommandGroup>
                <CommandItem
                  value="__auto__"
                  onSelect={() => {
                    onChange("");
                    setOpen(false);
                  }}
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      !value ? "opacity-100" : "opacity-0",
                    )}
                  />
                  <span className="truncate">{autoLabel}</span>
                </CommandItem>
              </CommandGroup>
            )}
            {filteredGroups.map((group) => (
              <CommandGroup key={group.heading} heading={group.heading}>
                {group.options.map((opt) => {
                  const selected = opt.value === value;
                  return (
                    <CommandItem
                      key={opt.value}
                      value={opt.value}
                      onSelect={() => {
                        onChange(opt.value);
                        setOpen(false);
                      }}
                    >
                      <Check
                        className={cn(
                          "mr-2 h-4 w-4",
                          selected ? "opacity-100" : "opacity-0",
                        )}
                      />
                      <span className="truncate">{opt.label}</span>
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            ))}
          </CommandList>
          <div className="px-3 py-1.5 border-t text-xs text-muted-foreground">
            {totalVisible} timezone{totalVisible === 1 ? "" : "s"}
          </div>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
