/**
 * Single-select searchable IANA timezone combobox.
 *
 * Two groups:
 *   1. "Fixed offset" — Etc/GMT±N zones for cameras set to a
 *      constant offset (no daylight saving), labeled as
 *      "UTC+3 (fixed, no daylight saving)".
 *   2. Geographic regions (Africa, America, Asia, ...) grouped
 *      alphabetically, each entry labeled "{City} (UTC±HH:MM)".
 *      The offset is computed live via Intl.DateTimeFormat so it
 *      reflects the current DST state.
 *
 * The search filter matches city name and IANA value only — NOT
 * the "(UTC±HH:MM)" suffix. Otherwise typing "UTC" would match
 * every option because they all contain that substring.
 *
 * Ported from AddaxAI-Connect's TimezoneSelect but uses the
 * Popover + Command (cmdk) pattern so it stays consistent with
 * the rest of the WebUI controls.
 */

import { useEffect, useMemo, useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react";

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
  /** Display label including the UTC offset. */
  label: string;
  /** IANA name used both as the value and the cmdk search key. */
  value: string;
  /** The part of the label users actually search against (city or "UTC+N"). */
  searchKey: string;
}

interface TimezoneGroup {
  heading: string;
  options: TimezoneOption[];
}

interface TimezoneSelectProps {
  /** Current IANA timezone string (e.g. "Europe/Amsterdam" or "Etc/GMT-3"). */
  value: string;
  /** Called with the newly selected IANA timezone. */
  onChange: (value: string) => void;
  /** Disable the control. */
  disabled?: boolean;
}

/** Fixed UTC offset group (no DST). Uses Etc/GMT± zone names. */
function buildFixedOffsetGroup(): TimezoneGroup {
  const options: TimezoneOption[] = [];
  for (let i = -12; i <= 14; i++) {
    if (i === 0) {
      options.push({
        label: "UTC (fixed, no daylight saving)",
        value: "UTC",
        searchKey: "UTC",
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
      searchKey: display,
    });
  }
  return { heading: "Fixed offset", options };
}

/** Build the geographic groups from Intl.supportedValuesOf. */
function buildGeographicGroups(): TimezoneGroup[] {
  const anyIntl = Intl as unknown as {
    supportedValuesOf?: (key: string) => string[];
  };
  const timezones =
    typeof anyIntl.supportedValuesOf === "function"
      ? anyIntl.supportedValuesOf("timeZone")
      : ["UTC"];

  const byRegion: Record<string, TimezoneOption[]> = {};
  const now = new Date();

  for (const tz of timezones) {
    const parts = tz.split("/");
    const region = parts[0];
    if (region === "Etc") continue; // handled in the fixed-offset group
    const city = parts[parts.length - 1].replace(/_/g, " ");

    // Current offset string from Intl (e.g., "GMT+02:00"). Rewrite
    // GMT → UTC so the label matches user expectations.
    let utcOffset = "";
    try {
      const formatter = new Intl.DateTimeFormat("en-US", {
        timeZone: tz,
        timeZoneName: "shortOffset",
      });
      const offsetPart = formatter
        .formatToParts(now)
        .find((p) => p.type === "timeZoneName");
      utcOffset = (offsetPart?.value ?? "").replace("GMT", "UTC");
    } catch {
      // If Intl rejects the zone for some reason, skip the offset.
      utcOffset = "";
    }

    const label = utcOffset ? `${city} (${utcOffset})` : city;
    if (!byRegion[region]) byRegion[region] = [];
    byRegion[region].push({ label, value: tz, searchKey: city });
  }

  // Sort entries inside each region, then return regions alphabetically.
  for (const region of Object.keys(byRegion)) {
    byRegion[region].sort((a, b) => a.label.localeCompare(b.label));
  }
  return Object.keys(byRegion)
    .sort()
    .map((region) => ({ heading: region, options: byRegion[region] }));
}

function buildGroups(): TimezoneGroup[] {
  return [buildFixedOffsetGroup(), ...buildGeographicGroups()];
}

export function TimezoneSelect({
  value,
  onChange,
  disabled,
}: TimezoneSelectProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");

  useEffect(() => {
    if (!open) setSearch("");
  }, [open]);

  const groups = useMemo(() => buildGroups(), []);

  // Find the option for the current value so the trigger can show
  // its full label (with offset), not just the raw IANA string.
  const currentLabel = useMemo(() => {
    for (const group of groups) {
      const found = group.options.find((opt) => opt.value === value);
      if (found) return found.label;
    }
    return value;
  }, [groups, value]);

  // Filter logic: match the IANA value OR the searchKey (city / "UTC+N").
  // We deliberately don't match against the "(UTC±HH:MM)" suffix so
  // typing "utc" doesn't return every timezone in the world.
  const filteredGroups = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return groups;
    const result: TimezoneGroup[] = [];
    for (const group of groups) {
      const filtered = group.options.filter((opt) => {
        return (
          opt.value.toLowerCase().includes(q) ||
          opt.searchKey.toLowerCase().includes(q)
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
    0
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
          <span className="truncate">{currentLabel || "Select timezone"}</span>
          <ChevronsUpDown className="ml-1 h-3.5 w-3.5 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[320px] p-0" align="start">
        <Command shouldFilter={false}>
          <CommandInput
            placeholder="Search by city or UTC offset..."
            value={search}
            onValueChange={setSearch}
          />
          <CommandList>
            <CommandEmpty>No timezones found.</CommandEmpty>
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
                          selected ? "opacity-100" : "opacity-0"
                        )}
                      />
                      <span>{opt.label}</span>
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
