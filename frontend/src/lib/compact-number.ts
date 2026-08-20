/**
 * Short number for places where the full figure does not fit.
 *
 * 132, 1.3K, 12K. Used on the Labels page's tab chips, where a count of
 * several thousand has to sit inside a segmented control next to the
 * sort dropdown.
 *
 * `Intl.NumberFormat` does this natively, so there is no rounding of our
 * own to get wrong and it follows the user's locale.
 */

// Pinned to "en" rather than the browser locale, which is the one
// deliberate choice here: en-GB renders "4.5k" and en-US "4.5K", and a
// chip that changes case depending on the machine is not worth the
// authenticity. The app is English-only today; when multi-language
// support lands this is the line to revisit.
const COMPACT = new Intl.NumberFormat("en", {
  notation: "compact",
  maximumFractionDigits: 1,
});

export function compactNumber(value: number): string {
  return COMPACT.format(value);
}
