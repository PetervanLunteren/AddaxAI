/**
 * The Trend page saves its chart and its table as CSV. A species name
 * with a comma, or a filter list with a quote in it, must not shift the
 * columns when the file is opened.
 */

import { describe, expect, it } from "vitest";
import { toCsv } from "./csv";

describe("toCsv", () => {
  it("writes a header and one line per row", () => {
    expect(toCsv(["species", "mean"], [["Red deer", 1.25], ["Pig", 0]])).toBe(
      "species,mean\r\nRed deer,1.25\r\nPig,0\r\n",
    );
  });

  it("quotes a cell with a comma, a quote or a line break", () => {
    expect(toCsv(["a"], [["Deer, red"], ['the "big" one'], ["two\nlines"]])).toBe(
      'a\r\n"Deer, red"\r\n"the ""big"" one"\r\n"two\nlines"\r\n',
    );
  });

  it("leaves a missing value empty and keeps a zero", () => {
    expect(toCsv(["sd", "low", "high"], [[null, undefined, 0]])).toBe(
      "sd,low,high\r\n,,0\r\n",
    );
  });

  it("writes only the header for an empty table", () => {
    expect(toCsv(["a", "b"], [])).toBe("a,b\r\n");
  });
});
