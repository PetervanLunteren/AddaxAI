/**
 * Unit tests for the path-math helpers behind the relink flows.
 *
 * The Windows cases are the point: the backend runs on the user's OS and
 * sends native paths, so these helpers must handle backslash paths on a
 * machine where they can never occur at runtime in development. The
 * mixed-separator case is a shipped regression: RelinkGroupBanner appends
 * "/" to the group prefix, which on Windows produced a prefix that
 * replacePrefix could not match, so every bulk relink sent the old path
 * back to the backend and failed with "Folder not found".
 */

import { describe, expect, it } from "vitest";
import { replacePrefix } from "./path-utils";

describe("replacePrefix", () => {
  describe("POSIX paths", () => {
    it("replaces the prefix of a child path", () => {
      expect(
        replacePrefix("/old/root/site/dep001", "/old/root", "/new/root")
      ).toBe("/new/root/site/dep001");
    });

    it("replaces the whole path when it equals the prefix", () => {
      expect(replacePrefix("/old/root", "/old/root", "/new/root")).toBe(
        "/new/root"
      );
    });

    it("leaves a path outside the prefix unchanged", () => {
      expect(replacePrefix("/elsewhere/dep001", "/old/root", "/new/root")).toBe(
        "/elsewhere/dep001"
      );
    });

    it("ignores trailing separators on both prefixes", () => {
      expect(
        replacePrefix("/old/root/dep001", "/old/root/", "/new/root/")
      ).toBe("/new/root/dep001");
    });

    it("does not match a sibling whose name shares the prefix string", () => {
      expect(
        replacePrefix("/old/root2/dep001", "/old/root", "/new/root")
      ).toBe("/old/root2/dep001");
    });
  });

  describe("Windows paths", () => {
    it("replaces the prefix of a child path", () => {
      expect(
        replacePrefix(
          "C:\\Users\\Terry\\Desktop\\TrailCam\\2026-09-22 +1",
          "C:\\Users\\Terry\\Desktop\\TrailCam",
          "D:\\AddaxAI\\TrailCam"
        )
      ).toBe("D:\\AddaxAI\\TrailCam\\2026-09-22 +1");
    });

    it("replaces the whole path when it equals the prefix", () => {
      expect(
        replacePrefix(
          "C:\\Users\\Terry\\Desktop\\TrailCam",
          "C:\\Users\\Terry\\Desktop\\TrailCam",
          "D:\\AddaxAI\\TrailCam"
        )
      ).toBe("D:\\AddaxAI\\TrailCam");
    });

    it("substitutes when the old prefix carries an appended forward slash", () => {
      // The regression: the banner hands the dialog `missingPath + "/"`,
      // so on Windows the prefix arrives with a foreign separator at the
      // end. It must still match the backslash path.
      expect(
        replacePrefix(
          "C:\\Users\\Terry\\Desktop\\TrailCam\\2026-09-22 +1",
          "C:\\Users\\Terry\\Desktop\\TrailCam/",
          "D:\\AddaxAI\\TrailCam"
        )
      ).toBe("D:\\AddaxAI\\TrailCam\\2026-09-22 +1");
    });

    it("keeps the new prefix's separator style in the result", () => {
      expect(
        replacePrefix(
          "C:\\old\\dep001",
          "C:\\old",
          "/Volumes/Drive/new"
        )
      ).toBe("/Volumes/Drive/new/dep001");
    });
  });
});
