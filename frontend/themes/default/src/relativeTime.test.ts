import { describe, expect, it } from "vitest";
import { relativeTime } from "./relativeTime";

// Local times, so the calendar rules hold in any time zone the tests run in.
const now = new Date(2026, 9, 7, 15, 30);
const at = (...parts: [number, number, number, number?, number?]) =>
  new Date(parts[0], parts[1], parts[2], parts[3] ?? 0, parts[4] ?? 0);

describe("relativeTime", () => {
  it.each([
    ["the same moment", now, "剛剛"],
    ["a time ahead of now", at(2026, 9, 7, 15, 35), "剛剛"],
    ["59 seconds ago", new Date(now.getTime() - 59_000), "剛剛"],
    ["5 minutes ago", at(2026, 9, 7, 15, 25), "5 分鐘前"],
    ["59 minutes ago", at(2026, 9, 7, 14, 31), "59 分鐘前"],
    ["2 hours ago on the same day", at(2026, 9, 7, 13, 0), "2 小時前"],
    ["just after midnight today", at(2026, 9, 7, 0, 5), "15 小時前"],
    ["late yesterday", at(2026, 9, 6, 23, 50), "昨天"],
    ["early yesterday", at(2026, 9, 6, 0, 10), "昨天"],
    ["two calendar days back", at(2026, 9, 5, 23, 0), "2 天前"],
    ["six calendar days back", at(2026, 9, 1, 9, 0), "6 天前"],
    ["a week back, this year", at(2026, 8, 30, 9, 0), "9/30"],
    ["last year", at(2025, 11, 31, 9, 0), "2025/12/31"],
  ])("reads %s", (_, then, expected) => {
    expect(relativeTime(then, now)).toBe(expected);
  });

  it("says 昨天 for an hour ago across midnight", () => {
    expect(relativeTime(at(2026, 9, 6, 23, 30), at(2026, 9, 7, 0, 30))).toBe("昨天");
  });
});
