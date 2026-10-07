const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

const startOfDay = (time: Date) =>
  new Date(time.getFullYear(), time.getMonth(), time.getDate()).getTime();

const exact = new Intl.DateTimeFormat("zh-TW", {
  dateStyle: "medium",
  timeStyle: "short",
});

/**
 * How long ago `then` was, in the local calendar: 剛剛, minutes, hours within
 * the same day, 昨天, days within a week, then the date. A time ahead of `now`
 * (a clock running behind the server's) reads as 剛剛.
 */
export function relativeTime(then: Date, now: Date): string {
  const elapsed = now.getTime() - then.getTime();
  if (elapsed < MINUTE) return "剛剛";
  if (elapsed < HOUR) return `${Math.floor(elapsed / MINUTE)} 分鐘前`;
  // Rounded so a daylight-saving day still counts as one.
  const days = Math.round((startOfDay(now) - startOfDay(then)) / DAY);
  if (days <= 0) return `${Math.floor(elapsed / HOUR)} 小時前`;
  if (days === 1) return "昨天";
  if (days < 7) return `${days} 天前`;
  const date = `${then.getMonth() + 1}/${then.getDate()}`;
  return then.getFullYear() === now.getFullYear()
    ? date
    : `${then.getFullYear()}/${date}`;
}

/** The full local date and time, for a tooltip. */
export const exactTime = (time: Date): string => exact.format(time);
