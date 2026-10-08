// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { isProfile, profileRequestErrorMessage, useProfile } from "./useProfile";
import type { Profile } from "./useProfile";

const RAW_SECRET = "provider response: Authorization Bearer sk-secret-456";

const PROFILE: Profile = {
  id: "default",
  theme: "default",
  brand: { name: "Knowva 知星", tagline: "點亮你的知識星系" },
  pages: {
    chat: { title: "知識問答" },
    documents: { title: "文件管理" },
  },
  suggestions: [
    { text: "知識庫裡有哪些主題？", catalog: true },
    { text: "整理知識庫的重點", catalog: false },
  ],
  composerPlaceholder: "問問你的文件…",
};

/** Answers `GET /api/profile` from a queue; the last answer repeats. */
const stubProfile = (answers: (() => Response | Promise<Response>)[]) => {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    expect(`${init?.method ?? "GET"} ${url}`).toBe("GET /api/profile");
    const answer = answers.length > 1 ? answers.shift()! : answers[0]!;
    return answer();
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useProfile", () => {
  it("is loading until the profile arrives, then returns it", async () => {
    stubProfile([() => Response.json(PROFILE)]);
    const { result } = renderHook(() => useProfile());

    expect(result.current.status).toBe("loading");
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current).toMatchObject({ status: "ready", profile: PROFILE });
  });

  it("reports a safe message for a failed response without its body", async () => {
    stubProfile([() => new Response(RAW_SECRET, { status: 500 })]);
    const { result } = renderHook(() => useProfile());

    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current).toMatchObject({
      status: "error",
      message: profileRequestErrorMessage(500),
    });
    expect(JSON.stringify(result.current)).not.toContain(RAW_SECRET);
  });

  it("reports the network message when the request cannot connect", async () => {
    stubProfile([
      () => {
        throw new TypeError(`Failed to fetch: ${RAW_SECRET}`);
      },
    ]);
    const { result } = renderHook(() => useProfile());

    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current).toMatchObject({
      status: "error",
      message: profileRequestErrorMessage(),
    });
  });

  it("treats a body of the wrong shape as a failure", async () => {
    const { suggestions: _omitted, ...missingSuggestions } = PROFILE;
    for (const body of [
      () => new Response("not json", { status: 200 }),
      () => Response.json(missingSuggestions),
      () => Response.json({ ...PROFILE, brand: { name: RAW_SECRET } }),
    ]) {
      stubProfile([body]);
      const { result, unmount } = renderHook(() => useProfile());
      await waitFor(() => expect(result.current.status).toBe("error"));
      expect(result.current).toMatchObject({
        message: profileRequestErrorMessage(502),
      });
      expect(JSON.stringify(result.current)).not.toContain(RAW_SECRET);
      unmount();
    }
  });

  it("loads again on reload after a failure", async () => {
    const fetchMock = stubProfile([
      () => new Response(null, { status: 503 }),
      () => Response.json(PROFILE),
    ]);
    const { result } = renderHook(() => useProfile());
    await waitFor(() => expect(result.current.status).toBe("error"));

    act(() => result.current.reload());

    expect(result.current.status).toBe("loading");
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("checks every field the themes read", () => {
    expect(isProfile(PROFILE)).toBe(true);
    expect(isProfile({ ...PROFILE, composerPlaceholder: null })).toBe(false);
    expect(
      isProfile({ ...PROFILE, suggestions: [{ text: "沒有標記" }] }),
    ).toBe(false);
    expect(
      isProfile({ ...PROFILE, pages: { chat: PROFILE.pages.chat } }),
    ).toBe(false);
    expect(isProfile([PROFILE])).toBe(false);
  });
});
