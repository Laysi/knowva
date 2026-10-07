import { useCallback, useEffect, useState } from "react";

/** A page's heading and the sentence under it. */
export type ProfilePage = { title: string };

/**
 * A suggested question. `catalog` marks a whole-knowledge-base overview
 * question; the backend tests hold every profile to that marking.
 */
export type ProfileSuggestion = { text: string; catalog: boolean };

/**
 * The interface defaults of the deployed industry profile, exactly as
 * `GET /api/profile` returns them. Prompt text and catalog keywords never
 * leave the server.
 */
export type Profile = {
  id: string;
  theme: string;
  brand: { name: string; tagline: string };
  pages: { chat: ProfilePage; documents: ProfilePage };
  suggestions: ProfileSuggestion[];
  composerPlaceholder: string;
};

type ProfileStatus =
  | { status: "loading" }
  | { status: "ready"; profile: Profile }
  | { status: "error"; message: string };

/**
 * Where the profile request stands. A theme renders its layout in every
 * state: copy is absent while loading, and on failure the theme shows the
 * safe `message` while chat and documents keep working.
 */
export type ProfileState = ProfileStatus & {
  /** Requests the profile again, back through `loading`. */
  reload: () => void;
};

export function profileRequestErrorMessage(status?: number): string {
  if (status === undefined) {
    return "連不上伺服器，無法載入介面設定，請確認網路後再試。";
  }
  return "介面設定載入失敗，請稍後再試。";
}

const isString = (value: unknown): value is string => typeof value === "string";

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isPage = (value: unknown): value is ProfilePage =>
  isRecord(value) && isString(value.title);

const isSuggestion = (value: unknown): value is ProfileSuggestion =>
  isRecord(value) && isString(value.text) && typeof value.catalog === "boolean";

/** The response is rendered as text, so only its shape needs checking. */
export function isProfile(value: unknown): value is Profile {
  return (
    isRecord(value) &&
    isString(value.id) &&
    isString(value.theme) &&
    isRecord(value.brand) &&
    isString(value.brand.name) &&
    isString(value.brand.tagline) &&
    isRecord(value.pages) &&
    isPage(value.pages.chat) &&
    isPage(value.pages.documents) &&
    Array.isArray(value.suggestions) &&
    value.suggestions.every(isSuggestion) &&
    isString(value.composerPlaceholder)
  );
}

/**
 * Loads the deployed profile once. Every failure, including a response of
 * the wrong shape, ends in a safe message; the raw response never reaches
 * the page.
 */
export function useProfile(): ProfileState {
  const [state, setState] = useState<ProfileStatus>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    const settle = (next: ProfileStatus) => {
      if (!controller.signal.aborted) setState(next);
    };

    void (async () => {
      try {
        const res = await fetch("/api/profile", { signal: controller.signal });
        if (!res.ok) {
          settle({
            status: "error",
            message: profileRequestErrorMessage(res.status),
          });
          return;
        }
        const data: unknown = await res.json().catch(() => undefined);
        settle(
          isProfile(data)
            ? { status: "ready", profile: data }
            : { status: "error", message: profileRequestErrorMessage(502) },
        );
      } catch {
        settle({ status: "error", message: profileRequestErrorMessage() });
      }
    })();

    return () => controller.abort();
  }, [attempt]);

  const reload = useCallback(() => {
    setState({ status: "loading" });
    setAttempt((value) => value + 1);
  }, []);

  return { ...state, reload };
}
