// @vitest-environment jsdom
import { describeThemeContract } from "@knowva/core/testing";
import App from "./App";
import { Documents } from "./Documents";

describeThemeContract({
  name: "default",
  renderChat: () => <App />,
  renderDocuments: () => <Documents />,
  stylesheets: ["./index.css"],
  baseUrl: import.meta.url,
});
