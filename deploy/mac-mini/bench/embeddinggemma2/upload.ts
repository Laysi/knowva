// Uploads every document in a directory to an isolated Knowva instance, one at
// a time, and waits for each to reach ready (summary included) before the next.
//   bun upload.ts <docs-dir> <base-url>
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const [dir, base] = process.argv.slice(2);
const order = [
  "iiot_industry_report.pdf", "iiot_maintenance_report.docx", "iiot_sensor_inventory.xlsx",
  "iiot_smart_factory_briefing.pptx", "iiot_factory_robot.jpg", "iiot_giga_press.png",
  "iiot_sensor_alert.eml", "iiot_load_cell_sensor.stp",
];
const names = readdirSync(dir).sort((a, b) => {
  const ia = order.indexOf(a), ib = order.indexOf(b);
  return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
});

for (const name of names) {
  const began = performance.now();
  const form = new FormData();
  form.append("file", new File([readFileSync(join(dir, name))], name));
  const upload = await fetch(`${base}/api/documents`, { method: "POST", body: form });
  const body = await upload.json();
  if (!upload.ok) {
    console.log(JSON.stringify({ name, status: upload.status, body }));
    // Keep going to report every file, but a partial corpus must not look like success.
    process.exitCode = 1;
    continue;
  }
  const id = body.id;
  let doc: any;
  let readyAt: number | undefined;
  for (;;) {
    await Bun.sleep(2000);
    const list = await (await fetch(`${base}/api/documents`)).json();
    doc = (Array.isArray(list) ? list : list.documents).find((d: any) => d.id === id);
    if (!doc || doc.status === "failed" || doc.processingError) break;
    if (doc.status === "ready") readyAt ??= performance.now();
    if (readyAt !== undefined && (doc.summary || performance.now() - readyAt > 300_000)) break;
  }
  console.log(JSON.stringify({
    name, id, status: doc?.status, error: doc?.processingError ?? doc?.error,
    ready_s: readyAt === undefined ? null : +((readyAt - began) / 1000).toFixed(1),
    summary: Boolean(doc?.summary),
  }));
  if (doc?.status !== "ready") process.exitCode = 1;
}
