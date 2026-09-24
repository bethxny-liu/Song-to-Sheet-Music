const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

// Run the actual TypeScript client with a mocked network, using existing tooling.
const code = ts.transpileModule(
  fs.readFileSync(path.join(__dirname, "../lib/api.ts"), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }
).outputText;

function client(fetch) {
  const exports = {};
  vm.runInNewContext(code, { exports, fetch, process: { env: {} } });
  return exports;
}

test("plain-text and HTML server errors are not reported as network failures", async () => {
  for (const body of ["Internal Server Error", "<html>Bad gateway</html>"]) {
    const api = client(async () => new Response(body, { status: 500 }));
    await assert.rejects(api.convertAudio(new FormData()), /Conversion failed on the server/);
  }
});

test("validation errors identify the field and explanation", async () => {
  const api = client(async () => Response.json({ detail: [
    { loc: ["body", "tempo_bpm"], msg: "Input should be greater than or equal to 40" }
  ] }, { status: 422 }));
  await assert.rejects(api.convertAudio(new FormData()), /tempo_bpm: Input should be greater than or equal to 40/);
});

test("audio validation messages are shown directly", async () => {
  const api = client(async () => Response.json({ detail: "Recording is silent." }, { status: 422 }));
  await assert.rejects(api.convertAudio(new FormData()), /Recording is silent/);
});

test("only fetch failures show the backend connection message", async () => {
  const api = client(async () => { throw new TypeError("Failed to fetch"); });
  await assert.rejects(api.convertAudio(new FormData()), /Cannot reach backend/);
});

test("a malformed successful response has a useful error", async () => {
  const api = client(async () => new Response("not JSON"));
  await assert.rejects(api.convertAudio(new FormData()), /unreadable conversion result/);
});

test("successful uploads preserve the form and return the result", async () => {
  const form = new FormData();
  form.set("tempo_bpm", "120");
  const api = client(async (url, options) => {
    assert.equal(url, "http://localhost:8000/jobs/convert");
    assert.equal(options.method, "POST");
    assert.equal(options.body, form);
    return Response.json({ job_id: "test" });
  });
  assert.equal((await api.convertAudio(form)).job_id, "test");
});
