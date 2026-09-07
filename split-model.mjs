#!/usr/bin/env node
/**
 * split-model.mjs
 *
 * Streams an ONNX model repo from the Hugging Face Hub straight into
 * sequential .partNNNNN chunk files as the bytes arrive — the full,
 * unsplit file is never written to disk, only the chunks are. Files
 * that end up fitting in a single chunk are written directly under
 * their original name (no chunking needed). Writes a manifest.json
 * describing how to reassemble everything.
 *
 * Requires Node 18+ (uses the built-in fetch).
 *
 * Usage:
 *   node split-model.mjs \
 *     --model onnx-community/Qwen3-4B-ONNX \
 *     --dtype q4f16 \
 *     --out ./qwen3-4b-cdn \
 *     --chunk-mb 19
 *
 * Upload the resulting output folder (including manifest.json) to your CDN
 * as-is, preserving the folder structure (e.g. the "onnx/" subfolder).
 */

import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    model: "onnx-community/Qwen3-4B-ONNX",
    dtype: "q4f16",
    out: "./qwen3-4b-cdn",
    chunkMb: 19,
  };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--model") opts.model = args[++i];
    else if (a === "--dtype") opts.dtype = args[++i];
    else if (a === "--out") opts.out = args[++i];
    else if (a === "--chunk-mb") opts.chunkMb = Number(args[++i]);
    else {
      console.error(`Unknown argument: ${a}`);
      process.exit(1);
    }
  }
  return opts;
}

const opts = parseArgs();
const CHUNK_SIZE = Math.round(opts.chunkMb * 1024 * 1024);

async function listRepoFiles(model) {
  const res = await fetch(`https://huggingface.co/api/models/${model}`);
  if (!res.ok) throw new Error(`Failed to list files for ${model}: ${res.status} ${res.statusText}`);
  const json = await res.json();
  return json.siblings.map((s) => s.rfilename);
}

// Keep every root-level file (config.json, tokenizer.json, etc.), and only
// the onnx/ subfolder files that match the requested dtype suffix, mirroring
// how Transformers.js picks a single weight file for a given dtype.
function shouldKeep(file, dtype) {
  if (!file.includes("/")) return true;
  if (file.startsWith("onnx/")) {
    const base = file.slice("onnx/".length);
    return base.includes(`_${dtype}.`);
  }
  return false;
}

function toPosix(p) {
  return p.split(path.sep).join("/");
}

function writeAsync(stream, buf) {
  return new Promise((resolve, reject) => stream.write(buf, (err) => (err ? reject(err) : resolve())));
}

function endAsync(stream) {
  return new Promise((resolve) => stream.end(resolve));
}

/**
 * Streams `rfile` from the Hub straight into fixed-size chunk files under
 * outDir, without ever holding the whole file in memory or on disk as a
 * single blob. Returns { size, chunkPaths } where chunkPaths is the
 * relative (posix) path of every chunk written, in order.
 */
async function downloadAndChunk(model, rfile, outDir) {
  const url = `https://huggingface.co/${model}/resolve/main/${rfile}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download ${rfile}: ${res.status} ${res.statusText}`);

  const destDir = path.dirname(path.join(outDir, rfile));
  await fsp.mkdir(destDir, { recursive: true });

  const chunkPath = (idx) => path.join(outDir, `${rfile}.part${String(idx).padStart(5, "0")}`);

  let chunkIdx = 0;
  let bytesInChunk = 0;
  let totalSize = 0;
  const chunkPaths = [];
  let ws = fs.createWriteStream(chunkPath(chunkIdx));

  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;

    let buf = Buffer.from(value.buffer, value.byteOffset, value.byteLength);
    totalSize += buf.length;

    while (buf.length > 0) {
      const room = CHUNK_SIZE - bytesInChunk;
      const take = Math.min(room, buf.length);
      await writeAsync(ws, buf.subarray(0, take));
      bytesInChunk += take;
      buf = buf.subarray(take);

      if (bytesInChunk === CHUNK_SIZE) {
        await endAsync(ws);
        chunkPaths.push(toPosix(`${rfile}.part${String(chunkIdx).padStart(5, "0")}`));
        chunkIdx++;
        bytesInChunk = 0;
        if (buf.length > 0 || !done) {
          ws = fs.createWriteStream(chunkPath(chunkIdx));
        }
      }
    }
  }

  if (bytesInChunk > 0) {
    await endAsync(ws);
    chunkPaths.push(toPosix(`${rfile}.part${String(chunkIdx).padStart(5, "0")}`));
  } else {
    // total size was an exact multiple of CHUNK_SIZE: the stream we opened
    // for the "next" chunk is empty — close and remove it.
    await endAsync(ws);
    await fsp.rm(chunkPath(chunkIdx), { force: true });
  }

  // If it all fit in one chunk, that chunk *is* the whole file — rename it
  // to the original filename and report it as unsplit.
  if (chunkPaths.length === 1) {
    const singlePartAbs = path.join(outDir, chunkPaths[0]);
    const finalAbs = path.join(outDir, rfile);
    await fsp.rename(singlePartAbs, finalAbs);
    return { size: totalSize, chunkPaths: null };
  }

  return { size: totalSize, chunkPaths };
}

async function main() {
  console.log(`Listing files for ${opts.model} ...`);
  const allFiles = await listRepoFiles(opts.model);
  const files = allFiles.filter((f) => shouldKeep(f, opts.dtype));
  if (files.length === 0) {
    throw new Error(
      `No files matched dtype "${opts.dtype}" in onnx/. Check the repo's file list on the Hub and pick a valid --dtype (e.g. fp32, q4, q4f16, int8).`
    );
  }
  console.log(`Will fetch ${files.length} files:\n - ${files.join("\n - ")}\n`);

  await fsp.mkdir(opts.out, { recursive: true });

  const manifest = {
    model_id: opts.model,
    dtype: opts.dtype,
    chunk_size: CHUNK_SIZE,
    generated_at: new Date().toISOString(),
    files: {},
  };

  for (const rfile of files) {
    process.stdout.write(`Downloading ${rfile} ... `);
    const { size, chunkPaths } = await downloadAndChunk(opts.model, rfile, opts.out);
    if (chunkPaths) {
      console.log(`${(size / 1024 / 1024).toFixed(1)} MB -> split into ${chunkPaths.length} chunks`);
    } else {
      console.log(`${(size / 1024 / 1024).toFixed(1)} MB (kept whole, under ${opts.chunkMb}MB)`);
    }
    manifest.files[rfile] = { size, chunks: chunkPaths };
  }

  const manifestPath = path.join(opts.out, "manifest.json");
  await fsp.writeFile(manifestPath, JSON.stringify(manifest, null, 2));

  console.log(`\nDone. Output written to: ${opts.out}`);
  console.log(`Manifest: ${manifestPath}`);
  console.log(`\nUpload the entire "${opts.out}" folder to your CDN, preserving the folder structure`);
  console.log(`(the "onnx/" subfolder and all .partNNNNN files must stay alongside manifest.json).`);
  console.log(`No full unsplit copy of any large file was written to disk — only the chunks exist.`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});