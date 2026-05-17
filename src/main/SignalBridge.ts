import http from "node:http";
import https from "node:https";
import { Buffer } from "node:buffer";
import type { LLMClient } from "./LLMClient";

import * as dotenv from "dotenv";
import { join } from "path";
dotenv.config({ path: join(__dirname, "../../.env") });

const SIGNAL_CLI_URL = process.env.SIGNAL_CLI_URL?.trim() || "http://localhost:8080";
const SIGNAL_ACCOUNT = process.env.SIGNAL_ACCOUNT?.trim() || "";
const SIGNAL_ALLOWED_SENDER = process.env.SIGNAL_ALLOWED_SENDER?.trim() || "";

const RECONNECT_INITIAL_MS = 1_000;
const RECONNECT_MAX_MS = 30_000;
const RECONNECT_FACTOR = 2;

type SignalDataMessage = {
  message?: string;
};

type SignalEnvelope = {
  source?: string;
  sourceNumber?: string;
  dataMessage?: SignalDataMessage;
};

type SignalEvent = {
  envelope?: SignalEnvelope;
  account?: string;
};

export class SignalBridge {
  private readonly client: LLMClient;
  private abortController: AbortController | null = null;
  private messageCounter = 0;
  private readonly signalMessageIds = new Set<string>();

  constructor(client: LLMClient) {
    this.client = client;
  }

  start(): void {
    if (!SIGNAL_ACCOUNT) {
      console.log("[SignalBridge] SIGNAL_ACCOUNT not set — Signal integration disabled");
      return;
    }
    if (this.abortController) return;

    // Reply to Signal whenever the LLM finishes a response to a Signal-originated message
    this.client.setResponseCallback((messageId, text) => {
      if (this.signalMessageIds.has(messageId)) {
        this.signalMessageIds.delete(messageId);
        this.sendToSignal(text).catch((err) => {
          console.error("[SignalBridge] failed to send reply to Signal:", err);
        });
      }
    });

    this.abortController = new AbortController();
    this.runLoop(this.abortController.signal).catch((err) => {
      console.error("[SignalBridge] fatal loop error:", err);
    });
    console.log(
      `[SignalBridge] started — connecting to ${SIGNAL_CLI_URL}, account ${SIGNAL_ACCOUNT}` +
        (SIGNAL_ALLOWED_SENDER ? `, filtering to sender ${SIGNAL_ALLOWED_SENDER}` : "")
    );
  }

  stop(): void {
    this.abortController?.abort();
    this.abortController = null;
  }

  private async runLoop(signal: AbortSignal): Promise<void> {
    let attempts = 0;
    while (!signal.aborted) {
      try {
        await this.connectAndStream(signal);
        if (signal.aborted) break;
        attempts++;
      } catch (err) {
        if (signal.aborted) break;
        console.error("[SignalBridge] stream error:", err);
        attempts++;
      }
      const raw = RECONNECT_INITIAL_MS * Math.pow(RECONNECT_FACTOR, attempts - 1);
      const delay = Math.min(raw, RECONNECT_MAX_MS) * (0.8 + Math.random() * 0.4);
      console.log(`[SignalBridge] reconnecting in ${Math.round(delay / 1000)}s…`);
      await sleep(delay, signal);
    }
  }

  private connectAndStream(abortSignal: AbortSignal): Promise<void> {
    return new Promise((resolve, reject) => {
      const base = normalizeBaseUrl(SIGNAL_CLI_URL);
      const url = new URL("/api/v1/events", base);
      if (SIGNAL_ACCOUNT) url.searchParams.set("account", SIGNAL_ACCOUNT);

      const transport = url.protocol === "https:" ? https : http;
      const req = transport.request(
        url,
        { method: "GET", headers: { Accept: "text/event-stream" } },
        (res) => {
          const status = res.statusCode ?? 0;
          if (status < 200 || status >= 300) {
            res.resume();
            reject(new Error(`signal-cli SSE returned HTTP ${status}`));
            return;
          }

          const onAbort = () => {
            res.destroy();
            req.destroy();
            resolve();
          };
          abortSignal.addEventListener("abort", onAbort, { once: true });

          let buffer = "";
          let pendingData = "";

          const flushEvent = () => {
            if (pendingData) {
              this.handleRawData(pendingData);
              pendingData = "";
            }
          };

          const processLine = (line: string) => {
            if (line === "") {
              flushEvent();
            } else if (line.startsWith("data:")) {
              const value = line.slice(5);
              const segment = value.startsWith(" ") ? value.slice(1) : value;
              pendingData = pendingData ? `${pendingData}\n${segment}` : segment;
            }
            // ignore event:/id:/comment lines — we only need data
          };

          res.on("data", (chunk: Buffer) => {
            buffer += chunk.toString("utf8");
            let nl = buffer.indexOf("\n");
            while (nl !== -1) {
              let line = buffer.slice(0, nl);
              buffer = buffer.slice(nl + 1);
              if (line.endsWith("\r")) line = line.slice(0, -1);
              processLine(line);
              nl = buffer.indexOf("\n");
            }
          });

          res.on("end", () => {
            abortSignal.removeEventListener("abort", onAbort);
            flushEvent();
            resolve();
          });

          res.on("error", (err) => {
            abortSignal.removeEventListener("abort", onAbort);
            reject(err);
          });
        }
      );

      req.on("error", reject);
      req.end();
    });
  }

  private handleRawData(raw: string): void {
    let event: SignalEvent;
    try {
      event = JSON.parse(raw);
    } catch {
      return;
    }

    const envelope = event.envelope;
    if (!envelope) return;

    const message = envelope.dataMessage?.message;
    if (!message?.trim()) return;

    const sender = envelope.sourceNumber ?? envelope.source ?? "";

    if (SIGNAL_ALLOWED_SENDER && !sameSender(sender, SIGNAL_ALLOWED_SENDER)) {
      return;
    }

    console.log(
      `[SignalBridge] message from ${sender || "unknown"}: "${message.slice(0, 80)}${message.length > 80 ? "…" : ""}"`
    );
    this.injectMessage(message.trim());
  }

  private injectMessage(text: string): void {
    const messageId = `signal-${Date.now()}-${++this.messageCounter}`;
    this.signalMessageIds.add(messageId);
    this.client.sendChatMessage({ message: text, messageId }).catch((err: unknown) => {
      this.signalMessageIds.delete(messageId);
      console.error("[SignalBridge] failed to forward message to LLM:", err);
    });
  }

  private async sendToSignal(text: string): Promise<void> {
    if (!SIGNAL_ACCOUNT || !SIGNAL_ALLOWED_SENDER) return;

    const id = `reply-${Date.now()}`;
    const body = JSON.stringify({
      jsonrpc: "2.0",
      method: "send",
      params: {
        account: SIGNAL_ACCOUNT,
        recipient: [SIGNAL_ALLOWED_SENDER],
        message: text,
      },
      id,
    });

    const base = normalizeBaseUrl(SIGNAL_CLI_URL);
    const url = new URL("/api/v1/rpc", base);
    const transport = url.protocol === "https:" ? https : http;

    await new Promise<void>((resolve, reject) => {
      const req = transport.request(
        url,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Content-Length": String(Buffer.byteLength(body)),
          },
        },
        (res) => {
          res.resume(); // drain response body
          res.on("end", () => {
            const status = res.statusCode ?? 0;
            if (status === 201 || (status >= 200 && status < 300)) {
              resolve();
            } else {
              reject(new Error(`signal-cli send returned HTTP ${status}`));
            }
          });
        }
      );
      req.on("error", reject);
      req.write(body);
      req.end();
    });

    console.log(`[SignalBridge] reply sent to ${SIGNAL_ALLOWED_SENDER}`);
  }
}

function normalizeBaseUrl(url: string): string {
  const trimmed = url.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, "");
  return `http://${trimmed}`.replace(/\/+$/, "");
}

function sameSender(a: string, b: string): boolean {
  const norm = (s: string) => s.replace(/[\s\-()]/g, "").toLowerCase();
  return norm(a) === norm(b);
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal?.aborted) {
      resolve();
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      resolve();
    }, { once: true });
  });
}
