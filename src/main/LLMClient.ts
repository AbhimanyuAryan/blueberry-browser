import { WebContents } from "electron";
import { execSync } from "child_process";
import { streamText, type LanguageModel, type CoreMessage } from "ai";
import { openai } from "@ai-sdk/openai";
import { anthropic } from "@ai-sdk/anthropic";
import { CopilotClient, approveAll } from "@github/copilot-sdk";
import type { CopilotSession } from "@github/copilot-sdk";
import * as dotenv from "dotenv";
import { join } from "path";
import type { Window } from "./Window";

// Load environment variables from .env file
dotenv.config({ path: join(__dirname, "../../.env") });

interface ChatRequest {
  message: string;
  messageId: string;
}

interface StreamChunk {
  content: string;
  isComplete: boolean;
}

type LLMProvider = "openai" | "anthropic" | "github";

const DEFAULT_MODELS: Record<LLMProvider, string | undefined> = {
  openai: "gpt-4o-mini",
  anthropic: "claude-3-5-sonnet-20241022",
  github: undefined, // let Copilot use its default model
};

const MAX_CONTEXT_LENGTH = 4000;
const DEFAULT_TEMPERATURE = 0.7;

export class LLMClient {
  private readonly webContents: WebContents;
  private window: Window | null = null;
  private readonly provider: LLMProvider;
  private readonly modelName: string | undefined;
  private readonly model: LanguageModel | null;
  private copilotClient: CopilotClient | null = null;
  private copilotSession: CopilotSession | null = null;
  private messages: CoreMessage[] = [];
  private onAssistantResponse?: (messageId: string, text: string) => void;

  constructor(webContents: WebContents) {
    this.webContents = webContents;
    this.provider = this.getProvider();
    this.modelName = this.getModelName();
    this.model = this.initializeModel();
    if (this.provider === "github") {
      this.copilotClient = this.initializeCopilotClient();
    }

    this.logInitializationStatus();
  }

  setWindow(window: Window): void {
    this.window = window;
  }

  setResponseCallback(cb: (messageId: string, text: string) => void): void {
    this.onAssistantResponse = cb;
  }

  private getProvider(): LLMProvider {
    const provider = process.env.LLM_PROVIDER?.toLowerCase();
    if (provider === "anthropic") return "anthropic";
    if (provider === "github") return "github";
    return "openai";
  }

  private getModelName(): string | undefined {
    return process.env.LLM_MODEL || DEFAULT_MODELS[this.provider];
  }

  private initializeModel(): LanguageModel | null {
    if (this.provider === "github") return null; // uses CopilotClient instead
    const apiKey = this.getApiKey();
    if (!apiKey) return null;

    const model = this.modelName!;
    switch (this.provider) {
      case "anthropic":
        return anthropic(model);
      case "openai":
        return openai(model);
      default:
        return null;
    }
  }

  private initializeCopilotClient(): CopilotClient {
    const token = this.resolveGitHubToken();
    return new CopilotClient(token ? { gitHubToken: token } : undefined);
  }

  private resolveGitHubToken(): string | undefined {
    if (process.env.GITHUB_TOKEN) return process.env.GITHUB_TOKEN;
    try {
      return execSync("gh auth token", { encoding: "utf-8" }).trim();
    } catch {
      return undefined;
    }
  }

  private getApiKey(): string | undefined {
    switch (this.provider) {
      case "anthropic":
        return process.env.ANTHROPIC_API_KEY;
      case "openai":
        return process.env.OPENAI_API_KEY;
      default:
        return undefined;
    }
  }

  private logInitializationStatus(): void {
    const isReady = this.provider === "github" ? this.copilotClient !== null : this.model !== null;
    if (isReady) {
      console.log(
        `✅ LLM Client initialized with ${this.provider} provider using model: ${this.modelName}`
      );
    } else {
      const keyNames: Record<LLMProvider, string> = {
        anthropic: "ANTHROPIC_API_KEY",
        openai: "OPENAI_API_KEY",
        github: "GITHUB_TOKEN",
      };
      console.error(
        `❌ LLM Client initialization failed: ${keyNames[this.provider]} not found in environment variables.\n` +
          `Please add your API key to the .env file in the project root.`
      );
    }
  }

  async sendChatMessage(request: ChatRequest): Promise<void> {
    try {
      // Get screenshot from active tab if available
      let screenshot: string | null = null;
      if (this.window) {
        const activeTab = this.window.activeTab;
        if (activeTab) {
          try {
            const image = await activeTab.screenshot();
            screenshot = image.toDataURL();
          } catch (error) {
            console.error("Failed to capture screenshot:", error);
          }
        }
      }

      // Build user message content with screenshot first, then text
      const userContent: any[] = [];
      
      // Add screenshot as the first part if available
      if (screenshot) {
        userContent.push({
          type: "image",
          image: screenshot,
        });
      }
      
      // Add text content
      userContent.push({
        type: "text",
        text: request.message,
      });

      // Create user message in CoreMessage format
      const userMessage: CoreMessage = {
        role: "user",
        content: userContent.length === 1 ? request.message : userContent,
      };
      
      this.messages.push(userMessage);

      // Send updated messages to renderer
      this.sendMessagesToRenderer();

      const isReady = this.provider === "github" ? this.copilotClient !== null : this.model !== null;
      if (!isReady) {
        this.sendErrorMessage(
          request.messageId,
          "LLM service is not configured. Please add your API key to the .env file."
        );
        return;
      }

      if (this.provider === "github") {
        await this.sendCopilotMessage(request);
      } else {
        const messages = await this.prepareMessagesWithContext(request);
        await this.streamResponse(messages, request.messageId);
      }
    } catch (error) {
      console.error("Error in LLM request:", error);
      this.handleStreamError(error, request.messageId);
    }
  }

  private async sendCopilotMessage(request: ChatRequest): Promise<void> {
    const client = this.copilotClient!;

    if (client.getState() !== "connected") {
      await client.start();
    }

    // Create a new session on the first message of a conversation, capturing
    // current page context as the system prompt so it stays consistent for
    // the whole conversation thread.
    if (!this.copilotSession) {
      let pageUrl: string | null = null;
      let pageText: string | null = null;
      if (this.window) {
        const activeTab = this.window.activeTab;
        if (activeTab) {
          pageUrl = activeTab.url;
          try {
            pageText = await activeTab.getTabText();
          } catch {
            // non-fatal
          }
        }
      }

      this.copilotSession = await client.createSession({
        ...(this.modelName ? { model: this.modelName } : {}),
        streaming: true,
        onPermissionRequest: approveAll,
        systemMessage: { mode: "replace", content: this.buildSystemPrompt(pageUrl, pageText) },
      });
    }

    const session = this.copilotSession;
    const messageIndex = this.messages.length;
    this.messages.push({ role: "assistant", content: "" });

    await new Promise<void>((resolve, reject) => {
      let accumulatedText = "";

      const unsubDelta = session.on("assistant.message_delta", (event: any) => {
        const delta: string = event.data.deltaContent ?? "";
        accumulatedText += delta;
        this.messages[messageIndex] = { role: "assistant", content: accumulatedText };
        this.sendMessagesToRenderer();
        this.sendStreamChunk(request.messageId, { content: delta, isComplete: false });
      });

      const unsubIdle = session.on("session.idle", () => {
        unsubDelta();
        unsubIdle();
        this.messages[messageIndex] = { role: "assistant", content: accumulatedText };
        this.sendMessagesToRenderer();
        this.sendStreamChunk(request.messageId, { content: accumulatedText, isComplete: true });
        this.onAssistantResponse?.(request.messageId, accumulatedText);
        resolve();
      });

      session.on("session.error", () => {
        unsubDelta();
        unsubIdle();
        reject(new Error("Copilot session encountered an error"));
      });

      session.send({ prompt: request.message });
    });
  }

  clearMessages(): void {
    this.messages = [];
    if (this.copilotSession) {
      this.copilotSession.disconnect().catch(() => {});
      this.copilotSession = null;
    }
    this.sendMessagesToRenderer();
  }

  getMessages(): CoreMessage[] {
    return this.messages;
  }

  private sendMessagesToRenderer(): void {
    this.webContents.send("chat-messages-updated", this.messages);
  }

  private async prepareMessagesWithContext(_request: ChatRequest): Promise<CoreMessage[]> {
    // Get page context from active tab
    let pageUrl: string | null = null;
    let pageText: string | null = null;
    
    if (this.window) {
      const activeTab = this.window.activeTab;
      if (activeTab) {
        pageUrl = activeTab.url;
        try {
          pageText = await activeTab.getTabText();
        } catch (error) {
          console.error("Failed to get page text:", error);
        }
      }
    }

    // Build system message
    const systemMessage: CoreMessage = {
      role: "system",
      content: this.buildSystemPrompt(pageUrl, pageText),
    };

    // Include all messages in history (system + conversation)
    return [systemMessage, ...this.messages];
  }

  private buildSystemPrompt(url: string | null, pageText: string | null): string {
    const parts: string[] = [
      "You are a helpful AI assistant integrated into a web browser.",
      "You can analyze and discuss web pages with the user.",
      "The user's messages may include screenshots of the current page as the first image.",
    ];

    if (url) {
      parts.push(`\nCurrent page URL: ${url}`);
    }

    if (pageText) {
      const truncatedText = this.truncateText(pageText, MAX_CONTEXT_LENGTH);
      parts.push(`\nPage content (text):\n${truncatedText}`);
    }

    parts.push(
      "\nPlease provide helpful, accurate, and contextual responses about the current webpage.",
      "If the user asks about specific content, refer to the page content and/or screenshot provided."
    );

    return parts.join("\n");
  }

  private truncateText(text: string, maxLength: number): string {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + "...";
  }

  private async streamResponse(
    messages: CoreMessage[],
    messageId: string
  ): Promise<void> {
    if (!this.model) {
      throw new Error("Model not initialized");
    }

    try {
      const result = await streamText({
        model: this.model,
        messages,
        temperature: DEFAULT_TEMPERATURE,
        maxRetries: 3,
        abortSignal: undefined, // Could add abort controller for cancellation
      });

      await this.processStream(result.textStream, messageId);
    } catch (error) {
      throw error; // Re-throw to be handled by the caller
    }
  }

  private async processStream(
    textStream: AsyncIterable<string>,
    messageId: string
  ): Promise<void> {
    let accumulatedText = "";

    // Create a placeholder assistant message
    const assistantMessage: CoreMessage = {
      role: "assistant",
      content: "",
    };
    
    // Keep track of the index for updates
    const messageIndex = this.messages.length;
    this.messages.push(assistantMessage);

    for await (const chunk of textStream) {
      accumulatedText += chunk;

      // Update assistant message content
      this.messages[messageIndex] = {
        role: "assistant",
        content: accumulatedText,
      };
      this.sendMessagesToRenderer();

      this.sendStreamChunk(messageId, {
        content: chunk,
        isComplete: false,
      });
    }

    // Final update with complete content
    this.messages[messageIndex] = {
      role: "assistant",
      content: accumulatedText,
    };
    this.sendMessagesToRenderer();

    // Send the final complete signal
    this.sendStreamChunk(messageId, {
      content: accumulatedText,
      isComplete: true,
    });

    this.onAssistantResponse?.(messageId, accumulatedText);
  }

  private handleStreamError(error: unknown, messageId: string): void {
    console.error("Error streaming from LLM:", error);

    const errorMessage = this.getErrorMessage(error);
    this.sendErrorMessage(messageId, errorMessage);
  }

  private getErrorMessage(error: unknown): string {
    if (!(error instanceof Error)) {
      return "An unexpected error occurred. Please try again.";
    }

    const message = error.message.toLowerCase();

    if (message.includes("401") || message.includes("unauthorized")) {
      return "Authentication error: Please check your API key in the .env file.";
    }

    if (message.includes("429") || message.includes("rate limit")) {
      return "Rate limit exceeded. Please try again in a few moments.";
    }

    if (
      message.includes("network") ||
      message.includes("fetch") ||
      message.includes("econnrefused")
    ) {
      return "Network error: Please check your internet connection.";
    }

    if (message.includes("timeout")) {
      return "Request timeout: The service took too long to respond. Please try again.";
    }

    return "Sorry, I encountered an error while processing your request. Please try again.";
  }

  private sendErrorMessage(messageId: string, errorMessage: string): void {
    this.sendStreamChunk(messageId, {
      content: errorMessage,
      isComplete: true,
    });
  }

  private sendStreamChunk(messageId: string, chunk: StreamChunk): void {
    this.webContents.send("chat-response", {
      messageId,
      content: chunk.content,
      isComplete: chunk.isComplete,
    });
  }
}
