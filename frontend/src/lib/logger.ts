/**
 * Frontend logging utility.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Explicit error handling
 * - No silent failures
 *
 * Batches logs and forwards to backend every 5 seconds.
 */

type LogLevel = "info" | "warn" | "error";

interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  context?: Record<string, unknown>;
}

class Logger {
  private buffer: LogEntry[] = [];
  private flushInterval: number = 5000; // 5 seconds
  private maxBufferSize: number = 100;
  private intervalId: number | null = null;

  constructor() {
    // Start periodic flush
    this.startPeriodicFlush();

    // Flush on page unload
    if (typeof window !== "undefined") {
      window.addEventListener("beforeunload", () => this.flush());
    }
  }

  private startPeriodicFlush() {
    this.intervalId = window.setInterval(() => {
      this.flush();
    }, this.flushInterval);
  }

  private createEntry(
    level: LogLevel,
    message: string,
    context?: Record<string, unknown>
  ): LogEntry {
    return {
      timestamp: new Date().toISOString(),
      level,
      message,
      context,
    };
  }

  /**
   * Log info message
   */
  info(message: string, context?: Record<string, unknown>) {
    const entry = this.createEntry("info", message, context);
    this.buffer.push(entry);
    console.log(`[INFO] ${message}`, context || "");

    if (this.buffer.length >= this.maxBufferSize) {
      this.flush();
    }
  }

  /**
   * Log warning message
   */
  warn(message: string, context?: Record<string, unknown>) {
    const entry = this.createEntry("warn", message, context);
    this.buffer.push(entry);
    console.warn(`[WARN] ${message}`, context || "");

    if (this.buffer.length >= this.maxBufferSize) {
      this.flush();
    }
  }

  /**
   * Log error message
   */
  error(message: string, context?: Record<string, unknown>) {
    const entry = this.createEntry("error", message, context);
    this.buffer.push(entry);
    console.error(`[ERROR] ${message}`, context || "");

    // Force flush on errors
    this.flush();
  }

  /**
   * Flush buffered logs to backend
   */
  private async flush() {
    if (this.buffer.length === 0) {
      return;
    }

    const logsToSend = [...this.buffer];
    this.buffer = [];

    try {
      const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

      await fetch(`${API_BASE_URL}/api/logs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ logs: logsToSend }),
      });

      console.debug(`[Logger] Sent ${logsToSend.length} logs to backend`);
    } catch (error) {
      // Re-add failed logs to buffer (but limit to prevent infinite growth)
      this.buffer = [...logsToSend.slice(-50), ...this.buffer];
      console.error("Failed to send logs to backend:", error);
    }
  }

  /**
   * Cleanup - stop periodic flush
   */
  destroy() {
    if (this.intervalId !== null) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    this.flush();
  }
}

// Singleton instance
export const logger = new Logger();
