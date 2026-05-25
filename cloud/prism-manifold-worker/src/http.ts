import type { JsonObject, JsonValue } from "./types";

export class HttpError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function jsonResponse(body: JsonValue, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store"
    }
  });
}

export async function readJsonObject(request: Request): Promise<JsonObject> {
  const parsed = (await request.json()) as unknown;
  if (!isJsonObject(parsed)) {
    throw new HttpError(400, "request body must be a JSON object");
  }
  return parsed;
}

export function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function requireString(value: JsonObject, key: string): string {
  const item = value[key];
  if (typeof item !== "string" || item.length === 0) {
    throw new HttpError(400, `${key} must be a non-empty string`);
  }
  return item;
}

export function optionalString(value: JsonObject, key: string): string | undefined {
  const item = value[key];
  if (item === undefined) {
    return undefined;
  }
  if (typeof item !== "string" || item.length === 0) {
    throw new HttpError(400, `${key} must be a non-empty string when provided`);
  }
  return item;
}

export function requireInteger(value: JsonObject, key: string): number {
  const item = value[key];
  if (typeof item !== "number" || !Number.isInteger(item)) {
    throw new HttpError(400, `${key} must be an integer`);
  }
  return item;
}

export function optionalInteger(value: JsonObject, key: string, fallback: number): number {
  const item = value[key];
  if (item === undefined) {
    return fallback;
  }
  if (typeof item !== "number" || !Number.isInteger(item)) {
    throw new HttpError(400, `${key} must be an integer when provided`);
  }
  return item;
}

export function requireNumberArray(value: JsonObject, key: string): number[] {
  const item: unknown = value[key];
  if (!Array.isArray(item) || !item.every((entry) => typeof entry === "number" && Number.isFinite(entry))) {
    throw new HttpError(400, `${key} must be an array of finite numbers`);
  }
  return item as number[];
}

export function requireJsonObject(value: JsonObject, key: string): JsonObject {
  const item = value[key];
  if (!isJsonObject(item)) {
    throw new HttpError(400, `${key} must be a JSON object`);
  }
  return item;
}

export function requireJsonObjectArray(value: JsonObject, key: string): JsonObject[] {
  const item = value[key];
  if (!Array.isArray(item) || !item.every(isJsonObject)) {
    throw new HttpError(400, `${key} must be an array of JSON objects`);
  }
  return item;
}
