export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

export interface Env {
  DB: D1Database;
  VECTOR_INDEX: VectorizeIndex;
  TENSOR_BUCKET: R2Bucket;
  CONFIG_CACHE: KVNamespace;
  DAG_QUEUE: Queue;
  SCAFFOLD_DO: DurableObjectNamespace;
  CF_ACCESS_CLIENT_ID: string;
  CF_ACCESS_CLIENT_SECRET: string;
  CLOUDFLARE_ACCOUNT_ID: string;
  R2_ACCESS_KEY_ID: string;
  R2_SECRET_ACCESS_KEY: string;
  R2_BUCKET_NAME?: string;
}

export interface CompletedPartRequest {
  part_number: number;
  etag: string;
}
