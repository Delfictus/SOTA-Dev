import {
  CompleteMultipartUploadCommand,
  CreateMultipartUploadCommand,
  S3Client,
  UploadPartCommand
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

import { ScaffoldState } from "./durable_objects/ScaffoldState";
import {
  HttpError,
  jsonResponse,
  optionalInteger,
  optionalString,
  readJsonObject,
  requireInteger,
  requireJsonObject,
  requireJsonObjectArray,
  requireNumberArray,
  requireString
} from "./http";
import type { CompletedPartRequest, Env, JsonObject, JsonValue } from "./types";

export { ScaffoldState };

const DEFAULT_BUCKET_NAME = "prism-tensors";
const DEFAULT_PRESIGN_EXPIRY_SECONDS = 900;
const MAX_PRESIGN_EXPIRY_SECONDS = 3600;
const MAX_VECTOR_K = 100;
const LOCAL_MPU_PREFIX = "__local_mpu";

function authenticate(request: Request, env: Env): Response | null {
  const clientId = request.headers.get("CF-Access-Client-Id");
  const clientSecret = request.headers.get("CF-Access-Client-Secret");
  if (
    clientId === null ||
    clientSecret === null ||
    clientId !== env.CF_ACCESS_CLIENT_ID ||
    clientSecret !== env.CF_ACCESS_CLIENT_SECRET
  ) {
    return jsonResponse({ error: "forbidden" }, 403);
  }
  return null;
}

function r2BucketName(env: Env): string {
  return env.R2_BUCKET_NAME ?? DEFAULT_BUCKET_NAME;
}

function s3Client(env: Env): S3Client {
  return new S3Client({
    region: "auto",
    endpoint: `https://${env.CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com`,
    credentials: {
      accessKeyId: env.R2_ACCESS_KEY_ID,
      secretAccessKey: env.R2_SECRET_ACCESS_KEY
    },
    forcePathStyle: true
  });
}

function s3Metadata(metadata: JsonObject | undefined): Record<string, string> | undefined {
  if (metadata === undefined) {
    return undefined;
  }
  const converted: Record<string, string> = {};
  for (const [key, value] of Object.entries(metadata)) {
    converted[key] = typeof value === "string" ? value : JSON.stringify(value);
  }
  return converted;
}

function useLocalMultipart(env: Env): boolean {
  return env.CLOUDFLARE_ACCOUNT_ID === "dev" || env.R2_ACCESS_KEY_ID === "dev";
}

function localMpuMetadataKey(uploadId: string): string {
  return `${LOCAL_MPU_PREFIX}/${uploadId}/metadata.json`;
}

function localMpuPartKey(uploadId: string, partNumber: number): string {
  return `${LOCAL_MPU_PREFIX}/${uploadId}/part-${partNumber.toString().padStart(5, "0")}`;
}

function localMpuToken(uploadId: string, partNumber: number): string {
  return `${uploadId}.${partNumber}`;
}

async function readLocalMpuMetadata(env: Env, uploadId: string): Promise<JsonObject> {
  const object = await env.TENSOR_BUCKET.get(localMpuMetadataKey(uploadId));
  if (object === null) {
    throw new HttpError(404, "local_multipart_upload_not_found");
  }
  const parsed = (await object.json()) as unknown;
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new HttpError(500, "local_multipart_metadata_corrupt");
  }
  return parsed as JsonObject;
}

function methodGuard(request: Request, method: string): void {
  if (request.method !== method) {
    throw new HttpError(405, "method_not_allowed");
  }
}

function completeParts(parts: JsonObject[]): CompletedPartRequest[] {
  return parts
    .map((part) => ({
      part_number: requireInteger(part, "part_number"),
      etag: requireString(part, "etag")
    }))
    .sort((left, right) => left.part_number - right.part_number);
}

async function createMultipartUpload(request: Request, env: Env): Promise<Response> {
  methodGuard(request, "POST");
  const body = await readJsonObject(request);
  const key = requireString(body, "key");
  const contentType = optionalString(body, "content_type") ?? "application/octet-stream";
  const metadata = body.metadata === undefined ? undefined : requireJsonObject(body, "metadata");
  if (useLocalMultipart(env)) {
    const uploadId = crypto.randomUUID();
    await env.TENSOR_BUCKET.put(
      localMpuMetadataKey(uploadId),
      JSON.stringify({
        key,
        content_type: contentType,
        metadata: metadata ?? {},
        created_at: new Date().toISOString(),
        mode: "local_miniflare_mpu"
      }),
      {
        httpMetadata: { contentType: "application/json" }
      }
    );
    return jsonResponse({
      key,
      upload_id: uploadId,
      bucket: r2BucketName(env),
      mode: "local_miniflare_mpu"
    });
  }
  const command = new CreateMultipartUploadCommand({
    Bucket: r2BucketName(env),
    Key: key,
    ContentType: contentType,
    Metadata: s3Metadata(metadata)
  });
  const result = await s3Client(env).send(command);
  if (result.UploadId === undefined) {
    throw new HttpError(502, "r2_create_multipart_upload_returned_no_upload_id");
  }
  return jsonResponse({
    key,
    upload_id: result.UploadId,
    bucket: r2BucketName(env)
  });
}

async function signMultipartPart(request: Request, env: Env): Promise<Response> {
  methodGuard(request, "POST");
  const body = await readJsonObject(request);
  const key = requireString(body, "key");
  const uploadId = requireString(body, "upload_id");
  const partNumber = requireInteger(body, "part_number");
  const requestedExpiry = optionalInteger(body, "expires_in_seconds", DEFAULT_PRESIGN_EXPIRY_SECONDS);
  const expiresIn = Math.min(Math.max(requestedExpiry, 60), MAX_PRESIGN_EXPIRY_SECONDS);
  if (useLocalMultipart(env)) {
    await readLocalMpuMetadata(env, uploadId);
    const url = new URL(request.url);
    url.pathname = "/api/v1/tensors/multipart/local-part";
    url.search = "";
    url.searchParams.set("key", key);
    url.searchParams.set("upload_id", uploadId);
    url.searchParams.set("part_number", partNumber.toString());
    url.searchParams.set("token", localMpuToken(uploadId, partNumber));
    return jsonResponse({
      key,
      upload_id: uploadId,
      part_number: partNumber,
      expires_in_seconds: expiresIn,
      url: url.toString(),
      mode: "local_miniflare_mpu"
    });
  }
  const command = new UploadPartCommand({
    Bucket: r2BucketName(env),
    Key: key,
    UploadId: uploadId,
    PartNumber: partNumber
  });
  const url = await getSignedUrl(s3Client(env), command, { expiresIn });
  return jsonResponse({
    key,
    upload_id: uploadId,
    part_number: partNumber,
    expires_in_seconds: expiresIn,
    url
  });
}

async function completeMultipartUpload(request: Request, env: Env): Promise<Response> {
  methodGuard(request, "POST");
  const body = await readJsonObject(request);
  const key = requireString(body, "key");
  const uploadId = requireString(body, "upload_id");
  const parts = completeParts(requireJsonObjectArray(body, "parts"));
  if (parts.length === 0) {
    throw new HttpError(400, "parts must not be empty");
  }
  if (useLocalMultipart(env)) {
    const metadata = await readLocalMpuMetadata(env, uploadId);
    const expectedKey = requireString(metadata, "key");
    if (expectedKey !== key) {
      throw new HttpError(409, "local_multipart_key_mismatch");
    }
    const buffers: ArrayBuffer[] = [];
    for (const part of parts) {
      const object = await env.TENSOR_BUCKET.get(localMpuPartKey(uploadId, part.part_number));
      if (object === null) {
        throw new HttpError(400, `missing local multipart part ${part.part_number}`);
      }
      if (object.etag !== part.etag) {
        throw new HttpError(409, `etag mismatch for local multipart part ${part.part_number}`);
      }
      buffers.push(await object.arrayBuffer());
    }
    const contentType = optionalString(metadata, "content_type") ?? "application/octet-stream";
    const customMetadata = s3Metadata(requireJsonObject(metadata, "metadata"));
    const finalObject = await env.TENSOR_BUCKET.put(key, new Blob(buffers), {
      httpMetadata: { contentType },
      customMetadata
    });
    return jsonResponse({
      key,
      upload_id: uploadId,
      bucket: r2BucketName(env),
      part_count: parts.length,
      location: null,
      etag: finalObject.etag,
      version_id: null,
      mode: "local_miniflare_mpu"
    });
  }
  const result = await s3Client(env).send(
    new CompleteMultipartUploadCommand({
      Bucket: r2BucketName(env),
      Key: key,
      UploadId: uploadId,
      MultipartUpload: {
        Parts: parts.map((part) => ({
          ETag: part.etag,
          PartNumber: part.part_number
        }))
      }
    })
  );
  return jsonResponse({
    key,
    upload_id: uploadId,
    bucket: r2BucketName(env),
    part_count: parts.length,
    location: result.Location ?? null,
    etag: result.ETag ?? null,
    version_id: result.VersionId ?? null
  });
}

async function localMultipartPart(request: Request, env: Env): Promise<Response> {
  methodGuard(request, "PUT");
  if (!useLocalMultipart(env)) {
    throw new HttpError(404, "not_found");
  }
  const url = new URL(request.url);
  const uploadId = url.searchParams.get("upload_id") ?? "";
  const partNumberText = url.searchParams.get("part_number") ?? "";
  const token = url.searchParams.get("token") ?? "";
  const partNumber = Number(partNumberText);
  if (uploadId.length === 0 || !Number.isInteger(partNumber) || partNumber < 1) {
    throw new HttpError(400, "invalid local multipart part URL");
  }
  if (token !== localMpuToken(uploadId, partNumber)) {
    throw new HttpError(403, "invalid local multipart token");
  }
  await readLocalMpuMetadata(env, uploadId);
  const object = await env.TENSOR_BUCKET.put(localMpuPartKey(uploadId, partNumber), request.body);
  if (object === null) {
    throw new HttpError(500, "local_multipart_part_write_failed");
  }
  return new Response(null, {
    status: 200,
    headers: {
      ETag: object.etag
    }
  });
}

async function scaffoldUpdate(request: Request, env: Env): Promise<Response> {
  methodGuard(request, "POST");
  const body = await readJsonObject(request);
  const scaffoldId = requireString(body, "scaffold_id");
  requireInteger(body, "expected_version");
  requireJsonObject(body, "data");
  const durableObjectId = env.SCAFFOLD_DO.idFromName(scaffoldId);
  const stub = env.SCAFFOLD_DO.get(durableObjectId);
  return stub.fetch(
    new Request(request.url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body)
    })
  );
}

async function vectorizeQuery(request: Request, env: Env): Promise<Response> {
  methodGuard(request, "POST");
  const body = await readJsonObject(request);
  const vector = requireNumberArray(body, "latent_vector");
  const k = Math.min(Math.max(optionalInteger(body, "k", 10), 1), MAX_VECTOR_K);
  const result = (await env.VECTOR_INDEX.query(vector, {
    topK: k,
    returnMetadata: true,
    returnValues: true
  })) as unknown as JsonValue;
  return jsonResponse({
    k,
    result
  });
}

async function route(request: Request, env: Env): Promise<Response> {
  const pathname = new URL(request.url).pathname;
  if (pathname === "/api/v1/tensors/multipart/local-part") {
    return localMultipartPart(request, env);
  }

  const authFailure = authenticate(request, env);
  if (authFailure !== null) {
    return authFailure;
  }

  switch (pathname) {
    case "/api/v1/tensors/multipart/create":
      return createMultipartUpload(request, env);
    case "/api/v1/tensors/multipart/sign-part":
      return signMultipartPart(request, env);
    case "/api/v1/tensors/multipart/complete":
      return completeMultipartUpload(request, env);
    case "/api/v1/scaffold/update":
      return scaffoldUpdate(request, env);
    case "/api/v1/vectorize/query":
      return vectorizeQuery(request, env);
    default:
      return jsonResponse({ error: "not_found", path: pathname }, 404);
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    try {
      return await route(request, env);
    } catch (error: unknown) {
      if (error instanceof HttpError) {
        return jsonResponse({ error: error.message }, error.status);
      }
      return jsonResponse({ error: "internal_error" }, 500);
    }
  }
};
