import { HttpError, jsonResponse, readJsonObject, requireInteger, requireJsonObject } from "../http";
import type { Env, JsonObject } from "../types";

type ScaffoldRecord = {
  scaffold_id: string;
  scaffold_version: number;
  updated_at: string;
  data: JsonObject;
};

export class ScaffoldState implements DurableObject {
  private readonly state: DurableObjectState;

  constructor(state: DurableObjectState, _env: Env) {
    this.state = state;
  }

  async fetch(request: Request): Promise<Response> {
    if (request.method !== "POST") {
      return jsonResponse({ error: "method_not_allowed" }, 405);
    }

    try {
      const body = await readJsonObject(request);
      const expectedVersion = requireInteger(body, "expected_version");
      const data = requireJsonObject(body, "data");
      const scaffoldId = typeof body.scaffold_id === "string" ? body.scaffold_id : "unknown_scaffold";
      const currentVersion = (await this.state.storage.get<number>("scaffold_version")) ?? 0;

      if (expectedVersion !== currentVersion) {
        return jsonResponse(
          {
            error: "version_conflict",
            current_version: currentVersion,
            expected_version: expectedVersion
          },
          409
        );
      }

      const nextVersion = currentVersion + 1;
      const record: ScaffoldRecord = {
        scaffold_id: scaffoldId,
        scaffold_version: nextVersion,
        updated_at: new Date().toISOString(),
        data
      };
      await this.state.storage.put("scaffold_version", nextVersion);
      await this.state.storage.put("scaffold_metadata", record);

      return jsonResponse({
        ok: true,
        scaffold_id: scaffoldId,
        scaffold_version: nextVersion,
        previous_version: currentVersion
      });
    } catch (error: unknown) {
      if (error instanceof HttpError) {
        return jsonResponse({ error: error.message }, error.status);
      }
      return jsonResponse({ error: "durable_object_update_failed" }, 500);
    }
  }
}
