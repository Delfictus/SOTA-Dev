# Autonomous Infrastructure Status

Generated: `2026-05-25T05:47:56.333427+00:00`

Overall status: `OK`

## Static Cloudflare Bindings

```json
{
  "status": "FOUND",
  "path": "/home/diddy/Desktop/Prism4D-bio/cloud/prism-manifold-worker/wrangler.toml",
  "worker_name": "prism-manifold-worker",
  "account_id": null,
  "d1_databases": [
    {
      "binding": "DB",
      "database_name": "prism_metadata",
      "database_id": "ef63a188-0634-4f68-bd6d-1b4cc1232b06"
    }
  ],
  "vectorize": [
    {
      "binding": "VECTOR_INDEX",
      "index_name": "dkl_latent_space"
    }
  ],
  "r2_buckets": [
    {
      "binding": "TENSOR_BUCKET",
      "bucket_name": "prism-tensors"
    }
  ],
  "kv_namespaces": [
    {
      "binding": "CONFIG_CACHE",
      "id": "342b9ddad269400085325884befca1f5"
    }
  ],
  "queues": {
    "producers": [
      {
        "binding": "DAG_QUEUE",
        "queue": "prism-dag-queue"
      }
    ]
  },
  "durable_objects": {
    "bindings": [
      {
        "name": "SCAFFOLD_DO",
        "class_name": "ScaffoldState"
      }
    ]
  }
}
```

## Live Checks

| check | status | return code |
|---|---|---:|
| worker_http | `ACCESS_PROTECTED` | 0 |
| d1_candidate_count | `OK` | 0 |
| vectorize_info | `OK` | 0 |
| r2_bucket_list | `OK` | 0 |
| queue_list | `OK` | 0 |

No redeploy was performed. AUTH_BLOCKED means the local terminal lacks usable wrangler credentials; the binding still exists in source configuration.
