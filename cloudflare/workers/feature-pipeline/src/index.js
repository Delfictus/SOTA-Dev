/**
 * PRISM-4D Feature Pipeline Worker — v4 feature-service hardening contract.
 *
 * Pinned against:
 *   cloudflare/d1/schema_phase4_site_tags.sql
 *   scripts/training/v4_feature_contract.yaml   (feature contract v4.1)
 *   docs/contracts/event_schema_v1.yaml
 *   docs/contracts/site_tags_json_v1.schema.json
 *   docs/contracts/persistence_contract.md      (persistence BLOCKED)
 *
 * HARD RULES ENFORCED IN CODE:
 *   1. No INSERT OR REPLACE on site_features anywhere. Only
 *      INSERT OR IGNORE + column-scoped UPDATE.
 *   2. No parent-object-as-scalar shortcuts. Dict-valued source fields
 *      (cold_phase_fraction, kcc, signal_preservation) are flattened.
 *   3. No silent coercion of absent values to 0. NULL = absent; 0.0 = zero.
 *   4. Writer ownership:
 *      - W1 (queue consumer)            → engine-sourced site_features columns
 *                                         + site_lining_residues + site_kcc_candidates
 *      - W2 (POST /site-features/:t/dcc) → min_dist_to_ligand, graded_score,
 *                                          dcc_metric_source + corrected_dcc
 *      - W3a (POST /site-features/:t/temporal) → phase_transition_ratio,
 *                                          warm_hold_spike_fraction
 *      - W3b (POST /site-features/:t/event-aggregates) → site_event_aggregates
 *      - W4 (POST /targets/:t/runtime)  → targets.engine_* columns
 *      - W5 (POST /site-features/:t/persistence) → RESERVED. BLOCKED.
 *   5. therm_class and classification closed enums — unknown values
 *      bucket to OTHER and log WARN (caveat 1 of implementation authorization).
 */

export { CampaignTracker } from './campaign_tracker.js';

const FEATURE_CONTRACT_VERSION = "v4.1";
const EVENT_CONTRACT_VERSION   = "event_schema_v1";

// Closed enums (per v4_feature_contract.yaml)
const CLASSIFICATION_ENUM = new Set([
    "ActiveSite", "AllostericSite", "SurfaceSite", "CrypticSite", "TransientSite"
]);
const THERM_CLASS_ENUM = new Set([
    "DYNAMIC", "CRYPTIC", "STATIC", "TRANSIENT"
]);

// Keys permitted inside site_tags_json (schema: site_tags_json_v1.schema.json).
// Any other key produced by the engine is dropped at the W1 boundary.
const SITE_TAGS_JSON_ALLOWED_KEYS = new Set([
    "mean_burial", "asymmetry_offset",
    "sti_n_spikes", "sti_n_voxels",
    "composite_v3_score", "composite_audit_score",
    "composite_v3_rank", "composite_audit_rank", "cryptic_rank",
    "ranker_version", "tokenized_token",
    "tide_trigger_residues",
]);
const SITE_TAGS_JSON_MAX_BYTES = 2048;

// ─────────────────────────────────────────────────────────────
//  Type-coercion helpers at the artifact→D1 boundary.
//  No silent coercion of absent → 0. NULL preserves semantics.
// ─────────────────────────────────────────────────────────────

function num(v)      { return (typeof v === "number" && Number.isFinite(v)) ? v : null; }
function int(v)      { return (typeof v === "number" && Number.isFinite(v)) ? Math.trunc(v) : null; }
function boolAsInt(v){ return v === true ? 1 : (v === false ? 0 : null); }
function str(v)      { return (typeof v === "string" && v.length > 0) ? v : null; }

function enumOrOther(v, enumSet, ctx, logger) {
    if (typeof v !== "string" || v.length === 0) return null;
    if (enumSet.has(v)) return v;
    // Unknown value: bucket to OTHER, emit structured telemetry.
    logger.push({
        kind: "enum_unknown",
        enum: ctx.enum_name,
        target: ctx.target,
        site_name: ctx.site_name,
        observed_value: v,
    });
    return "OTHER";
}

function clampSiteTagsJson(rawSite, logger, ctx) {
    const allowed = {};
    for (const k of SITE_TAGS_JSON_ALLOWED_KEYS) {
        if (rawSite[k] !== undefined) allowed[k] = rawSite[k];
    }
    const serialized = JSON.stringify(allowed);
    if (serialized.length > SITE_TAGS_JSON_MAX_BYTES) {
        logger.push({
            kind: "site_tags_json_oversize",
            target: ctx.target, site_name: ctx.site_name,
            size: serialized.length, cap: SITE_TAGS_JSON_MAX_BYTES,
        });
        return null;
    }
    return serialized;
}

// ─────────────────────────────────────────────────────────────
//  W1 helpers
// ─────────────────────────────────────────────────────────────

function safeSub(obj, key) { return (obj && typeof obj === "object") ? obj[key] : undefined; }

function extractSiteFeaturesRow(s, target, siteName, bs, logger, sourceStamp, nowIso) {
    // Flatten cold_phase_fraction (dict → 7 scalars; §2 subfield pinning).
    const cpf = (s.cold_phase_fraction && typeof s.cold_phase_fraction === "object") ? s.cold_phase_fraction : {};

    // Flatten signal_preservation (dict → 10 scalars; no derived summary).
    const sp = (s.signal_preservation && typeof s.signal_preservation === "object") ? s.signal_preservation : {};

    // Flatten kcc scalar subfields (dict → 17 scalars; list subfields → site_kcc_candidates).
    const kcc = (s.kcc && typeof s.kcc === "object") ? s.kcc : {};

    // Worker-computed derivations (guarded against absent volumes / lining).
    const vol = num(s.volume);
    const spread = (vol && vol > 0) ? Math.cbrt(vol) : null;
    const spikeDensity = (vol && vol > 0 && num(s.spike_count) !== null)
        ? num(s.spike_count) / vol : null;
    let burialMeanMinDist = null;
    if (Array.isArray(s.lining_residues) && s.lining_residues.length > 0) {
        const dists = s.lining_residues
            .map(lr => (num(lr && lr.min_distance) !== null)
                ? num(lr.min_distance)
                : num(lr && lr.min_distance_angstrom))   // legacy key
            .filter(d => d !== null);
        if (dists.length > 0) {
            burialMeanMinDist = dists.reduce((a, b) => a + b, 0) / dists.length;
        }
    }

    const classification = enumOrOther(
        s.classification, CLASSIFICATION_ENUM,
        { enum_name: "classification", target, site_name: siteName },
        logger,
    );
    const thermClass = enumOrOther(
        s.therm_class, THERM_CLASS_ENUM,
        { enum_name: "therm_class", target, site_name: siteName },
        logger,
    );

    // is_cryptic derivation rule (explicit, documented in persistence-analogous style):
    // if therm_class is CRYPTIC → 1; else if therm_class present → 0; else NULL.
    let isCryptic = null;
    if (s.is_cryptic !== undefined) {
        isCryptic = boolAsInt(s.is_cryptic);
    } else if (thermClass !== null) {
        isCryptic = (thermClass === "CRYPTIC") ? 1 : 0;
    }

    const tagsJson = clampSiteTagsJson(s, logger, { target, site_name: siteName });

    // Values tuple MUST match the UPDATE column list below, in order.
    return {
        pk: { target, site_name: siteName, created_at: nowIso },
        values: [
            int(s.spike_count),
            int(bs.n_streams),                          // top-level; not hardcoded
            num(s.unsat_frac),
            spread,
            num(s.volume),
            burialMeanMinDist,                           // ingest-computed `burial`
            num(s.burial_score),                         // engine-native engine_burial_score
            spikeDensity,
            num(s.druggability),
            num(s.aromatic_score),
            Array.isArray(s.lining_residues) ? s.lining_residues.length : null,
            num(s.quality_score),
            num(s.rank_score),
            num(s.engine_geo),
            num(s.engine_chem),
            num(s.engine_phys),
            num(s.engine_vcs),
            num(s.tokenized_score),
            num(s.cryptic_score),
            int(s.gtck_rank),
            int(s.rank),
            num(s.rank_C), num(s.rank_G), num(s.rank_K), num(s.rank_L), num(s.rank_T),
            classification,
            thermClass,
            boolAsInt(s.is_druggable),
            isCryptic,
            int(s.catalytic_residue_count),
            num(s.ccns_tau),
            num(s.hysteresis_asymmetry),
            num(s.relative_asymmetry),
            // cold_phase_fraction subfields
            num(cpf.cold), num(cpf.hot), num(cpf.delta),
            int(cpf.heating_spike_count), num(cpf.heating_spike_rate),
            int(cpf.cooling_spike_count), num(cpf.cooling_spike_rate),
            num(s.onset_score),
            num(s.breathing_score),
            num(s.kinetic_accessibility),
            num(s.effective_delta_g_kcal_mol),
            num(s.delta_g_aromatic_kcal_mol),
            num(s.delta_g_cooperative_kcal_mol),
            num(s.delta_g_dewetting_kcal_mol),
            num(s.delta_g_electrostatic_kcal_mol),
            num(s.delta_g_sti_kcal_mol),
            num(s.frustrated_solvent_score),
            num(s.ray_escape_ratio),
            // signal_preservation subfields
            num(sp.causality_density),
            int(sp.coupled_voxels),
            int(sp.max_recurrence),
            num(sp.mean_recurrence),
            int(sp.n_voxels),
            int(sp.primary_residue_count),
            int(sp.primary_residue_id),
            num(sp.residue_concentration),
            int(sp.total_coupling),
            int(sp.total_recurrence),
            num(s.localization_score_raw),
            num(s.sphericity),
            // kcc scalar subfields
            int(kcc.active_causal_steps),
            int(kcc.total_steps),
            int(kcc.best_kcc_candidate_index),
            int(kcc.driver_residue_id),
            num(kcc.burst_motion),
            num(kcc.direction_score),
            num(kcc.kcc_confidence),
            num(kcc.lag_corr_peak),
            num(kcc.local_cov),
            num(kcc.motion_efficiency),
            num(kcc.temporal_corr),
            num(kcc.site_burst_motion),
            num(kcc.site_causal_lag),
            num(kcc.site_direction_score),
            num(kcc.site_lag_corr_peak),
            num(kcc.site_local_cov),
            num(kcc.site_motion_efficiency),
            num(s.tide_coupling_score),
            num(s.source_diversity),
            num(s.uv_enrichment_score),
            num(s.wd_coherence),
            tagsJson,
            sourceStamp.source,
            sourceStamp.source_version,
            // updated_at removed per v4.1 narrow patch (D1 column-cap unblock)
        ],
    };
}

// Column list for W1's column-scoped UPDATE. Order MUST match extractSiteFeaturesRow.values.
const W1_UPDATE_COLUMNS = [
    "spike_count","n_streams","unsat_frac","spread","volume","burial","engine_burial_score",
    "spike_density","druggability","aromatic_score","n_lining_residues",
    "quality_score","rank_score","engine_geo","engine_chem","engine_phys","engine_vcs",
    "tokenized_score","cryptic_score","gtck_rank","rank",
    "rank_C","rank_G","rank_K","rank_L","rank_T",
    "classification","therm_class","is_druggable","is_cryptic","catalytic_residue_count",
    "ccns_tau","hysteresis_asymmetry","relative_asymmetry",
    "cold_phase_cold_fraction","cold_phase_hot_fraction","cold_phase_delta",
    "cold_phase_heating_spike_count","cold_phase_heating_spike_rate",
    "cold_phase_cooling_spike_count","cold_phase_cooling_spike_rate",
    "onset_score","breathing_score","kinetic_accessibility",
    "effective_delta_g_kcal_mol",
    "delta_g_aromatic_kcal_mol","delta_g_cooperative_kcal_mol","delta_g_dewetting_kcal_mol",
    "delta_g_electrostatic_kcal_mol","delta_g_sti_kcal_mol",
    "frustrated_solvent_score","ray_escape_ratio",
    "signal_preservation_causality_density","signal_preservation_coupled_voxels",
    "signal_preservation_max_recurrence","signal_preservation_mean_recurrence",
    "signal_preservation_n_voxels","signal_preservation_primary_residue_count",
    "signal_preservation_primary_residue_id","signal_preservation_residue_concentration",
    "signal_preservation_total_coupling","signal_preservation_total_recurrence",
    "localization_score_raw","sphericity",
    "kcc_active_causal_steps","kcc_total_steps","kcc_best_candidate_index","kcc_driver_residue_id",
    "kcc_burst_motion","kcc_direction_score","kcc_confidence","kcc_lag_corr_peak","kcc_local_cov",
    "kcc_motion_efficiency","kcc_temporal_corr",
    "kcc_site_burst_motion","kcc_site_causal_lag","kcc_site_direction_score",
    "kcc_site_lag_corr_peak","kcc_site_local_cov","kcc_site_motion_efficiency",
    "tide_coupling_score","source_diversity","uv_enrichment_score","wd_coherence",
    "site_tags_json","source","source_version",
    // updated_at removed per v4.1 narrow patch (D1 column-cap unblock)
];

function w1SiteFeaturesSQL() {
    const setClause = W1_UPDATE_COLUMNS.map(c => `${c}=?`).join(",\n    ");
    return {
        insert: "INSERT OR IGNORE INTO site_features (target, site_name, created_at) VALUES (?, ?, ?)",
        update: `UPDATE site_features SET\n    ${setClause}\n  WHERE target=? AND site_name=?`,
    };
}

// ─────────────────────────────────────────────────────────────
//  Queue consumer — W1
// ─────────────────────────────────────────────────────────────

export default {
    async queue(batch, env, ctx) {
        for (const msg of batch.messages) {
            try {
                const obj = msg.body.object;
                const key = obj.key;
                const parts = key.split('/');
                if (parts.length < 3 || (parts[0] !== '10k-runs' && parts[0] !== '10k-runs-pct70')) {
                    console.log(`Ignoring non-10k-runs object: ${key}`);
                    msg.ack();
                    continue;
                }
                const target = parts[1];
                const spikePct = parts[0] === '10k-runs-pct70' ? 70 : 95;
                await processTarget(target, env, { spikePercentile: spikePct });
                msg.ack();
            } catch (err) {
                console.error(`Queue processing error:`, err);
                msg.retry();
            }
        }
    },

    async fetch(request, env) {
        const url = new URL(request.url);
        const path = url.pathname;

        if (path === '/' || path === '/health') {
            return Response.json({
                service: 'prism-feature-pipeline',
                feature_contract_version: FEATURE_CONTRACT_VERSION,
                event_contract_version: EVENT_CONTRACT_VERSION,
                endpoints: [
                    'GET  /targets[?status=&spike_percentile=]',
                    'GET  /targets/:target',
                    'GET  /features/:target',
                    'GET  /site-features/:target[?fields=ranker|full]',
                    'GET  /site-lining-residues/:target',
                    'GET  /site-kcc-candidates/:target',
                    'GET  /site-event-aggregates/:target',
                    'GET  /dcc[?grade=]',
                    'GET  /dcc/:target',
                    'POST /site-features/:target/temporal',
                    'POST /site-features/:target/dcc',
                    'POST /site-features/:target/event-aggregates',
                    'POST /site-features/:target/persistence  (RESERVED — BLOCKED)',
                    'POST /targets/:target/runtime',
                    'POST /reprocess/:target[?pct=]',
                ],
            });
        }

        // ── Targets ──
        if (path === '/targets') {
            const status = url.searchParams.get('status');
            const pct = url.searchParams.get('spike_percentile');
            let sql = 'SELECT * FROM targets WHERE 1=1';
            const binds = [];
            if (status) { sql += ' AND status = ?'; binds.push(status); }
            if (pct) { sql += ' AND spike_percentile = ?'; binds.push(parseInt(pct)); }
            sql += ' ORDER BY target';
            const rows = await env.DB.prepare(sql).bind(...binds).all();
            return Response.json({ count: rows.results.length, targets: rows.results });
        }
        if (path.startsWith('/targets/') && request.method === 'GET') {
            const target = path.substring('/targets/'.length);
            const row = await env.DB.prepare('SELECT * FROM targets WHERE target = ?')
                .bind(target).first();
            return row ? Response.json(row) : new Response('not found', { status: 404 });
        }

        // ── W4: runtime ──
        if (request.method === 'POST' && path.startsWith('/targets/') && path.endsWith('/runtime')) {
            const target = path.substring('/targets/'.length, path.length - '/runtime'.length);
            const body = await safeJson(request);
            if (!body || typeof body !== 'object') return badJson();
            await env.DB.prepare(
                `INSERT OR IGNORE INTO targets (target) VALUES (?)`
            ).bind(target).run();
            await env.DB.prepare(
                `UPDATE targets SET
                    engine_time_seconds            = ?,
                    engine_flags                   = ?,
                    engine_commit                  = ?,
                    engine_n_streams               = ?,
                    engine_mode                    = ?,
                    engine_simulation_time_sec     = ?,
                    engine_total_steps_per_stream  = ?,
                    lining_residue_cutoff_angstroms = ?,
                    binding_sites_json_sha256      = ?,
                    ground_truth_json_sha256       = ?
                 WHERE target = ?`
            ).bind(
                num(body.engine_time_seconds),
                str(body.engine_flags),
                str(body.engine_commit),
                int(body.engine_n_streams),
                str(body.engine_mode),
                num(body.engine_simulation_time_sec),
                int(body.engine_total_steps_per_stream),
                num(body.lining_residue_cutoff_angstroms),
                str(body.binding_sites_json_sha256),
                str(body.ground_truth_json_sha256),
                target,
            ).run();
            return Response.json({ target, runtime_updated: true });
        }

        // ── Features (legacy residue_features table, unchanged) ──
        if (path.startsWith('/features/')) {
            const target = path.substring('/features/'.length);
            const rows = await env.DB.prepare(
                'SELECT * FROM residue_features WHERE target = ? ORDER BY residue_id'
            ).bind(target).all();
            return Response.json({ target, count: rows.results.length, residues: rows.results });
        }

        // ── W2: DCC endpoint ──
        if (request.method === 'POST'
            && path.startsWith('/site-features/')
            && path.endsWith('/dcc')) {
            const target = path.substring('/site-features/'.length, path.length - '/dcc'.length);
            const body = await safeJson(request);
            if (!body || typeof body !== 'object') return badJson();
            const sites = Array.isArray(body.sites) ? body.sites : null;
            if (!sites) return Response.json({ error: 'missing sites[]' }, { status: 400 });

            const nowIso = new Date().toISOString();
            const stmts = [];
            for (const r of sites) {
                if (!r || !r.site_name) continue;
                stmts.push(env.DB.prepare(
                    `INSERT OR IGNORE INTO site_features (target, site_name, created_at) VALUES (?, ?, ?)`
                ).bind(target, r.site_name, nowIso));
                stmts.push(env.DB.prepare(
                    `UPDATE site_features SET
                        min_dist_to_ligand = ?,
                        graded_score       = ?,
                        dcc_metric_source  = ?
                     WHERE target = ? AND site_name = ?`
                ).bind(
                    num(r.min_dist_to_ligand),
                    num(r.graded_score),
                    str(r.dcc_metric_source),
                    target, r.site_name,
                ));
            }
            if (body.corrected_dcc) {
                const c = body.corrected_dcc;
                // corrected_dcc is single-owner W2; INSERT OR REPLACE is permitted here only.
                stmts.push(env.DB.prepare(
                    `INSERT OR REPLACE INTO corrected_dcc (
                        target, centroid_dcc, spike_dcc, spike_site, n_parquet_sites, dcc_grade,
                        ligand_centroid_x, ligand_centroid_y, ligand_centroid_z,
                        holo_source, is_pandda_fragment, is_templated_complex,
                        nucleic_chains, skip_reason, valid_for_dcc_validation, dcc_metric_used
                    ) VALUES (?,?,?,?,?,?, ?,?,?, ?,?,?, ?,?,?,?)`
                ).bind(
                    target, num(c.centroid_dcc), num(c.spike_dcc), str(c.spike_site),
                    int(c.n_parquet_sites), str(c.dcc_grade),
                    num(c.ligand_centroid_x), num(c.ligand_centroid_y), num(c.ligand_centroid_z),
                    str(c.holo_source), boolAsInt(c.is_pandda_fragment), boolAsInt(c.is_templated_complex),
                    str(c.nucleic_chains), str(c.skip_reason), boolAsInt(c.valid_for_dcc_validation),
                    str(c.dcc_metric_used),
                ));
            }
            if (stmts.length > 0) await env.DB.batch(stmts);
            return Response.json({ target, updated_sites: sites.length });
        }

        // ── W3a: temporal endpoint (column-scoped UPDATE) ──
        if (request.method === 'POST'
            && path.startsWith('/site-features/')
            && path.endsWith('/temporal')) {
            const target = path.substring('/site-features/'.length, path.length - '/temporal'.length);
            const body = await safeJson(request);
            if (!Array.isArray(body)) return Response.json({ error: 'expected array' }, { status: 400 });
            const nowIso = new Date().toISOString();
            const stmts = [];
            for (const r of body) {
                if (!r || !r.site_name) continue;
                stmts.push(env.DB.prepare(
                    `INSERT OR IGNORE INTO site_features (target, site_name, created_at) VALUES (?, ?, ?)`
                ).bind(target, r.site_name, nowIso));
                stmts.push(env.DB.prepare(
                    `UPDATE site_features SET
                        phase_transition_ratio   = ?,
                        warm_hold_spike_fraction = ?
                     WHERE target = ? AND site_name = ?`
                ).bind(
                    num(r.phase_transition_ratio),
                    num(r.warm_hold_spike_fraction),
                    target, r.site_name,
                ));
            }
            if (stmts.length > 0) await env.DB.batch(stmts);
            return Response.json({ target, updated: body.length });
        }

        // ── W3b: event-aggregates endpoint ──
        if (request.method === 'POST'
            && path.startsWith('/site-features/')
            && path.endsWith('/event-aggregates')) {
            const target = path.substring('/site-features/'.length, path.length - '/event-aggregates'.length);
            const body = await safeJson(request);
            if (!Array.isArray(body)) return Response.json({ error: 'expected array' }, { status: 400 });
            return await processEventAggregates(env, target, body);
        }

        // ── W5: persistence (RESERVED — BLOCKED) ──
        if (request.method === 'POST'
            && path.startsWith('/site-features/')
            && path.endsWith('/persistence')) {
            return Response.json({
                error: "persistence endpoint is RESERVED and BLOCKED",
                see: "docs/contracts/persistence_contract.md",
            }, { status: 423 });  // 423 Locked
        }

        // ── GET /site-features/:target ──
        if (request.method === 'GET' && path.startsWith('/site-features/')) {
            const target = path.substring('/site-features/'.length);
            const fields = url.searchParams.get('fields') || 'full';
            let cols;
            if (fields === 'ranker') {
                // Lean projection — v4 FEATURE_COLS + ids + label source.
                cols = [
                    'target', 'site_name',
                    'spike_count', 'n_streams', 'unsat_frac', 'persistence',
                    'spread', 'burial', 'spike_density',
                    'druggability', 'aromatic_score', 'n_lining_residues',
                    'phase_transition_ratio', 'warm_hold_spike_fraction',
                    'min_dist_to_ligand', 'graded_score', 'source',
                ];
            } else {
                cols = ['*'];
            }
            const rows = await env.DB.prepare(
                `SELECT ${cols.join(',')} FROM site_features WHERE target = ? ORDER BY spike_count DESC`
            ).bind(target).all();
            return Response.json({ target, count: rows.results.length, sites: rows.results,
                                   projection: fields });
        }

        if (path.startsWith('/site-lining-residues/')) {
            const target = path.substring('/site-lining-residues/'.length);
            const rows = await env.DB.prepare(
                'SELECT * FROM site_lining_residues WHERE target = ? ORDER BY site_name, residue_id'
            ).bind(target).all();
            return Response.json({ target, count: rows.results.length, lining: rows.results });
        }

        if (path.startsWith('/site-kcc-candidates/')) {
            const target = path.substring('/site-kcc-candidates/'.length);
            const rows = await env.DB.prepare(
                'SELECT * FROM site_kcc_candidates WHERE target = ? ORDER BY site_name, candidate_rank'
            ).bind(target).all();
            return Response.json({ target, count: rows.results.length, candidates: rows.results });
        }

        if (path.startsWith('/site-event-aggregates/')) {
            const target = path.substring('/site-event-aggregates/'.length);
            const rows = await env.DB.prepare(
                'SELECT * FROM site_event_aggregates WHERE target = ? ORDER BY site_name'
            ).bind(target).all();
            return Response.json({ target, count: rows.results.length, aggregates: rows.results });
        }

        if (path === '/dcc') {
            const grade = url.searchParams.get('grade');
            let sql = 'SELECT * FROM corrected_dcc';
            const binds = [];
            if (grade) { sql += ' WHERE dcc_grade = ?'; binds.push(grade); }
            sql += ' ORDER BY spike_dcc ASC';
            const rows = await env.DB.prepare(sql).bind(...binds).all();
            return Response.json({ count: rows.results.length, records: rows.results });
        }
        if (path.startsWith('/dcc/')) {
            const target = path.substring('/dcc/'.length);
            const row = await env.DB.prepare('SELECT * FROM corrected_dcc WHERE target = ?')
                .bind(target).first();
            return row ? Response.json(row) : new Response('not found', { status: 404 });
        }

        if (path === '/stats') {
            const targets = await env.DB.prepare('SELECT COUNT(*) as n FROM targets').first();
            const dcc = await env.DB.prepare('SELECT COUNT(*) as n FROM corrected_dcc').first();
            const sites = await env.DB.prepare('SELECT COUNT(*) as n FROM site_features').first();
            const lining = await env.DB.prepare('SELECT COUNT(*) as n FROM site_lining_residues').first();
            const kcc = await env.DB.prepare('SELECT COUNT(*) as n FROM site_kcc_candidates').first();
            const evAgg = await env.DB.prepare('SELECT COUNT(*) as n FROM site_event_aggregates').first();
            const quar = await env.DB.prepare('SELECT COUNT(*) as n FROM quarantined_event_aggregates').first();
            const persistNonNull = await env.DB.prepare(
                'SELECT COUNT(*) as n FROM site_features WHERE persistence IS NOT NULL'
            ).first();
            return Response.json({
                feature_contract_version: FEATURE_CONTRACT_VERSION,
                targets: targets?.n ?? 0,
                corrected_dcc: dcc?.n ?? 0,
                site_features: sites?.n ?? 0,
                site_lining_residues: lining?.n ?? 0,
                site_kcc_candidates: kcc?.n ?? 0,
                site_event_aggregates: evAgg?.n ?? 0,
                quarantined_event_aggregates: quar?.n ?? 0,
                persistence_nonnull_rows: persistNonNull?.n ?? 0,   // must be 0 while BLOCKED
            });
        }

        if (path.startsWith('/campaign/')) {
            const doId = env.CAMPAIGN.idFromName('global');
            const stub = env.CAMPAIGN.get(doId);
            return stub.fetch(request);
        }

        if (request.method === 'POST' && path.startsWith('/reprocess/')) {
            const target = path.substring('/reprocess/'.length);
            const pct = parseInt(url.searchParams.get('pct') || '95');
            try {
                await processTarget(target, env, { spikePercentile: pct });
                return Response.json({ target, reprocessed: true, spike_percentile: pct });
            } catch (e) {
                return Response.json({ target, error: String(e) }, { status: 500 });
            }
        }

        return new Response('not found', { status: 404 });
    },
};

function badJson() { return Response.json({ error: 'invalid JSON' }, { status: 400 }); }
async function safeJson(request) {
    try { return await request.json(); } catch (e) { return null; }
}

// ─────────────────────────────────────────────────────────────
//  W1 processTarget — queue-driven ingest
// ─────────────────────────────────────────────────────────────

async function processTarget(target, env, opts = {}) {
    const m = target.match(/^([0-9a-z]{4})_chain([A-Z0-9])$/i);
    if (!m) {
        console.log(`Target name mismatch: ${target}`);
        return;
    }
    const pdb_id = m[1].toLowerCase();
    const chain  = m[2].toUpperCase();
    const spikePct = opts.spikePercentile ?? 95;
    const r2Prefix = spikePct === 70 ? '10k-runs-pct70' : '10k-runs';

    const bsObj = await env.R2.get(`${r2Prefix}/${target}/${target}.binding_sites.json`);
    if (!bsObj) {
        console.log(`No binding_sites.json for ${target} under ${r2Prefix}`);
        return;
    }
    const bs = await bsObj.json();
    const sites = Array.isArray(bs.sites) ? bs.sites : [];

    let ligand_code = null, ligand_heavy_atoms = null;
    const gtObj = await env.R2.get(`${r2Prefix}/${target}/${target}_ground_truth.json`);
    if (gtObj) {
        const gt = await gtObj.json();
        ligand_code = str(gt?.ligand?.resname);
        ligand_heavy_atoms = int(gt?.ligand?.n_atoms);
    }

    const atom_count = int(bs.n_atoms);
    const nowIso = new Date().toISOString();
    const sourceStamp = {
        source: `r2_event_v2_${r2Prefix}`,
        source_version: FEATURE_CONTRACT_VERSION,
    };
    const logger = [];

    // ── targets row ──
    await env.DB.prepare(
        `INSERT OR IGNORE INTO targets (target) VALUES (?)`
    ).bind(target).run();
    await env.DB.prepare(
        `UPDATE targets SET
            pdb_id = ?, chain = ?, atom_count = ?,
            ligand_code = ?, ligand_heavy_atoms = ?,
            n_sites_detected = ?, status = 'completed',
            run_date = ?, spike_percentile = ?,
            engine_n_streams = ?, engine_mode = ?,
            engine_simulation_time_sec = ?, engine_total_steps_per_stream = ?,
            lining_residue_cutoff_angstroms = ?,
            feature_contract_version = ?
         WHERE target = ?`
    ).bind(
        pdb_id, chain, atom_count,
        ligand_code, ligand_heavy_atoms,
        sites.length,
        nowIso, spikePct,
        int(bs.n_streams), str(bs.mode),
        num(bs.simulation_time_sec), int(bs.total_steps_per_stream),
        num(bs.lining_residue_cutoff_angstroms),
        FEATURE_CONTRACT_VERSION,
        target,
    ).run();

    // ── site_features rows + site_lining_residues + site_kcc_candidates ──
    const sql = w1SiteFeaturesSQL();
    const stmts = [];
    let totalSpikes = 0;

    for (const s of sites) {
        const sid = s.id;
        if (sid === null || sid === undefined) continue;
        const siteName = `site${sid}`;
        totalSpikes += int(s.spike_count) || 0;

        const row = extractSiteFeaturesRow(s, target, siteName, bs, logger, sourceStamp, nowIso);

        // A. INSERT OR IGNORE (create row if new).
        stmts.push(env.DB.prepare(sql.insert).bind(row.pk.target, row.pk.site_name, row.pk.created_at));

        // B. Column-scoped UPDATE, preserving W2/W3a/W3b/W5 columns.
        stmts.push(env.DB.prepare(sql.update).bind(...row.values, target, siteName));

        // C. site_lining_residues — DELETE then INSERT (single-owner W1).
        stmts.push(env.DB.prepare(
            'DELETE FROM site_lining_residues WHERE target = ? AND site_name = ?'
        ).bind(target, siteName));
        if (Array.isArray(s.lining_residues)) {
            for (const lr of s.lining_residues) {
                const resid = int(lr && lr.resid);
                if (resid === null) continue;
                stmts.push(env.DB.prepare(
                    `INSERT OR IGNORE INTO site_lining_residues (
                        target, site_name, residue_id, residue_name, chain,
                        min_distance, n_atoms, is_catalytic, spike_attribution_count
                     ) VALUES (?,?,?, ?,?, ?, ?, ?, ?)`
                ).bind(
                    target, siteName, resid,
                    str(lr.resname), str(lr.chain),
                    num(lr.min_distance),
                    int(lr.n_atoms),
                    boolAsInt(lr.is_catalytic),
                    int(lr.spike_attribution_count),
                ));
            }
        }

        // D. site_kcc_candidates — DELETE then INSERT (single-owner W1).
        stmts.push(env.DB.prepare(
            'DELETE FROM site_kcc_candidates WHERE target = ? AND site_name = ?'
        ).bind(target, siteName));
        const kcc = (s.kcc && typeof s.kcc === "object") ? s.kcc : {};
        const candLists = [
            kcc.candidate_residue_ids,
            kcc.candidate_causal_weights,
            kcc.candidate_residue_support,
            kcc.candidate_kcc_burst_motion,
            kcc.candidate_kcc_causal_lag,
            kcc.candidate_kcc_confidence,
            kcc.candidate_kcc_direction_score,
            kcc.candidate_kcc_local_cov,
        ];
        const allArrays = candLists.every(a => Array.isArray(a));
        if (allArrays) {
            const lens = candLists.map(a => a.length);
            const n = lens[0];
            if (lens.some(l => l !== n)) {
                logger.push({ kind: "kcc_candidate_length_mismatch",
                              target, site_name: siteName, lengths: lens });
            } else {
                for (let j = 0; j < n; j++) {
                    stmts.push(env.DB.prepare(
                        `INSERT OR IGNORE INTO site_kcc_candidates (
                            target, site_name, candidate_rank,
                            candidate_residue_id, candidate_causal_weight, candidate_residue_support,
                            candidate_burst_motion, candidate_causal_lag, candidate_confidence,
                            candidate_direction_score, candidate_local_cov
                        ) VALUES (?,?,?, ?,?,?, ?,?,?, ?,?)`
                    ).bind(
                        target, siteName, j,
                        int(kcc.candidate_residue_ids[j]),
                        num(kcc.candidate_causal_weights[j]),
                        num(kcc.candidate_residue_support[j]),
                        num(kcc.candidate_kcc_burst_motion[j]),
                        num(kcc.candidate_kcc_causal_lag[j]),
                        num(kcc.candidate_kcc_confidence[j]),
                        num(kcc.candidate_kcc_direction_score[j]),
                        num(kcc.candidate_kcc_local_cov[j]),
                    ));
                }
            }
        }
    }

    // Chunk into D1 batches of ≤900 statements (leave headroom under the 1000-stmt cap).
    const CHUNK = 900;
    for (let i = 0; i < stmts.length; i += CHUNK) {
        await env.DB.batch(stmts.slice(i, i + CHUNK));
    }

    if (env.ANALYTICS) {
        env.ANALYTICS.writeDataPoint({
            blobs: [target, pdb_id, chain, ligand_code ?? 'none', FEATURE_CONTRACT_VERSION],
            doubles: [totalSpikes, sites.length, atom_count ?? 0, logger.length],
            indexes: [target],
        });
        for (const ev of logger) {
            env.ANALYTICS.writeDataPoint({
                blobs: [ev.kind, target, ev.site_name ?? '', JSON.stringify(ev).slice(0, 512)],
                doubles: [1],
                indexes: [target],
            });
        }
    }

    const doId = env.CAMPAIGN.idFromName('global');
    const stub = env.CAMPAIGN.get(doId);
    await stub.fetch(new Request('http://internal/complete', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
            target, status: 'completed', n_sites: sites.length, spike_count: totalSpikes,
        }),
    }));

    console.log(`✓ W1 ${target}: ${sites.length} sites, ${totalSpikes.toLocaleString()} spikes, ${logger.length} logged anomalies`);
}

// ─────────────────────────────────────────────────────────────
//  W3b — site_event_aggregates ingest with quarantine policy
// ─────────────────────────────────────────────────────────────

async function processEventAggregates(env, target, rows) {
    const nowIso = new Date().toISOString();
    const THRESH_PHASE = 0.001;   // closed enum
    const THRESH_SOURCE = 0.001;  // closed enum
    const THRESH_TYPE = 0.01;     // open enum

    const stmts = [];
    let accepted = 0, quarantined = 0;
    for (const r of rows) {
        if (!r || !r.site_name) continue;
        const n_events = Math.max(int(r.n_events) ?? 0, 0);
        const unknownPhase = int(r.count_phase_unknown) ?? 0;
        const unknownSource = int(r.count_source_other) ?? 0;
        const unknownType = int(r.count_type_other) ?? 0;
        const phaseFrac = n_events > 0 ? unknownPhase / n_events : 0;
        const sourceFrac = n_events > 0 ? unknownSource / n_events : 0;
        const typeFrac = n_events > 0 ? unknownType / n_events : 0;

        const quarantineReasons = [];
        if (phaseFrac > THRESH_PHASE) quarantineReasons.push(`phase_unknown_frac=${phaseFrac.toFixed(4)}`);
        if (sourceFrac > THRESH_SOURCE) quarantineReasons.push(`source_other_frac=${sourceFrac.toFixed(4)}`);
        if (typeFrac > THRESH_TYPE) quarantineReasons.push(`type_other_frac=${typeFrac.toFixed(4)}`);

        // Parent-row create (shared between site_features parent and this child).
        stmts.push(env.DB.prepare(
            `INSERT OR IGNORE INTO site_features (target, site_name, created_at) VALUES (?, ?, ?)`
        ).bind(target, r.site_name, nowIso));

        if (quarantineReasons.length > 0) {
            quarantined++;
            stmts.push(env.DB.prepare(
                `INSERT OR REPLACE INTO quarantined_event_aggregates (
                    target, site_name, event_contract_version, n_events,
                    count_phase_unknown, count_source_other, count_type_other,
                    quarantine_reason, quarantine_detail_json, computed_at
                ) VALUES (?,?,?,?, ?,?,?, ?,?,?)`
            ).bind(
                target, r.site_name,
                str(r.event_contract_version) ?? EVENT_CONTRACT_VERSION,
                n_events,
                unknownPhase, unknownSource, unknownType,
                quarantineReasons.join("; "),
                JSON.stringify({ phaseFrac, sourceFrac, typeFrac }),
                nowIso,
            ));
            continue;
        }

        accepted++;
        // site_event_aggregates is single-owner (W3b). INSERT OR REPLACE is permitted here.
        stmts.push(env.DB.prepare(
            `INSERT OR REPLACE INTO site_event_aggregates (
                target, site_name, event_contract_version, n_events,
                count_phase_cold_hold, count_phase_warm_hold, count_phase_heating,
                count_phase_cooling, count_phase_cold_return, count_phase_unknown,
                count_source_uv, count_source_lif, count_source_efp, count_source_other,
                count_type_bnz, count_type_unk, count_type_anion, count_type_cation, count_type_other,
                mean_intensity, std_intensity, mean_vibrational_energy, mean_water_density,
                mean_n_nearby_excited, nonzero_wavelength_count, aromatic_attribution_count,
                source_entropy_nat,
                phase_transition_ratio, warm_hold_spike_fraction,
                computed_at
            ) VALUES (?,?,?,?, ?,?,?,?,?,?, ?,?,?,?, ?,?,?,?,?, ?,?,?,?,?,?,?, ?, ?,?, ?)`
        ).bind(
            target, r.site_name,
            str(r.event_contract_version) ?? EVENT_CONTRACT_VERSION,
            n_events,
            int(r.count_phase_cold_hold), int(r.count_phase_warm_hold),
            int(r.count_phase_heating), int(r.count_phase_cooling),
            int(r.count_phase_cold_return), unknownPhase,
            int(r.count_source_uv), int(r.count_source_lif), int(r.count_source_efp), unknownSource,
            int(r.count_type_bnz), int(r.count_type_unk), int(r.count_type_anion),
            int(r.count_type_cation), unknownType,
            num(r.mean_intensity), num(r.std_intensity),
            num(r.mean_vibrational_energy), num(r.mean_water_density),
            num(r.mean_n_nearby_excited),
            int(r.nonzero_wavelength_count), int(r.aromatic_attribution_count),
            num(r.source_entropy_nat),
            num(r.phase_transition_ratio), num(r.warm_hold_spike_fraction),
            nowIso,
        ));
    }

    // Record event_contract_version on the target row (W3b stamps it).
    stmts.push(env.DB.prepare(
        `UPDATE targets SET event_contract_version = ? WHERE target = ?`
    ).bind(EVENT_CONTRACT_VERSION, target));

    const CHUNK = 900;
    for (let i = 0; i < stmts.length; i += CHUNK) {
        await env.DB.batch(stmts.slice(i, i + CHUNK));
    }
    return Response.json({ target, accepted, quarantined, total: rows.length });
}
