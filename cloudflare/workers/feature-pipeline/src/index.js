/**
 * PRISM-4D Feature Pipeline Worker
 *
 * Three capabilities in one worker:
 *
 *   1. Queue consumer (Phase 2.2): fires on R2 `binding_sites.json` uploads,
 *      reads the target's data from R2, extracts per-site features, writes to
 *      D1 `site_features` and `targets` tables. Also pushes telemetry to
 *      Analytics Engine.
 *
 *   2. Read API (Phase 5.3): HTTP endpoints for RunPod to pull training data
 *      from D1 without touching R2 directly.
 *
 *   3. Campaign tracker access (Phase 2.3): proxy HTTP requests to the
 *      CampaignTracker Durable Object.
 *
 * The Durable Object itself is defined in `campaign_tracker.js`.
 */

export { CampaignTracker } from './campaign_tracker.js';

export default {
    /**
     * Queue consumer — Phase 2.2.
     * Fires when a `binding_sites.json` lands in R2. Extracts features and
     * writes to D1. One target per message.
     */
    async queue(batch, env, ctx) {
        for (const msg of batch.messages) {
            try {
                const obj = msg.body.object;
                // e.g. "10k-runs/9ymg_chainC/9ymg_chainC.binding_sites.json"
                // or  "10k-runs-pct70/9ymg_chainC/9ymg_chainC.binding_sites.json"
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

    /**
     * HTTP handler — Phase 5.3 (read API) + Phase 2.3 (campaign tracker proxy).
     */
    async fetch(request, env) {
        const url = new URL(request.url);
        const path = url.pathname;

        // Health check
        if (path === '/' || path === '/health') {
            return Response.json({
                service: 'prism-feature-pipeline',
                status: 'ok',
                endpoints: [
                    'GET /targets[?status=completed&spike_percentile=95]',
                    'GET /targets/:target',
                    'GET /features/:target',
                    'GET /site-features/:target',
                    'GET /dcc[?grade=EXCELLENT]',
                    'GET /dcc/:target',
                    'GET /campaign/status',
                    'POST /campaign/complete  (body: {target,status,dcc,spike_count})',
                    'GET /stats',
                ],
            });
        }

        // ── Phase 5.3 read API ──
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

        if (path.startsWith('/targets/')) {
            const target = path.substring('/targets/'.length);
            const row = await env.DB.prepare(
                'SELECT * FROM targets WHERE target = ?'
            ).bind(target).first();
            return row ? Response.json(row) : new Response('not found', { status: 404 });
        }

        if (path.startsWith('/features/')) {
            const target = path.substring('/features/'.length);
            const rows = await env.DB.prepare(
                'SELECT * FROM residue_features WHERE target = ? ORDER BY residue_id'
            ).bind(target).all();
            return Response.json({ target, count: rows.results.length, residues: rows.results });
        }

        // Offline temporal-feature upload (POST) — must be checked before the
        // generic GET /site-features/<target> handler below.
        if (request.method === 'POST'
            && path.startsWith('/site-features/')
            && path.endsWith('/temporal')) {
            const target = path.substring('/site-features/'.length,
                                          path.length - '/temporal'.length);
            let body;
            try {
                body = await request.json();
            } catch (e) {
                return Response.json({ error: 'invalid JSON' }, { status: 400 });
            }
            if (!Array.isArray(body)) {
                return Response.json({ error: 'expected array' }, { status: 400 });
            }
            const stmts = body.map(r => env.DB.prepare(
                `UPDATE site_features
                   SET phase_transition_ratio = ?, warm_hold_spike_fraction = ?
                 WHERE target = ? AND site_name = ?`
            ).bind(
                r.phase_transition_ratio ?? null,
                r.warm_hold_spike_fraction ?? null,
                target, r.site_name,
            ));
            if (stmts.length > 0) {
                await env.DB.batch(stmts);
            }
            return Response.json({ target, updated: stmts.length });
        }

        if (request.method === 'GET' && path.startsWith('/site-features/')) {
            const target = path.substring('/site-features/'.length);
            const rows = await env.DB.prepare(
                'SELECT * FROM site_features WHERE target = ? ORDER BY spike_count DESC'
            ).bind(target).all();
            return Response.json({ target, count: rows.results.length, sites: rows.results });
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
            const row = await env.DB.prepare(
                'SELECT * FROM corrected_dcc WHERE target = ?'
            ).bind(target).first();
            return row ? Response.json(row) : new Response('not found', { status: 404 });
        }

        if (path === '/stats') {
            const targets = await env.DB.prepare('SELECT COUNT(*) as n FROM targets').first();
            const dcc = await env.DB.prepare('SELECT COUNT(*) as n FROM corrected_dcc').first();
            const sites = await env.DB.prepare('SELECT COUNT(*) as n FROM site_features').first();
            const residues = await env.DB.prepare('SELECT COUNT(*) as n FROM residue_features').first();
            const embeds = await env.DB.prepare('SELECT COUNT(*) as n FROM esm2_embeddings').first();
            return Response.json({
                targets: targets?.n ?? 0,
                corrected_dcc: dcc?.n ?? 0,
                site_features: sites?.n ?? 0,
                residue_features: residues?.n ?? 0,
                esm2_embeddings: embeds?.n ?? 0,
            });
        }

        // ── Phase 2.3 campaign tracker proxy ──
        if (path.startsWith('/campaign/')) {
            const doId = env.CAMPAIGN.idFromName('global');
            const stub = env.CAMPAIGN.get(doId);
            return stub.fetch(request);
        }

        // ── Manual backfill: re-process a target that missed the queue ──
        // POST /reprocess/<target>?pct=70  (default pct=95)
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

/**
 * Process a single target: read binding_sites.json + list of spike
 * parquets from R2, compute per-site features, write to D1, push
 * telemetry to Analytics Engine.
 */
async function processTarget(target, env, opts = {}) {
    const m = target.match(/^([0-9a-z]{4})_chain([A-Z0-9])$/i);
    if (!m) {
        console.log(`Target name mismatch: ${target}`);
        return;
    }
    const pdb_id = m[1].toLowerCase();
    const chain = m[2].toUpperCase();
    const spikePct = opts.spikePercentile ?? 95;
    const r2Prefix = spikePct === 70 ? '10k-runs-pct70' : '10k-runs';

    // Read binding_sites.json
    const bsObj = await env.R2.get(`${r2Prefix}/${target}/${target}.binding_sites.json`);
    if (!bsObj) {
        console.log(`No binding_sites.json for ${target} under ${r2Prefix}`);
        return;
    }
    const bs = await bsObj.json();
    const sites = bs.sites || [];

    // Read ground truth if it exists
    let ligand_code = null;
    let ligand_heavy_atoms = null;
    const gtObj = await env.R2.get(`${r2Prefix}/${target}/${target}_ground_truth.json`);
    if (gtObj) {
        const gt = await gtObj.json();
        ligand_code = gt?.ligand?.resname ?? null;
        ligand_heavy_atoms = gt?.ligand?.n_atoms ?? null;
    }

    // Atom/residue count from binding_sites.json
    const atom_count = bs.n_atoms ?? null;

    // ── Write targets row ──
    await env.DB.prepare(
        `INSERT OR REPLACE INTO targets (
            target, pdb_id, chain, atom_count, ligand_code, ligand_heavy_atoms,
            n_sites_detected, status, run_date, spike_percentile
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?)`
    ).bind(
        target, pdb_id, chain, atom_count,
        ligand_code, ligand_heavy_atoms,
        sites.length,
        new Date().toISOString(),
        spikePct
    ).run();

    // ── Write site_features rows ──
    const stmts = [];
    let total_spikes = 0;
    for (const s of sites) {
        const siteName = `site${s.id ?? 'unknown'}`;
        const spike_count = s.spike_count ?? 0;
        total_spikes += spike_count;

        const vol = s.volume ?? s.volume_angstrom3 ?? null;
        const spread = (vol && vol > 0) ? Math.cbrt(vol) : null;
        const spike_density = (spike_count && spread && spread > 0)
            ? spike_count / Math.pow(spread, 3) : null;

        // Burial from lining residues if present
        let burial = null;
        if (Array.isArray(s.lining_residues) && s.lining_residues.length > 0) {
            const dists = s.lining_residues
                .map(lr => lr.min_distance ?? lr.min_distance_angstrom)
                .filter(d => typeof d === 'number');
            if (dists.length > 0) {
                burial = dists.reduce((a, b) => a + b, 0) / dists.length;
            }
        }

        // Unsat_frac + tokenized_score may be in the JSON from post-phase-7 runs
        const unsat_frac = s.unsat_frac ?? null;

        stmts.push(
            env.DB.prepare(
                `INSERT OR REPLACE INTO site_features (
                    target, site_name, spike_count, n_streams,
                    unsat_frac, spread, burial, spike_density, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'r2_event')`
            ).bind(target, siteName, spike_count, 4,
                   unsat_frac, spread, burial, spike_density)
        );
    }

    if (stmts.length > 0) {
        await env.DB.batch(stmts);
    }

    // ── Analytics Engine telemetry (Phase 2.4) ──
    if (env.ANALYTICS) {
        env.ANALYTICS.writeDataPoint({
            blobs: [target, pdb_id, chain, ligand_code ?? 'none'],
            doubles: [total_spikes, sites.length, atom_count ?? 0],
            indexes: [target],
        });
    }

    // ── Campaign tracker update (Phase 2.3) ──
    const doId = env.CAMPAIGN.idFromName('global');
    const stub = env.CAMPAIGN.get(doId);
    await stub.fetch(new Request('http://internal/complete', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
            target, status: 'completed', n_sites: sites.length, spike_count: total_spikes,
        }),
    }));

    console.log(`✓ Processed ${target}: ${sites.length} sites, ${total_spikes.toLocaleString()} spikes`);
}
