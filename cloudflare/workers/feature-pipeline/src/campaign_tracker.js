/**
 * Campaign Tracker — Phase 2.3 Durable Object.
 *
 * Single-instance counter of a running corpus campaign. Updated by the
 * queue consumer as targets complete; queryable by anyone with the
 * campaign/status endpoint.
 */

export class CampaignTracker {
    constructor(state, env) {
        this.state = state;
        this.env = env;
    }

    async fetch(request) {
        const url = new URL(request.url);
        const path = url.pathname;

        if (path === '/campaign/status' || path === '/status') {
            const state = await this.getState();
            return Response.json(state);
        }

        if (path === '/campaign/complete' || path === '/complete') {
            const { target, status, n_sites, spike_count, dcc } = await request.json();
            const state = await this.getState();
            state.targets[target] = {
                status: status ?? 'completed',
                n_sites: n_sites ?? null,
                spike_count: spike_count ?? null,
                dcc: dcc ?? null,
                timestamp: Date.now(),
            };
            if ((status ?? 'completed') === 'completed') {
                state.completed = (state.completed ?? 0) + 1;
                state.total_spikes = (state.total_spikes ?? 0) + (spike_count ?? 0);
                state.total_sites = (state.total_sites ?? 0) + (n_sites ?? 0);
            } else {
                state.failed = (state.failed ?? 0) + 1;
            }
            state.last_update = Date.now();
            await this.state.storage.put('campaign', state);
            return Response.json({ ok: true, completed: state.completed, failed: state.failed });
        }

        if (path === '/campaign/reset') {
            await this.state.storage.delete('campaign');
            return Response.json({ ok: true, reset: true });
        }

        return new Response('unknown campaign path: ' + path, { status: 404 });
    }

    async getState() {
        const state = await this.state.storage.get('campaign');
        return state ?? {
            started: Date.now(),
            last_update: null,
            completed: 0,
            failed: 0,
            total_spikes: 0,
            total_sites: 0,
            targets: {},
        };
    }
}
