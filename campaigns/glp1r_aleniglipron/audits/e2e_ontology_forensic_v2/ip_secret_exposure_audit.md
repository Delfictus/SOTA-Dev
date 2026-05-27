# IP Secret Exposure Audit

row_count: 34

| path | line | finding_type | risk_flag | redacted_fingerprint | notes |
| --- | --- | --- | --- | --- | --- |
| campaigns/glp1r_aleniglipron/visualizer_app/assets/index-CqvL8LbP.js | 37 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | !0,r...tion\|sha256:68f477927ebcdb29\|len:69 | Secret value intentionally omitted |
| scripts/vectorize_active_learning.py | 167 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | api_...auth\|sha256:14e5e97a2139c5b8\|len:8 | Secret value intentionally omitted |
| scripts/audit_e2e_ontology_pipeline.py | 204 | PATENT_RISK_DISCLOSURE | PATENT_RISK_DISCLOSURE | "PAT....I),\|sha256:4fda6b56c3c806a7\|len:124 | Line text intentionally omitted |
| scripts/audit_e2e_ontology_pipeline.py | 205 | TRADE_SECRET_LEAK | TRADE_SECRET_LEAK | "TRA....I),\|sha256:2f970aaf2c4655b8\|len:111 | Line text intentionally omitted |
| scripts/audit_e2e_ontology_pipeline.py | 206 | CLIENT_SENSITIVE_DATA | CLIENT_SENSITIVE_DATA | "CLI....I),\|sha256:d8f5dfa3a27962fe\|len:106 | Line text intentionally omitted |
| scripts/quarantine/rebuild_patent_docx.py | 22 | PATENT_RISK_DISCLOSURE | PATENT_RISK_DISCLOSURE | "FIE...RY",\|sha256:8bb3a8673f8ca99b\|len:50 | Line text intentionally omitted |
| scripts/quarantine/rebuild_patent_docx.py | 26 | PATENT_RISK_DISCLOSURE | PATENT_RISK_DISCLOSURE | "DES...ON",\|sha256:36885c6aca15aba5\|len:69 | Line text intentionally omitted |
| scripts/quarantine/rebuild_patent_docx_v2.py | 22 | PATENT_RISK_DISCLOSURE | PATENT_RISK_DISCLOSURE | "FIE...ON",\|sha256:96798ad22775924d\|len:25 | Line text intentionally omitted |
| scripts/managed-agents/setup_agents.py | 462 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | os.e...get(\|sha256:b0782cef1bb041f1\|len:15 | Secret value intentionally omitted |
| scripts/managed-agents/setup_agents.py | 468 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | api_...key)\|sha256:6a4d7cbd42d79ca9\|len:8 | Secret value intentionally omitted |
| scripts/managed-agents/r2_upload_webhook.py | 73 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | os.e...get(\|sha256:b0782cef1bb041f1\|len:15 | Secret value intentionally omitted |
| scripts/managed-agents/r2_upload_webhook.py | 86 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | api_...key)\|sha256:6a4d7cbd42d79ca9\|len:8 | Secret value intentionally omitted |
| scripts/production/w3b_event_aggregates_pin.py | 4 | PATENT_RISK_DISCLOSURE | PATENT_RISK_DISCLOSURE | Non-...vice\|sha256:8cd57d5e9bdf575e\|len:61 | Line text intentionally omitted |
| scripts/production/w4_pin_runtime.py | 11 | PATENT_RISK_DISCLOSURE | PATENT_RISK_DISCLOSURE | Non-...le):\|sha256:ebbe2f94ec6e8a53\|len:36 | Line text intentionally omitted |
| scripts/production/w4_pin_runtime.py | 121 | PATENT_RISK_DISCLOSURE | PATENT_RISK_DISCLOSURE | # No...ied.\|sha256:3e60fb6371af53b9\|len:77 | Line text intentionally omitted |
| scripts/production/w2_dcc_pin.py | 4 | PATENT_RISK_DISCLOSURE | PATENT_RISK_DISCLOSURE | Non-...vice\|sha256:8cd57d5e9bdf575e\|len:61 | Line text intentionally omitted |
| scripts/prism-r2-sync/setup_managed_agents.sh | 47 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | $ANT..._KEY\|sha256:895970180f456767\|len:18 | Secret value intentionally omitted |
| scripts/prism-r2-sync/setup_managed_agents.sh | 167 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | \$AN...KEY\\|sha256:d84113e70531a2fc\|len:20 | Secret value intentionally omitted |
| scripts/prism-r2-sync/setup_managed_agents.sh | 178 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | \$AN...KEY\\|sha256:d84113e70531a2fc\|len:20 | Secret value intentionally omitted |
| scripts/prism-r2-sync/setup_managed_agents.sh | 187 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | \$AN...KEY\\|sha256:d84113e70531a2fc\|len:20 | Secret value intentionally omitted |
| scripts/prism-r2-sync/setup_credential_vault.sh | 382 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | ${AN...one}\|sha256:27a1adbad5565c36\|len:26 | Secret value intentionally omitted |
| crates/prism-gpu/src/kernels/mega_fused_pocket_kernel.cu | 2375 | PATENT_RISK_DISCLOSURE | PATENT_RISK_DISCLOSURE | PATE...IMS:\|sha256:8253661334dd9089\|len:18 | Line text intentionally omitted |
| crates/prism-nhs/src/tokenized_ranker.rs | 183 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | self...unt,\|sha256:fc7b621870c78f7c\|len:31 | Secret value intentionally omitted |
| crates/prism-nhs/src/tokenized_ranker.rs | 228 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | rank...unt,\|sha256:2a01e65bf26f033d\|len:33 | Secret value intentionally omitted |
| crates/prism-nhs/src/tokenized_ranker.rs | 326 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | r.co...000,\|sha256:6d99924426854daf\|len:24 | Secret value intentionally omitted |
| docs/PRISM4D_Complete_Technical_Reference.md | 4 | TRADE_SECRET_LEAK | TRADE_SECRET_LEAK | **Cl...ling\|sha256:fcee2b89b00e39c4\|len:67 | Line text intentionally omitted |
| docs/USPTO_Patent_Application_v2.md | 91 | PATENT_RISK_DISCLOSURE | PATENT_RISK_DISCLOSURE | ### ...tion\|sha256:df426e3f243fca66\|len:28 | Line text intentionally omitted |
| docs/USPTO_Patent_Application_v2.md | 107 | PATENT_RISK_DISCLOSURE | PATENT_RISK_DISCLOSURE | The ...ion.\|sha256:085b3d0281c937fb\|len:231 | Line text intentionally omitted |
| docs/USPTO_Patent_Application_v2.md | 135 | PATENT_RISK_DISCLOSURE | PATENT_RISK_DISCLOSURE | The ...ure.\|sha256:960df86f003c6807\|len:385 | Line text intentionally omitted |
| docs/PRODUCTION_LOGBOOK.md | 78 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | pris...ctus\|sha256:addbbcd2cc0da5bf\|len:23 | Secret value intentionally omitted |
| docs/sops/innovations/INNOVATION_REGISTRY.md | 20 | TRADE_SECRET_LEAK | TRADE_SECRET_LEAK | > - ...md`)\|sha256:533745c15742e305\|len:104 | Line text intentionally omitted |
| docs/sops/innovations/INNOVATION_REGISTRY.md | 48 | TRADE_SECRET_LEAK | TRADE_SECRET_LEAK | - **...red)\|sha256:cfaa21c07c18246e\|len:80 | Line text intentionally omitted |
| docs/sops/infrastructure/CLOUDFLARE_INVENTORY.md | 149 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | pris...ctus\|sha256:addbbcd2cc0da5bf\|len:23 | Secret value intentionally omitted |
| docs/sops/infrastructure/CLOUDFLARE_INVENTORY.md | 154 | GENERIC_SECRET_ASSIGNMENT | CREDENTIAL_EXPOSED | pris...ctus\|sha256:addbbcd2cc0da5bf\|len:23 | Secret value intentionally omitted |
