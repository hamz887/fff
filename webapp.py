# -*- coding: utf-8 -*-
"""
webapp.py — academic Flask interface for the saved detection system (python main.py serve).

Robustness: upload cap (MAX_CONTENT_LENGTH), row limit, schema validation with structured HTTP 400 responses,
explicit missing-field warnings, no tracebacks leaked to the client.
"""
import io
import json

import pandas as pd

from .detector import Detector, SchemaError

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>KAN-IDS Detection System</title>
<style>
 body{font-family:Georgia,'Times New Roman',serif;margin:0;background:#fafaf7;color:#222}
 header{background:#1f3a5f;color:#fff;padding:18px 32px}header h1{margin:0;font-size:22px}
 header p{margin:4px 0 0;font-size:13px;opacity:.85}main{max-width:1100px;margin:24px auto;padding:0 20px}
 .card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:16px 20px;margin-bottom:18px}
 h2{font-size:17px;border-bottom:1px solid #ddd;padding-bottom:6px;margin-top:0;color:#1f3a5f}
 table{border-collapse:collapse;width:100%;font-size:13px;font-family:Arial,sans-serif}
 th,td{border-bottom:1px solid #eee;padding:6px 8px;text-align:left}th{background:#f1f3f6}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
 .kpi{background:#f1f3f6;border-radius:6px;padding:10px;text-align:center;font-family:Arial,sans-serif}
 .kpi b{display:block;font-size:20px;color:#1f3a5f}.A{color:#b3261e;font-weight:bold}.N{color:#1e7b34;font-weight:bold}
 button{background:#1f3a5f;color:#fff;border:0;padding:8px 18px;border-radius:4px;cursor:pointer}
 textarea{width:100%;height:90px;font-family:monospace;font-size:12px}small{color:#666}
 .warn{border-color:#e0b000;background:#fffbea}.err{border-color:#b3261e;background:#fdf0ef}
</style></head><body>
<header><h1>KAN-IDS — Edge-IIoT Intrusion Detection System</h1>
<p>Primary deployed model: <b>{{card.architecture}}</b> ({{card.role}}) · hierarchical detection: binary detector
({{card.threshold_policy}}) followed by a softmax attack-type classifier</p></header>
<main>
<div class="card"><h2>Model card — {{card.architecture}}</h2>
<p><small><b>Training:</b> {{card.training_paradigm}}<br><b>Features:</b> {{card.feature_selection}}</small></p><div class="grid">
<div class="kpi"><b>{{card.architecture}}</b>architecture</div><div class="kpi"><b>{{card.n_features}}</b>input features</div>
<div class="kpi"><b>{{'%.4f'|format(card.test_metrics.f1)}}</b>test F1</div><div class="kpi"><b>{{'%.4f'|format(card.test_metrics.mcc)}}</b>test MCC</div>
<div class="kpi"><b>{{'%.4f'|format(card.test_metrics.far)}}</b>test FAR</div><div class="kpi"><b>{{card.attack_types|length}}</b>attack types</div>
</div><p><small>Raw fields required: {{card.raw_fields_needed|join(', ')}}</small></p></div>
<div class="card"><h2>Submit traffic records</h2>
<form method="post" action="/detect" enctype="multipart/form-data">
<p>CSV file with Edge-IIoTset columns (extra columns are ignored): <input type="file" name="file" accept=".csv"></p>
<p>or paste CSV text (with header):</p><textarea name="text"></textarea>
<p><label><input type="checkbox" name="allow_missing" value="1"> allow missing required fields (imputed with 0 —
not recommended)</label></p><p><small>Maximum upload: {{max_mb}} MB, {{max_rows}} records.</small></p>
<p><button type="submit">Detect</button></p></form></div>
{% if warnings %}<div class="card warn"><h2>Warnings</h2><ul>{% for w in warnings %}<li>{{w}}</li>{% endfor %}</ul></div>{% endif %}
{% if summary %}<div class="card"><h2>Detection summary ({{n}} records)</h2><div class="grid">
{% for k,v in summary %}<div class="kpi"><b>{{v}}</b>{{k}}</div>{% endfor %}</div></div>
<div class="card"><h2>Per-record results {% if n > shown %}<small>(first {{shown}})</small>{% endif %}</h2><table>
<tr><th>#</th><th>Decision</th><th>Attack probability</th><th>Attack type</th><th>Type confidence</th></tr>
{% for i,r in rows %}<tr><td>{{i}}</td><td class="{{r.decision[0]}}">{{r.decision}}</td><td>{{'%.4f'|format(r.attack_probability)}}</td>
<td>{{r.attack_type}}</td><td>{{'' if r.type_confidence != r.type_confidence else '%.3f'|format(r.type_confidence)}}</td></tr>{% endfor %}
</table></div>{% endif %}{% if error %}<div class="card err"><h2>Input rejected (HTTP 400)</h2><p>{{error}}</p>
{% if details %}<pre style="white-space:pre-wrap;font-size:12px">{{details}}</pre>{% endif %}</div>{% endif %}
</main></body></html>"""


def create_app(bundle_path):
    from flask import Flask, request, render_template_string, jsonify
    det = Detector(bundle_path)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = det.cfg.max_upload_mb * 1024 * 1024
    base = dict(card=det.card, max_mb=det.cfg.max_upload_mb, max_rows=f"{det.cfg.max_rows:,}")
    model_meta = {k: det.card.get(k) for k in ("architecture", "model_key", "role", "training_paradigm",
                                                "feature_selection", "n_features", "threshold", "threshold_policy")}

    def page(status=200, **kw):
        return render_template_string(PAGE, **{**base, "summary": None, "warnings": None, "error": None,
                                               "details": None, **kw}), status

    def api_error(status, code, message, details=None):
        return jsonify({"error": code, "message": message, "details": details or {}}), status

    @app.errorhandler(413)
    def too_large(_):
        msg = f"request larger than {det.cfg.max_upload_mb} MB"
        return api_error(413, "payload_too_large", msg) if request.path.startswith("/api") else page(413, error=msg)

    @app.route("/")
    def index():
        return page()

    @app.route("/detect", methods=["POST"])
    def detect():
        try:
            f = request.files.get("file")
            if f and f.filename:
                if not f.filename.lower().endswith(".csv"):
                    raise SchemaError("only .csv files are accepted")
                df = pd.read_csv(f, low_memory=False)
            elif request.form.get("text", "").strip():
                df = pd.read_csv(io.StringIO(request.form["text"]), low_memory=False)
            else:
                raise SchemaError("no file or text submitted")
            out, warns = det.detect(df, allow_missing=request.form.get("allow_missing") == "1")
        except SchemaError as e:
            return page(400, error=str(e), details=json.dumps(e.details, indent=2) if e.details else None)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            return page(400, error=f"malformed CSV: {type(e).__name__}")
        except Exception as e:
            app.logger.exception("detection failed")
            return page(500, error=f"internal error ({type(e).__name__})")
        summ = [("Normal", int((out.decision == "Normal").sum())), ("Attack", int((out.decision == "Attack").sum()))]
        summ += [(t, int(n)) for t, n in out[out.decision == "Attack"].attack_type.value_counts().items()]
        shown = min(len(out), 500)
        return page(summary=summ, warnings=warns, n=len(out), shown=shown,
                    rows=list(enumerate(out.head(shown).itertuples(), 1)))

    @app.route("/api/detect", methods=["POST"])
    def api():
        """Body: a JSON list of records, or {"records": [...], "allow_missing": false}."""
        payload = request.get_json(silent=True)
        if payload is None:
            return api_error(400, "invalid_json", "the request body must be valid JSON (Content-Type: application/json)")
        allow = request.args.get("allow_missing", "false").lower() == "true"
        if isinstance(payload, dict):
            allow = bool(payload.get("allow_missing", allow))
            payload = payload.get("records")
        if not isinstance(payload, list) or not payload:
            return api_error(400, "invalid_records", "expected a non-empty list of records (JSON objects)")
        if not all(isinstance(r, dict) for r in payload):
            return api_error(400, "invalid_records", "every record must be a JSON object of field: value")
        try:
            out, warns = det.detect(pd.DataFrame(payload), allow_missing=allow)
        except SchemaError as e:
            return api_error(400, "schema_error", str(e), e.details)
        except Exception as e:
            app.logger.exception("API detection failed")
            return api_error(500, "internal_error", f"unexpected {type(e).__name__}")
        return jsonify({"model": model_meta, "n": len(out), "warnings": warns,
                        "results": json.loads(out.to_json(orient="records"))})

    @app.route("/api/model", methods=["GET"])
    def api_model():
        return jsonify({**model_meta, "test_metrics": det.card.get("test_metrics", {}),
                        "attack_type_metrics": det.card.get("attack_type_metrics", {}),
                        "features": det.card.get("features", []),
                        "raw_fields_needed": det.card.get("raw_fields_needed", [])})
    return app
