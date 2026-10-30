            "note": "Descriptive group disparities only; not proof of discrimination. Interpret small groups cautiously."}
    return reports

@app.post("/api/train")
def train(req: TrainRequest):
    df = datasets.get(req.dataset_id)
    if df is None: raise HTTPException(404, "Dataset not found.")
    if req.target not in df.columns: raise HTTPException(400, "Target column not found.")
    if not 0.1 <= req.test_size <= 0.4: raise HTTPException(400, "Test size must be between 0.10 and 0.40.")
    y, positive, original_classes = normalize_binary(df[req.target], req.default_label)
    valid = df[req.target].notna()
    df, y = df.loc[valid].copy(), y.loc[valid]
    excluded = set([req.target] + req.drop_columns + req.demographic_columns)
    X = df.drop(columns=[c for c in excluded if c in df.columns])
    if X.shape[1] == 0: raise HTTPException(400, "No predictive features remain after exclusions.")
    if y.value_counts().min() < 2: raise HTTPException(400, "Each target class needs at least two records.")
    try:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=req.test_size, random_state=42, stratify=y)
    except ValueError as exc:
        raise HTTPException(400, f"Unable to create stratified split: {exc}")
    estimators = {
        "Credit AI – A": LogisticRegression(max_iter=1200, class_weight="balanced"),
        "Credit AI – B": RandomForestClassifier(n_estimators=180, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "Credit AI – C": GradientBoostingClassifier(random_state=42)
    }
    run_id = str(uuid.uuid4())
    result_models = {}
    for name, estimator in estimators.items():
        pipe = make_pipeline(X_train, estimator)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else None
        metrics = metric_pack(y_test.to_numpy(), pred, prob)
        fair = fairness_report(df.loc[idx_test], y_test.reset_index(drop=True),
                               pred, req.demographic_columns)
        result_models[name] = {"metrics": metrics, "fairness": fair}
        models[(run_id, name)] = {"pipeline": pipe, "feature_columns": list(X.columns)}
    evaluations[run_id] = {"dataset_id": req.dataset_id, "target": req.target,
        "positive_label": positive, "classes": original_classes, "models": result_models,
        "test_indices": list(map(str, idx_test)), "created_at": datetime.now(timezone.utc).isoformat()}
    record_history("training", req.dataset_id, f"Trained 3 models; target={req.target}")
    return {"run_id": run_id, "target": req.target, "positive_label": positive,
            "test_records": int(len(X_test)), "models": result_models}

class ExternalRequest(BaseModel):
    actual_column: str
    prediction_column: str
    demographic_columns: list[str] = []
    default_label: Optional[str] = None

@app.post("/api/external-audit")
async def external_audit(file: UploadFile = File(...), config: str = ""):
    import json
    try: cfg = ExternalRequest(**json.loads(config))
    except Exception as exc: raise HTTPException(400, f"Invalid audit configuration: {exc}")
    df = read_upload(file.filename or "predictions.csv", await file.read())
    for c in [cfg.actual_column, cfg.prediction_column]:
        if c not in df.columns: raise HTTPException(400, f"Column '{c}' not found.")
    actual, positive, _ = normalize_binary(df[cfg.actual_column], cfg.default_label)
    pred_vals = df[cfg.prediction_column]
    # Predicted labels are mapped to the same positive convention as actual labels.
    unique_pred = list(pred_vals.dropna().unique())
    if len(unique_pred) != 2: raise HTTPException(400, "Prediction column must contain exactly two classes.")
    positive_pred = next((v for v in unique_pred if str(v) == positive), sorted(unique_pred, key=lambda x:str(x))[-1])
    mask = df[cfg.actual_column].notna() & pred_vals.notna()
    actual, pred = actual[mask], (pred_vals[mask] == positive_pred).astype(int)
    selected = df.loc[mask]
    metrics = metric_pack(actual.to_numpy(), pred.to_numpy())
    fair = fairness_report(selected, actual.reset_index(drop=True), pred.to_numpy(), cfg.demographic_columns)
    audit_id = str(uuid.uuid4())
    evaluations[audit_id] = {"dataset_id":"external-upload", "target":cfg.actual_column,
        "models":{"External AI": {"metrics":metrics,"fairness":fair}},
        "created_at":datetime.now(timezone.utc).isoformat()}
    record_history("external_audit", "external-upload", f"Audited external predictions; n={len(actual)}")
    return {"audit_id":audit_id, "models":evaluations[audit_id]["models"],
            "rows_evaluated":int(len(actual)), "positive_label":positive}

@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    if run_id not in evaluations: raise HTTPException(404, "Evaluation not found.")
    return evaluations[run_id]

@app.get("/api/runs/{run_id}/errors")
def get_errors(run_id: str, model: str):
    ev = evaluations.get(run_id)
    if not ev: raise HTTPException(404, "Evaluation not found.")
    if model not in ev["models"]: raise HTTPException(404, "Model not found.")
    return {"message":"Detailed row-level error inspection is not retained in this minimal build. Aggregate confusion-matrix errors are available.",
            "metrics":ev["models"][model]["metrics"]}

@app.get("/api/runs/{run_id}/report.csv")
def report_csv(run_id: str):
    ev = evaluations.get(run_id)
    if not ev: raise HTTPException(404, "Evaluation not found.")
    rows=[]
    for name, data in ev["models"].items():
        rows.append({"model":name, **data["metrics"]})
    out=io.StringIO(); pd.DataFrame(rows).to_csv(out,index=False)
    return StreamingResponse(iter([out.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition":f'attachment; filename="credvion_{run_id[:8]}_audit.csv"'})

@app.get("/api/runs/{run_id}/report.pdf")
def report_pdf(run_id: str):
    ev=evaluations.get(run_id)
    if not ev: raise HTTPException(404,"Evaluation not found.")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError: raise HTTPException(500,"Install reportlab to generate PDF reports.")
    buff=io.BytesIO(); doc=SimpleDocTemplate(buff,pagesize=A4); styles=getSampleStyleSheet()
    story=[Paragraph("CREDVION — AI CREDIT-RISK AUDIT",styles["Title"]),
           Paragraph(f"Evaluation date: {ev['created_at']}",styles["Normal"]),
           Paragraph(f"Dataset: {ev['dataset_id']} | Target: {ev['target']}",styles["Normal"]),Spacer(1,16)]
    data=[["Model","Accuracy","Precision","Recall","F1","FP","FN","Error rate"]]
    for name,item in ev["models"].items():
        m=item["metrics"]; data.append([name,*[f"{m[k]:.4f}" for k in ["accuracy","precision","recall","f1"]],
                                        str(m["fp"]),str(m["fn"]),f"{m['error_rate']:.4f}"])
    table=Table(data,repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#173d2b")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.grey),
        ("PADDING",(0,0),(-1,-1),7)]))
    story += [table,Spacer(1,14),Paragraph("Fairness interpretation",styles["Heading2"]),
              Paragraph("Group differences are descriptive signals for further investigation. They do not, by themselves, establish discrimination or causation. Review sample sizes, data quality, context and applicable law.",styles["BodyText"]),
              Paragraph("Limitations: This report evaluates supplied labels and predictions on the evaluated records only. It is not a lending decision or legal compliance certification.",styles["BodyText"])]

