from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

import pandas as pd

from app.perception import perceive
from app.decision import decide
from app.actions import generate_retention_actions


def run_agent(
    customers: pd.DataFrame,
    model,
    required_features: list[str],
    log_dir: str | Path,
    selection_fraction: float = 0.15
) -> dict:
    run_id = str(uuid.uuid4())
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    events_path = log_path / "agent_events.jsonl"

    def log_event(event_type: str, details: dict):
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "event_type": event_type,
            "details": details
        }

        with events_path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(event, ensure_ascii=False, default=str) + "\n"
            )

    # 1. Percepción
    perceived = perceive(customers, required_features)

    log_event("perception", {
        "status": "success",
        "customers_received": len(perceived),
        "features_received": len(required_features)
    })

    # 2. Predicción
    probabilities = model.predict_proba(
        perceived[required_features]
    )[:, 1]

    risk_results = pd.DataFrame({
        "customerID": perceived["customerID"].to_numpy(),
        "risk_score": probabilities
    })

    # 3. Decisión
    ranking, priority_segment = decide(
        risk_results,
        selection_fraction=selection_fraction
    )

    log_event("decision", {
        "status": "success",
        "customers_evaluated": len(ranking),
        "selection_fraction": selection_fraction,
        "customers_selected": len(priority_segment),
        "selected_customer_ids": priority_segment["customerID"].tolist()
    })

    # 4. Acción simulada
    actions = generate_retention_actions(priority_segment)

    log_event("action", {
        "status": "success",
        "actions_generated": len(actions),
        "execution_status": "SIMULATED_NOT_SENT",
        "action_ids": actions["action_id"].tolist()
    })

    # 5. Guardar resultados por ejecución
    ranking_path = log_path / f"ranking_{run_id}.csv"
    priority_path = log_path / f"priority_{run_id}.csv"
    actions_path = log_path / f"actions_{run_id}.csv"

    ranking.to_csv(ranking_path, index=False)
    priority_segment.to_csv(priority_path, index=False)
    actions.to_csv(actions_path, index=False)

    return {
        "run_id": run_id,
        "ranking": ranking,
        "priority_segment": priority_segment,
        "actions": actions,
        "events_path": str(events_path),
        "ranking_path": str(ranking_path),
        "priority_path": str(priority_path),
        "actions_path": str(actions_path)
    }
