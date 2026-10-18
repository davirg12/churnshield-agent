from datetime import datetime, timezone
import uuid
import pandas as pd


def generate_retention_actions(
    priority_segment: pd.DataFrame
) -> pd.DataFrame:
    """
    Genera recomendaciones simuladas para clientes priorizados.
    No envía mensajes ni ejecuta acciones comerciales reales.
    """
    required = {"customerID", "risk_score", "priority_rank"}
    missing = required - set(priority_segment.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias: {sorted(missing)}"
        )

    if priority_segment.empty:
        return pd.DataFrame(columns=[
            "action_id",
            "timestamp_utc",
            "customerID",
            "risk_score",
            "priority_rank",
            "action_type",
            "status"
        ])

    if priority_segment["customerID"].isna().any():
        raise ValueError("Hay clientes sin customerID.")

    if priority_segment["customerID"].duplicated().any():
        raise ValueError("Hay clientes duplicados en las acciones.")

    if priority_segment["risk_score"].isna().any():
        raise ValueError("Hay puntuaciones de riesgo nulas.")

    if not priority_segment["risk_score"].between(0, 1).all():
        raise ValueError("Las puntuaciones deben estar entre 0 y 1.")

    timestamp = datetime.now(timezone.utc).isoformat()
    actions = []

    for _, customer in priority_segment.iterrows():
        score = float(customer["risk_score"])

        if score >= 0.75:
            action_type = "priority_retention_offer"
        elif score >= 0.50:
            action_type = "personalized_retention_review"
        else:
            action_type = "preventive_follow_up"

        actions.append({
            "action_id": str(uuid.uuid4()),
            "timestamp_utc": timestamp,
            "customerID": customer["customerID"],
            "risk_score": score,
            "priority_rank": int(customer["priority_rank"]),
            "action_type": action_type,
            "status": "SIMULATED_NOT_SENT"
        })

    return pd.DataFrame(actions)
