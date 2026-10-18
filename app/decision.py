import math
import pandas as pd


def decide(
    risk_results: pd.DataFrame,
    selection_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ordena clientes por riesgo y selecciona el segmento prioritario.

    Devuelve:
    - ranking: todos los clientes ordenados por riesgo.
    - priority_segment: clientes seleccionados para una acción de retención.
    """
    if not 0 < selection_fraction <= 1:
        raise ValueError("selection_fraction debe ser mayor que 0 y máximo 1.")

    required = {"customerID", "risk_score"}
    missing = required - set(risk_results.columns)

    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    if risk_results.empty:
        raise ValueError("No hay resultados de riesgo para priorizar.")

    if risk_results["customerID"].isna().any():
        raise ValueError("Hay resultados sin customerID.")

    if risk_results["customerID"].duplicated().any():
        raise ValueError("Hay customerID duplicados en los resultados.")

    if risk_results["risk_score"].isna().any():
        raise ValueError("Hay puntuaciones de riesgo nulas.")

    if not risk_results["risk_score"].between(0, 1).all():
        raise ValueError("Las puntuaciones deben estar entre 0 y 1.")

    ranking = (
        risk_results[["customerID", "risk_score"]]
        .copy()
        .sort_values(
            ["risk_score", "customerID"],
            ascending=[False, True]
        )
        .reset_index(drop=True)
    )

    ranking.insert(0, "priority_rank", range(1, len(ranking) + 1))

    n_selected = math.ceil(selection_fraction * len(ranking))
    ranking["selected_for_retention"] = False
    ranking.loc[:n_selected - 1, "selected_for_retention"] = True

    priority_segment = ranking.head(n_selected).copy()

    return ranking, priority_segment
