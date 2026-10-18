import pandas as pd


def perceive(customers: pd.DataFrame, required_features: list[str]) -> pd.DataFrame:
    """
    Valida y prepara los datos recibidos por el agente.

    Exige customerID y todas las características utilizadas por el modelo.
    No modifica el DataFrame original.
    """
    if not isinstance(customers, pd.DataFrame):
        raise TypeError("La entrada debe ser un DataFrame de pandas.")

    if customers.empty:
        raise ValueError("No se recibieron clientes para evaluar.")

    required_columns = ["customerID"] + required_features
    missing_columns = [
        column for column in required_columns
        if column not in customers.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas obligatorias: {missing_columns}"
        )

    data = customers[required_columns].copy()

    if data["customerID"].isna().any():
        raise ValueError("Hay clientes sin customerID.")

    if data["customerID"].duplicated().any():
        raise ValueError("Hay customerID duplicados en la entrada.")

    if data[required_features].isna().all(axis=1).any():
        raise ValueError(
            "Hay clientes sin ninguna característica disponible."
        )

    return data
