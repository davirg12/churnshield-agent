import pandas as pd
import pytest

from app.perception import perceive
from app.decision import decide
from app.actions import generate_retention_actions


# ---------- PERCEPCIÓN ----------

def test_perception_accepts_valid_customers():
    customers = pd.DataFrame({
        "customerID": ["C001", "C002"],
        "tenure": [12, 24],
        "Contract": ["Month-to-month", "Two year"]
    })

    result = perceive(customers, ["tenure", "Contract"])

    assert len(result) == 2
    assert list(result.columns) == [
        "customerID", "tenure", "Contract"
    ]


def test_perception_rejects_empty_input():
    customers = pd.DataFrame(
        columns=["customerID", "tenure", "Contract"]
    )

    with pytest.raises(ValueError, match="No se recibieron clientes"):
        perceive(customers, ["tenure", "Contract"])


def test_perception_rejects_missing_column():
    customers = pd.DataFrame({
        "customerID": ["C001"],
        "tenure": [12]
    })

    with pytest.raises(ValueError, match="Faltan columnas"):
        perceive(customers, ["tenure", "Contract"])


def test_perception_rejects_duplicate_customer_ids():
    customers = pd.DataFrame({
        "customerID": ["C001", "C001"],
        "tenure": [12, 24],
        "Contract": ["Month-to-month", "Two year"]
    })

    with pytest.raises(ValueError, match="duplicados"):
        perceive(customers, ["tenure", "Contract"])


def test_perception_rejects_missing_customer_id():
    customers = pd.DataFrame({
        "customerID": ["C001", None],
        "tenure": [12, 24],
        "Contract": ["Month-to-month", "Two year"]
    })

    with pytest.raises(ValueError, match="sin customerID"):
        perceive(customers, ["tenure", "Contract"])


def test_perception_rejects_customer_without_features():
    customers = pd.DataFrame({
        "customerID": ["C001", "C002"],
        "tenure": [None, 24],
        "Contract": [None, "Two year"]
    })

    with pytest.raises(
        ValueError,
        match="ninguna característica"
    ):
        perceive(customers, ["tenure", "Contract"])


# ---------- DECISIÓN ----------

def test_decision_selects_ceil_of_requested_fraction():
    results = pd.DataFrame({
        "customerID": ["C1", "C2", "C3", "C4", "C5"],
        "risk_score": [0.9, 0.8, 0.7, 0.4, 0.1]
    })

    ranking, priority = decide(results, selection_fraction=0.15)

    # ceil(5 * 0.15) = 1
    assert len(priority) == 1
    assert priority.iloc[0]["customerID"] == "C1"


def test_decision_rejects_invalid_fraction():
    results = pd.DataFrame({
        "customerID": ["C1"],
        "risk_score": [0.9]
    })

    with pytest.raises(ValueError, match="selection_fraction"):
        decide(results, selection_fraction=0)


def test_decision_rejects_duplicate_customers():
    results = pd.DataFrame({
        "customerID": ["C1", "C1"],
        "risk_score": [0.9, 0.8]
    })

    with pytest.raises(ValueError, match="duplicados"):
        decide(results)


def test_decision_rejects_risk_outside_zero_and_one():
    results = pd.DataFrame({
        "customerID": ["C1", "C2"],
        "risk_score": [0.9, 1.2]
    })

    with pytest.raises(ValueError, match="entre 0 y 1"):
        decide(results)


def test_decision_sorts_risk_descending():
    results = pd.DataFrame({
        "customerID": ["C1", "C2", "C3"],
        "risk_score": [0.2, 0.9, 0.5]
    })

    ranking, priority = decide(results, selection_fraction=1.0)

    assert ranking["customerID"].tolist() == ["C2", "C3", "C1"]
    assert ranking["risk_score"].is_monotonic_decreasing


# ---------- ACCIONES ----------

def test_actions_assign_correct_action_types():
    priority = pd.DataFrame({
        "customerID": ["C1", "C2", "C3"],
        "risk_score": [0.80, 0.60, 0.30],
        "priority_rank": [1, 2, 3]
    })

    actions = generate_retention_actions(priority)

    assert actions["action_type"].tolist() == [
        "priority_retention_offer",
        "personalized_retention_review",
        "preventive_follow_up"
    ]


def test_actions_are_marked_as_simulated():
    priority = pd.DataFrame({
        "customerID": ["C1", "C2"],
        "risk_score": [0.80, 0.60],
        "priority_rank": [1, 2]
    })

    actions = generate_retention_actions(priority)

    assert len(actions) == 2
    assert actions["action_id"].is_unique
    assert actions["timestamp_utc"].notna().all()
    assert actions["status"].eq("SIMULATED_NOT_SENT").all()


def test_actions_reject_duplicate_customers():
    priority = pd.DataFrame({
        "customerID": ["C1", "C1"],
        "risk_score": [0.80, 0.60],
        "priority_rank": [1, 2]
    })

    with pytest.raises(ValueError, match="duplicados"):
        generate_retention_actions(priority)
