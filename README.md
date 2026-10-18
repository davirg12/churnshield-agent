# CHURNSHIELD-AGENT

## 1. Objetivo

CHURNSHIELD-AGENT es un prototipo de agente inteligente que estima el riesgo de abandono de clientes y prioriza el 15 % con mayor puntuación para acciones de retención simuladas.

El sistema no envía comunicaciones ni ejecuta campañas reales.

## 2. Arquitectura

El agente implementa el ciclo percepción → decisión → acción:

- `app/perception.py`: valida los datos de entrada y los identificadores.
- `app/agent.py`: coordina la ejecución y obtiene las puntuaciones del modelo.
- `app/decision.py`: ordena los clientes por riesgo y selecciona el segmento prioritario.
- `app/actions.py`: genera acciones de retención simuladas.
- `logs/agent_events.jsonl`: registra eventos de percepción, decisión y acción.

La selección utiliza una fracción del 15 % y redondea hacia arriba.

## 3. Tecnologías

- Python
- Pandas
- Scikit-learn
- Pytest
- Git y GitHub

Las dependencias se encuentran en `requirements.txt`.


## 4. Dataset y preparación

Se utiliza el dataset IBM Telco Customer Churn, con 7.043 registros y variables relacionadas con clientes, servicios contratados, antigüedad, cargos y abandono histórico.

Los datos originales se conservan en `data/raw/` y los procesados en `data/processed/`. Los 11 valores vacíos de `TotalCharges`, asociados a clientes con antigüedad cero, se convirtieron a `0.0`.

El identificador `customerID` se utiliza para reconocer a cada cliente, pero se excluye de las características predictivas.

**Limitación temporal:** la etiqueta `Churn` no demuestra por sí sola que un cliente vaya a abandonar el servicio durante los próximos 30 días. Este horizonte sigue pendiente de validación.

## 5. Modelo y resultados

Se evaluó un modelo de regresión logística con procesamiento de variables numéricas y categóricas. La puntuación se utiliza para ordenar clientes por riesgo; no se ha demostrado que esté calibrada como probabilidad.

| Indicador | Resultado |
|---|---:|
| Clientes evaluados | 1.409 |
| Clientes priorizados | 212 |
| Proporción priorizada | 15,05 % |
| Precision@15% | 71,70 % |
| Objetivo de precisión | >= 75 % |
| Estado del objetivo | No cumplido |

La fórmula utilizada es:

`Precision@15% = abandonos reales en el segmento priorizado / total de clientes priorizados`

El resultado no alcanza el objetivo predefinido del 75 %. Además, el conjunto de prueba se inspeccionó durante el desarrollo, por lo que esta evaluación no debe considerarse completamente independiente.

## 6. Instalación

Clona el repositorio e instala las dependencias:

```bash
git clone https://github.com/MauroPrasca29/churnshield-agent.git
cd churnshield-agent
python -m venv .venv
python -m pip install -r requirements.txt
```

## 7. Pruebas automatizadas

Ejecuta desde la raíz del proyecto:

```bash
python -m pytest -v tests/
```

La última ejecución registró 14 pruebas pasadas.

## 8. Evidencias

- `reports/evaluacion_modelo.csv`: resultados del modelo.
- `reports/clientes_prioritarios.csv`: clientes priorizados.
- `reports/matriz_escenarios.csv`: definición de escenarios.
- `reports/resultados_escenarios.csv`: resultados de diez escenarios.
- `reports/evaluacion_metricas_smart.csv`: métricas SMART.
- `reports/test_components_report.json`: informe de pruebas.
- `logs/agent_events.jsonl`: eventos registrados del agente.

## 9. Limitaciones y trabajo futuro

1. Evaluar el modelo con un conjunto de prueba independiente.
2. Obtener datos temporales para validar el horizonte de 30 días.
3. Mejorar Precision@15% usando entrenamiento y validación, sin ajustar el modelo al conjunto de prueba.
4. Evaluar calibración, sesgos y estabilidad.
5. Probar integraciones con CRM antes de cualquier despliegue real.
6. Implementar y probar el manejo de fallos de servicios externos.

## 10. Estado del prototipo

El proyecto incluye los componentes del agente, datos preparados, evaluaciones y pruebas automatizadas. Las acciones son simuladas. La precisión objetivo del 75 % y el horizonte de 30 días siguen pendientes de validación satisfactoria.
