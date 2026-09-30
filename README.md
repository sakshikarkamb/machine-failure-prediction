# Machine Failure Prediction — AI-Based Predictive Maintenance System

A complete end-to-end Machine Learning and Predictive Maintenance web application designed to anticipate industrial machine failures before they occur. Powered by a tuned Random Forest pipeline, a robust FastAPI backend service, and an interactive Streamlit dashboard.

---

## 📌 Problem Statement

In modern manufacturing, unexpected machinery breakdowns cause costly unplanned downtime, production bottlenecks, and hazardous working conditions. Traditional maintenance strategies either:
- **Run-to-failure (Reactive):** Wait until breakdown, leading to expensive repairs and lost productivity.
- **Scheduled Maintenance (Preventive):** Service equipment at rigid intervals regardless of condition, causing unnecessary part replacements and labor expenses.

**Predictive Maintenance (PdM)** uses real-time operating telemetry (temperatures, speed, torque, tool wear) to predict failures dynamically, allowing maintenance teams to intervene proactively.

---

## 🎯 Project Objective

- Build an industrial predictive maintenance system capable of accurately classifying whether a machine is at risk of failure (`Machine failure = 1`) or operating nominally (`Machine failure = 0`).
- Deploy the untouched trained model through a high-performance **FastAPI** REST backend.
- Provide operators with an intuitive **Streamlit** dashboard showing real-time risk probabilities, parameter summaries, model metrics, feature importances, and historical logs.

---

## 🔮 Future Scope

- **Multi-failure mode classification:** Predict specific failure modes (Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure).
- **IoT Streaming Ingestion:** Connect Kafka or MQTT broker for real-time live sensor streaming.
- **Historical Database:** Persist telemetry and predictions into PostgreSQL / TimescaleDB for continuous drift monitoring.
