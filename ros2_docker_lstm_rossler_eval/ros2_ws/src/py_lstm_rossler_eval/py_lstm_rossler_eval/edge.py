import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import time
import random
import json
from py_lstm_rossler_eval.hyperparam_io import stream_hyperparams_xlsx


# ---- LSTM Model Definition (ohne Defaults!) ----
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [B, T, input_size]
        out, _ = self.lstm(x)
        # letzter Zeitschritt
        return self.fc(out[:, -1, :])

class EdgeLSTMNode(Node):
    def __init__(self):
        super().__init__('edge_lstm_rossler_node')

        # ======= zentrale Parameter =======
        self.model_params = {
            "input_size": 3,
            "hidden_size": 128,
            "output_size": 3,
            "num_layers": 1,
        }
        self.training_params = {
            "sequence_length": 50,
            "lr": 0.0001,
            "epochs": 50,
            "batch_size": 64,
        }
        self.data_params = {
            "train_len": 5000,
            "pred_len": 500,
        }
        self.runtime_params = {
            "model_path": "/ros2_ws/model.pt",
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hp_path = os.getenv("HYPERPARAMS_XLSX", "/hyperparams.xlsx")
        self.hp_sheet = os.getenv("HYPERPARAMS_SHEET", "grid")
        self.hp_iter = stream_hyperparams_xlsx(self.hp_path, self.hp_sheet)
        self.current_cfg = None

        # NEW: Steuer-Subscriber für NEXT/RETRY
        self.ctrl_sub = self.create_subscription(
            String, 'sweep_control', self.handle_control, 10
        )

        self.load_next_hyperparams()

        # Publisher / Subscriber
        self.subscription = self.create_subscription(Float64MultiArray, 'rossler_input', self.handle_input, 10)
        self.publisher = self.create_publisher(Float64MultiArray, 'rossler_output', 10)
        self.param_pub = self.create_publisher(String, 'rossler_hyperparams', 10)
        self.training_time_pub = self.create_publisher(String, 'rossler_training_time', 10)
        self.model_size_pub = self.create_publisher(String, 'size_model', 10)

        self.get_logger().info("Edge LSTM node ready and waiting for Rössler input.")

        self.sent_params = False
        self.create_timer(0.5, self.send_hyperparams_once)

    def send_hyperparams_once(self):
        if self.sent_params:
            return
        subs = self.param_pub.get_subscription_count()
        if subs <= 0:
            self.get_logger().info("Waiting for Agent subscriber on 'rossler_hyperparams'...")
            return

        payload = {
            "model": "LSTM",
            "hidden_size": self.model_params["hidden_size"],
            "num_layers":  self.model_params["num_layers"],
            "lr":          self.training_params["lr"],
            "batch_size":  self.training_params["batch_size"],
            "sequence_length": self.training_params["sequence_length"],
            "epochs":      self.training_params["epochs"],
            "dropout":     float(self.training_params.get("dropout", 0.0)),
        }
        self.param_pub.publish(String(data=json.dumps(payload)))
        self.sent_params = True
        self.get_logger().info(f"Hyperparameters sent to Agent (subs={subs}): {payload}")


    def create_sequences(self, X, Y, seq_len):
        """
        Erwartet bereits skalierte X, Y (float32).
        X: [train_len, input_size], Y: [train_len, output_size]
        Samples: (X[i:i+T], Y[i+T-1]) mit T=seq_len
        """
        x_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            x_seq.append(X[i:i+seq_len])
            y_seq.append(Y[i+seq_len-1])
        return np.asarray(x_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


    def send_model_size(self):
        path = self.runtime_params["model_path"]
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            msg = String()
            msg.data = f"{size_mb:.6f}"
            self.model_size_pub.publish(msg)
            self.get_logger().info(f"Model size sent: {size_mb:.6f} MB")

    # ---------------- Main handler ----------------
    def handle_input(self, msg: Float64MultiArray):
        p = self.model_params
        t = self.training_params
        d = self.data_params
        model_path = self.runtime_params["model_path"]

        input_size = p["input_size"]
        output_size = p["output_size"]
        train_len = d["train_len"]
        pred_len = d["pred_len"]
        seq_len = t["sequence_length"]

        data = np.array(msg.data, dtype=np.float32)

        expected_length = (train_len*2 + pred_len) * input_size
        if data.size != expected_length:
            self.get_logger().error(f"Unexpected input length: {data.size}, expected: {expected_length}")
            return

        # Agent sendet skaliertes X, Y, test (StandardScaler)
        X_scaled = data[:train_len*input_size].reshape(train_len, input_size)
        Y_scaled = data[train_len*input_size:train_len*2*input_size].reshape(train_len, output_size)
        test_scaled = data[train_len*2*input_size:].reshape(pred_len, input_size)

        # sequence_length validieren
        if seq_len < 1 or seq_len > train_len - 1:
            self.get_logger().error(f"Invalid sequence_length={seq_len}. Must be in [1, {train_len-1}].")
            return

        # Modell laden oder trainieren
        if os.path.exists(model_path):
            # 1) Modell jetzt erst erzeugen, DANN laden
            self.model = LSTMModel(
                input_size=p["input_size"],
                hidden_size=p["hidden_size"],
                output_size=p["output_size"],
                num_layers=p["num_layers"]
            ).to(self.device)

            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.get_logger().info(f"Model loaded from {model_path}. No training required.")

        else:
            # 2) Training: Seeds setzen → Sequenzen bauen → Modell instanziieren → trainieren
            torch.manual_seed(42)
            np.random.seed(42)

            self.get_logger().info(
                "No model found. Starting training with params: "
                f"hidden_size={p['hidden_size']}, num_layers={p['num_layers']}, "
                f"lr={t['lr']}, sequence_length={seq_len}, "
                f"batch_size={t['batch_size']}, epochs={t['epochs']}"
            )

            # --- Sequenzen (float32) ---
            X_seq, Y_seq = self.create_sequences(X_scaled, Y_scaled, seq_len)
            # (Sicherheitsnetz, falls upstream mal dtype ändert)
            X_seq = X_seq.astype(np.float32, copy=False)
            Y_seq = Y_seq.astype(np.float32, copy=False)

            X_tensor = torch.tensor(X_seq, dtype=torch.float32)
            Y_tensor = torch.tensor(Y_seq, dtype=torch.float32)

            dataset = TensorDataset(X_tensor, Y_tensor)
            loader = DataLoader(
                dataset,
                batch_size=self.training_params["batch_size"],
                shuffle=True,
                drop_last=False,
                pin_memory=(self.device.type == "cuda"),
                num_workers=0  # ROS-Umgebung: stabil/portabel
            )

            # --- Modell ---
            self.model = LSTMModel(
                input_size=p["input_size"],
                hidden_size=p["hidden_size"],
                output_size=p["output_size"],
                num_layers=p["num_layers"]
            ).to(self.device)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_params["lr"])
            loss_fn = nn.MSELoss()

            # --- Training ---
            self.model.train()
            start_time = time.perf_counter()

            for epoch in range(self.training_params["epochs"]):
                epoch_loss = 0.0
                n = 0
                for xb, yb in loader:
                    # exakt wie im Script: float32 + Device
                    xb = xb.to(self.device, non_blocking=True).float()  # [B, T, input_size]
                    yb = yb.to(self.device, non_blocking=True).float()  # [B, output_size]
                    optimizer.zero_grad()
                    out = self.model(xb)                # [B, output_size]
                    loss = loss_fn(out, yb)
                    loss.backward()
                    optimizer.step()
                    bsz = xb.size(0)
                    epoch_loss += loss.item() * bsz
                    n += bsz

                avg_loss = epoch_loss / max(n, 1)
                self.get_logger().info(f"Epoch {epoch+1}/{self.training_params['epochs']} - train MSE: {avg_loss:.6e}")

            training_time = time.perf_counter() - start_time

            # --- Speichern + Metadaten publizieren ---
            torch.save(self.model.state_dict(), model_path)
            self.send_model_size()
            self.get_logger().info(f"Training completed and model saved. Duration: {training_time:.3f} seconds")

            msg_time = String()
            msg_time.data = f"{training_time:.6f}"
            self.training_time_pub.publish(msg_time)


        # --------- Autoregressive Vorhersage (skaliert) ---------
        self.model.eval()

        predictions_scaled = []
        step_times = []

        # Startsequenz: letzte T Schritte aus dem (skalierten) Trainingseingang
        last_seq = X_scaled[-seq_len:]  # [T, input_size]
        last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, T, input_size]

        with torch.no_grad():
            for _ in range(pred_len):
                t0 = time.perf_counter()
                pred_scaled = self.model(last_seq)  # [1, output_size]
                t1 = time.perf_counter()

                step_times.append(t1 - t0)
                predictions_scaled.append(pred_scaled.cpu().numpy().flatten())

                # autoregressiv updaten (hier: input_size == output_size)
                last_seq = torch.cat([last_seq[:, 1:], pred_scaled.unsqueeze(1)], dim=1)

        predictions_scaled = np.array(predictions_scaled, dtype=np.float64)  # [pred_len, output_size]
        step_times = np.array(step_times, dtype=np.float64)                  # [pred_len]

        # Sende: [vorhersagen_skal., timings]
        msg_out = Float64MultiArray()
        msg_out.data = np.concatenate([predictions_scaled.flatten(), step_times]).tolist()
        self.publisher.publish(msg_out)

        self.get_logger().info(
            f"{len(predictions_scaled)} predictions sent. "
            f"Avg pred step: {np.mean(step_times)*1e3:.3f} ms | "
            f"Device: {self.device}"
        )


    def load_next_hyperparams(self):
        try:
            row = next(self.hp_iter)
        except StopIteration:
            self.get_logger().info("🎉 Keine weiteren Hyperparameterzeilen (LSTM).")
            # Optional: DONE an Agent signalisieren
            if self.param_pub.get_subscription_count() > 0:
                self.param_pub.publish(String(data=json.dumps({"done": True, "model": "LSTM"})))
            return

        # Erwartete Spalten: hidden_size, num_layers, lr, sequence_length
        lp = self.model_params
        tp = self.training_params
        def _get(key, cast, default):
            try:
                v = row.get(key, default)
                return default if v is None or v=="" else cast(v)
            except Exception:
                return default

        self.model_params["hidden_size"] = _get("hidden_size", int, lp["hidden_size"])
        self.model_params["num_layers"]  = _get("num_layers",  int, lp["num_layers"])
        self.training_params["lr"]       = _get("lr",          float, tp["lr"])
        self.training_params["sequence_length"] = _get("sequence_length", int, tp["sequence_length"])

        # Modell-Datei löschen, damit neu trainiert wird
        try:
            if os.path.exists(self.runtime_params["model_path"]):
                os.remove(self.runtime_params["model_path"])
                self.get_logger().info(f"Altes LSTM-Modell gelöscht: {self.runtime_params['model_path']}")
        except Exception as e:
            self.get_logger().warn(f"Konnte LSTM-Modell nicht löschen: {e}")

        self.sent_params = False  # sorgt dafür, dass send_hyperparams_once neu publisht

    def handle_control(self, msg: String):
        cmd = msg.data.strip().upper()
        if cmd == "NEXT":
            self.get_logger().info("➡️  NEXT empfangen – nächste LSTM-Hyperparameterzeile laden.")
            self.load_next_hyperparams()
        elif cmd == "RETRY":
            self.get_logger().info("🔁 RETRY empfangen – aktuelle Hyperparameter erneut senden.")
            self.sent_params = False
        else:
            self.get_logger().warn(f"Unbekannter Control-Command: {cmd}")


def main():
    rclpy.init()
    node = EdgeLSTMNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
