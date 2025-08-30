import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import time
import json

# ---- LSTM Model Definition (no defaults) ----
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [B, T, input_size]
        out, _ = self.lstm(x)
        # take the last time step
        return self.fc(out[:, -1, :])

class EdgeLSTMNode(Node):
    def __init__(self):
        super().__init__('edge_lstm_rossler_node')

        # ======= core parameters =======
        self.model_params = {
            "input_size": 3,
            "hidden_size": 64,
            "output_size": 3,
            "num_layers": 3,
        }
        self.training_params = {
            "sequence_length": 20,
            "lr": 0.0005,
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

        # Publishers / Subscribers
        self.subscription = self.create_subscription(Float64MultiArray, 'rossler_input', self.handle_input, 10)
        self.publisher = self.create_publisher(Float64MultiArray, 'rossler_output', 10)
        self.param_pub = self.create_publisher(String, 'rossler_hyperparams', 10)
        self.training_time_pub = self.create_publisher(String, 'rossler_training_time', 10)
        self.model_size_pub = self.create_publisher(String, 'size_model', 10)

        self.get_logger().info("Edge LSTM node ready and waiting for Rossler input.")

        self.sent_params = False
        self.create_timer(0.5, self.send_hyperparams_once)

    def send_hyperparams_once(self):
        if not self.sent_params:
            payload = {
                **self.model_params,
                **self.training_params,
                **self.data_params,
            }
            msg = String()
            msg.data = json.dumps(payload)
            self.param_pub.publish(msg)
            self.get_logger().info(f"Hyperparameters sent to Agent: {payload}")
            self.sent_params = True

    def create_sequences(self, X, Y, seq_len):
        """
        Expects already scaled X and Y.
        X: [train_len, input_size], Y: [train_len, output_size]
        Samples: (X[i:i+T], Y[i+T-1]) with T = seq_len
        """
        x_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            x_seq.append(X[i:i+seq_len])
            y_seq.append(Y[i+seq_len-1])
        return np.array(x_seq), np.array(y_seq)

    def send_model_size(self):
        path = self.runtime_params["model_path"]
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            msg = String()
            msg.data = f"{size_mb:.6f}"
            self.model_size_pub.publish(msg)
            self.get_logger().info(f"Model size published: {size_mb:.6f} MB")

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

        data = np.array(msg.data, dtype=np.float64)

        expected_length = (train_len*2 + pred_len) * input_size
        if data.size != expected_length:
            self.get_logger().error(f"Unexpected input length: {data.size}, expected: {expected_length}")
            return

        # Agent sends scaled X, Y, test (StandardScaler)
        X_scaled = data[:train_len*input_size].reshape(train_len, input_size)
        Y_scaled = data[train_len*input_size:train_len*2*input_size].reshape(train_len, output_size)
        test_scaled = data[train_len*2*input_size:].reshape(pred_len, input_size)

        # validate sequence_length
        if seq_len < 1 or seq_len > train_len - 1:
            self.get_logger().error(f"Invalid sequence_length={seq_len}. Must be in [1, {train_len-1}].")
            return

        # load model or train from scratch
        if os.path.exists(model_path):
            # 1) build the model first, THEN load weights
            self.model = LSTMModel(
                input_size=p["input_size"],
                hidden_size=p["hidden_size"],
                output_size=p["output_size"],
                num_layers=p["num_layers"]
            ).to(self.device)

            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.get_logger().info(f"Model loaded from {model_path}. No training required.")
        else:
            # 2) training: set seeds first → then instantiate model
            torch.manual_seed(42)
            np.random.seed(42)

            self.get_logger().info(
                "No model found. Starting training with params: "
                f"hidden_size={p['hidden_size']}, num_layers={p['num_layers']}, "
                f"lr={t['lr']}, sequence_length={t['sequence_length']}, "
                f"batch_size={t['batch_size']}, epochs={t['epochs']}"
            )

            # build model now (deterministic initial weights)
            self.model = LSTMModel(
                input_size=p["input_size"],
                hidden_size=p["hidden_size"],
                output_size=p["output_size"],
                num_layers=p["num_layers"]
            ).to(self.device)

            # build sequences
            X_seq, Y_seq = self.create_sequences(X_scaled, Y_scaled, seq_len)
            X_tensor = torch.tensor(X_seq, dtype=torch.float32)
            Y_tensor = torch.tensor(Y_seq, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, Y_tensor)
            loader = DataLoader(dataset, batch_size=self.training_params["batch_size"], shuffle=True, drop_last=False)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_params["lr"])
            loss_fn = nn.MSELoss()

            self.model.train()
            start_time = time.perf_counter()

            for epoch in range(self.training_params["epochs"]):
                epoch_loss = 0.0
                for xb, yb in loader:
                    xb = xb.to(self.device)  # [B, T, input_size]
                    yb = yb.to(self.device)  # [B, output_size]
                    optimizer.zero_grad()
                    pred = self.model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * xb.size(0)

                avg_loss = epoch_loss / len(dataset)
                self.get_logger().info(f"Epoch {epoch+1}/{self.training_params['epochs']} - train MSE: {avg_loss:.6e}")

            training_time = time.perf_counter() - start_time
            torch.save(self.model.state_dict(), model_path)
            self.send_model_size()
            self.get_logger().info(f"Training completed and model saved. Duration: {training_time:.3f} seconds")

            # publish training time
            msg_time = String()
            msg_time.data = f"{training_time:.6f}"
            self.training_time_pub.publish(msg_time)

        # --------- Autoregressive prediction (scaled) ---------
        self.model.eval()

        # initial sequence: last T steps from (scaled) training input
        last_seq = X_scaled[-seq_len:]  # [T, input_size]
        last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, T, input_size]

        predictions_scaled = []
        step_times = []

        with torch.no_grad():
            for _ in range(pred_len):
                t0 = time.perf_counter()
                pred_scaled = self.model(last_seq)  # [1, output_size]
                t1 = time.perf_counter()

                step_times.append(t1 - t0)
                predictions_scaled.append(pred_scaled.cpu().numpy().flatten())

                # autoregressively update (here: input_size == output_size)
                last_seq = torch.cat([last_seq[:, 1:], pred_scaled.unsqueeze(1)], dim=1)

        predictions_scaled = np.array(predictions_scaled, dtype=np.float64)  # [pred_len, output_size]
        step_times = np.array(step_times, dtype=np.float64)                  # [pred_len]

        # Publish: [predictions_scaled, timings]
        msg_out = Float64MultiArray()
        msg_out.data = np.concatenate([predictions_scaled.flatten(), step_times]).tolist()
        self.publisher.publish(msg_out)

        self.get_logger().info(
            f"{len(predictions_scaled)} predictions sent. "
            f"Avg pred step: {np.mean(step_times)*1e3:.3f} ms | "
            f"Device: {self.device}"
        )

def main():
    rclpy.init()
    node = EdgeLSTMNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
