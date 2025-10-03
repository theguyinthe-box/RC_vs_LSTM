import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
import numpy as np
import tqdm as tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import time
import json
from py_reservoir_rossler_eval.hyperparam_io import stream_hyperparams_xlsx


# ---- Torch Reservoir Model ----
def powerlaw_random(dim, alpha = 1.75, x_min = 1):
    """Sample numbers from a powerlaw"""
    rands = torch.rand(dim)
    out = x_min * (1 - rands) ** (-1 / (alpha - 1))
    return out

def random_powerlaw_matrix(dim, out_dim = None,
                           alpha = 1.75,
                           x_min = 0.1,
                           normalize_det_to = None,
                           normalize_radius_to = 1,
                           sparsity = 0.0):
    if out_dim is None:
        out_dim = dim
    diagonal = powerlaw_random(out_dim,
                               alpha = alpha,
                               x_min = x_min)
    if normalize_det_to is not None:
        log_det = torch.log(diagonal).sum()
        det_n = torch.exp(log_det / dim)
        diagonal *= normalize_det_to / det_n
    elif normalize_radius_to is not None:
        radius = torch.max(diagonal)
        diagonal *= normalize_radius_to / radius
    # make some entries negative
    negative_mask = torch.rand(out_dim) > 0.5
    diagonal[negative_mask] = -diagonal[negative_mask]
    matrix = torch.diag(diagonal)

    sparsity = min(1, sparsity)
    rot = _make_orthogonal(torch.randn(out_dim, dim) * (torch.rand(out_dim, dim) > sparsity))
    return rot.T @ matrix @ rot

class Reservoir(nn.Module):
    def __init__(self, units,
                 weight = None,
                 bias = True,
                 bias_scale = 0.1,
                 spectral_radius = 1,
                 det_norm = None,
                 activation = torch.nn.Tanh(),
                 leaking_rate = 1,
                 sparsity = 0,
                 powerlaw_alpha = 1.75,
                 **activation_kwargs):  
        super.__init__()
        if weight is None:
            weight = random_powerlaw_matrix(units,
                                            alpha = powerlaw_alpha,
                                            normalize_radius_to = spectral_radius,
                                            normalize_det_to = det_norm,
                                            sparsity = sparsity)
        if bias:
            initial_bias = 2 * torch.rand(units) - 1
            bias_sum = initial_bias.sum()
            initial_bias -= bias_sum / units
            bias = initial_bias * bias_scale
            self.register_buffer("bias", bias)
        else:
            self.register_buffer("bias", torch.zeros(units))
        self.register_buffer("weight", weight)
        # persistent around -2.375, growing attractor after
        self.leaking_rate = leaking_rate
        self.activation = activation(**activation_kwargs)

    def forward(self, x, n_steps = 1):
        with torch.no_grad():
            for _ in range(n_steps):
                y = x @ self.weight + self.bias
                y = torch.nn.functional.tanh(y)
                y = self.activation(y)
        return y

    def train(projection, net, readout, data, device,
          n_epochs = 1, batch_size = 512, lr= 1e-2,
          n_steps = 30):
        """
        Training the readout of the reservoir.
        """
        pbar = tqdm(range(len(data) * n_epochs // batch_size))
        accuracies = []

        optimizer = torch.optim.SGD(readout.parameters(),
                                    lr = lr)

        for epoch in range(n_epochs):
            train_loader = torch.utils.data.DataLoader(data,
                                                       batch_size = batch_size,
                                                       shuffle = True)
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)

                x = projection(x)
                # detach and clone ensures no training until after
                state = net(x, n_steps = n_steps).detach().clone().requires_grad_(True)
                y_hat = readout(state)

                y_out = torch.argmax(y_hat, dim = 1)
                loss = torch.nn.functional.cross_entropy(y_hat, y)
                loss.backward()
                optimizer.step()

                acc = (y_out == y).float().mean()
                accuracies.append(acc.item())
                
                pbar.update(1)
        
        return accuracies

class EdgeReservoirNode(Node):
    def __init__(self):
        super().__init__('edge_reservoir_rossler_node')
        self.set_seed(42)

        # Store hyperparameters
        self.reservoir_params = {
            "input_size": 3,
            "output_size": 3,
            "units": 500,
            "spectral_radius": 1.6,
            "leaking_rate": 0.3,
            "input_scaling": 0.1
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

        # Parameters
        self.train_len = 5000
        self.pred_len = 500

        self.model_path = "/ros2_ws/model.pkl"

        self.hp_path = os.getenv("HYPERPARAMS_XLSX", "/hyperparams.xlsx")
        self.hp_sheet = os.getenv("HYPERPARAMS_SHEET", "grid")
        self.hp_iter = stream_hyperparams_xlsx(self.hp_path, self.hp_sheet)
        
        self.sent_params = False   # already have this
        self.current_cfg = None

        # NEW: control subscriber for NEXT/RETRY
        self.ctrl_sub = self.create_subscription(
            String, 'sweep_control', self.handle_control, 10
        )

        # On startup: pull the first row and apply it to the reservoir parameters
        self.load_next_hyperparams()

        # Publishers / Subscribers
        self.subscription = self.create_subscription(Float64MultiArray, 'rossler_input', self.handle_input, 10)
        self.publisher = self.create_publisher(Float64MultiArray, 'rossler_output', 10)
        self.param_publisher = self.create_publisher(String, 'rossler_hyperparams', 10)
        self.training_time_pub = self.create_publisher(String, 'rossler_training_time', 10)
        self.model_size_pub = self.create_publisher(String, 'size_model', 10)

        self.get_logger().info("Edge reservoir node ready and waiting for Rössler input.")

        self.create_timer(0.5, self.send_hyperparams_once)

    def send_hyperparams_once(self):
        if not self.sent_params:
            payload = {
                **self.reservoir_params,
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
        p = self.reservoir_params
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
            self.model = Reservoir(
                input_size = p["input_size"],
                output_size = p["output_size"],
                units = p["units"],
                spectral_radius = p["spectral_radius"],
                leaking_rate = p["leaking_rate"],
                input_scaling = p["input_scaling"]
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
            self.model = Reservoir(
                input_size = p["input_size"],
                output_size = p["output_size"],
                units = p["units"],
                spectral_radius = p["spectral_radius"],
                leaking_rate = p["leaking_rate"],
                input_scaling = p["input_scaling"]
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
    node = EdgeReservoirNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
