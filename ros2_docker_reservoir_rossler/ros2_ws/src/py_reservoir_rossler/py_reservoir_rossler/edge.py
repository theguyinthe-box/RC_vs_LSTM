import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
import numpy as np
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import set_seed
import joblib
import time
import os
import json

class EdgeReservoirNode(Node):
    def __init__(self):
        super().__init__('edge_reservoir_rossler_node')

        # Publishers / Subscribers
        self.subscription = self.create_subscription(Float64MultiArray, 'rossler_input', self.handle_input, 10)
        self.publisher = self.create_publisher(Float64MultiArray, 'rossler_output', 10)
        self.param_publisher = self.create_publisher(String, 'rossler_hyperparams', 10)
        self.training_time_pub = self.create_publisher(String, 'rossler_training_time', 10)
        self.model_size_pub = self.create_publisher(String, 'size_model', 10)

        self.get_logger().info("Edge reservoir node initialized. Waiting for Rössler input...")

        # Hyperparameters
        self.reservoir_params = {
            "units": 500,
            "spectral_radius": 1.6,
            "leaking_rate": 0.3,
            "input_scaling": 0.1
        }
        self.readout_params = {
            "ridge_alpha": 0.0001
        }

        # Data lengths
        self.train_len = 5000
        self.pred_len = 500

        self.model_path = "/ros2_ws/model.pkl"
        self.sent_params = False

        # Periodically publish hyperparameters until sent once
        self.create_timer(0.5, self.send_hyperparams_once)

    def send_hyperparams_once(self):
        if not self.sent_params:
            params = {**self.reservoir_params, **self.readout_params}
            param_msg = String()
            param_msg.data = json.dumps(params)
            self.param_publisher.publish(param_msg)
            self.get_logger().info("Hyperparameters sent to agent.")
            self.sent_params = True

    def handle_input(self, msg):
        data = np.array(msg.data)
        expected_length = (self.train_len + self.train_len + self.pred_len) * 3
        if data.size != expected_length:
            self.get_logger().error(f"Unexpected input payload length: {data.size}; expected {expected_length}.")
            return

        X = data[:self.train_len * 3].reshape(self.train_len, 3)
        Y = data[self.train_len * 3:self.train_len * 6].reshape(self.train_len, 3)
        test = data[self.train_len * 6:self.train_len * 6 + self.pred_len * 3].reshape(self.pred_len, 3)

        # Load existing model or train a new one
        if os.path.exists(self.model_path):
            self.get_logger().info(f"Found saved model at {self.model_path}. Skipping training.")
            model = joblib.load(self.model_path)
        else:
            self.get_logger().info("No saved model found. Starting training...")

            self.set_seed(42)

            reservoir = Reservoir(
                units=self.reservoir_params["units"],
                sr=self.reservoir_params["spectral_radius"],
                lr=self.reservoir_params["leaking_rate"],
                input_scaling=self.reservoir_params["input_scaling"]
            )
            readout = Ridge(ridge=self.readout_params["ridge_alpha"])
            model = reservoir >> readout

            start_time = time.perf_counter()
            model = model.fit(X, Y, warmup=100, reset=True)
            training_duration = time.perf_counter() - start_time

            joblib.dump(model, self.model_path)

            # Publish model size and training time
            training_time_msg = String()
            training_time_msg.data = f"{training_duration:.6f}"
            self.training_time_pub.publish(training_time_msg)

            model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
            size_msg = String()
            size_msg.data = f"{model_size_mb:.6f}"
            self.model_size_pub.publish(size_msg)

            self.get_logger().info(f"Training completed and model saved. Duration: {training_duration:.3f} s.")
            self.get_logger().info(f"Model size published: {model_size_mb:.6f} MB.")

        # Autoregressive prediction with per-step timing
        predictions = []
        last_input = X[-1].reshape(1, -1)
        timings = []

        for _ in range(test.shape[0]):
            start = time.perf_counter()
            pred = model.run(last_input)
            end = time.perf_counter()
            timings.append(end - start)
            predictions.append(pred.ravel())
            last_input = pred

        avg_time = np.mean(timings)
        min_time = np.min(timings)
        max_time = np.max(timings)

        self.get_logger().info(
            f"Per-step prediction latency — avg: {avg_time*1000:.3f} ms | min: {min_time*1000:.3f} ms | max: {max_time*1000:.3f} ms"
        )

        # Publish predictions and per-step latencies
        pred_array = np.array(predictions)
        msg_out = Float64MultiArray()
        msg_out.data = np.concatenate([pred_array.flatten(), timings]).tolist()
        self.publisher.publish(msg_out)
        self.get_logger().info(f"Published {len(pred_array)} predictions and {len(timings)} per-step latencies to agent.")

    def set_seed(self, seed):
        # Set global random seed for reproducibility
        set_seed(seed)

def main():
    rclpy.init()
    node = EdgeReservoirNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
