import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
import jax
import jax.numpy as np
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
import time
import json
from py_reservoir_rossler_eval.logger import Logger

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_reservoir_rossler_node')

        # Parameters
        self.pred_len = 500
        self.shift = 300
        self.train_len = 5000
        self.warmup_runs = 10
        self.eval_runs = 40
        self.pause_sec = 1.0

        # Publishers & Subscribers
        self.publisher = self.create_publisher(Float64MultiArray, 'rossler_input', 10)
        self.subscription = self.create_subscription(Float64MultiArray, 'rossler_output', self.handle_prediction, 10)
        self.subscription_params = self.create_subscription(String, 'rossler_hyperparams', self.handle_hyperparams, 10)
        self.subscription_training_time = self.create_subscription(String, 'rossler_training_time', self.handle_training_time, 10)
        self.subscription_model_size = self.create_subscription(String, 'size_model', self.handle_model_size, 10)
        self.sweep_ctrl_pub = self.create_publisher(String, 'sweep_control', 10)

        # Logger and status flags
        self.logger = None
        self.received_params = False
        self.sent = False
        self.send_time = None
        self.run_count = 0
        self.training_time = None

        self.get_logger().info("Agent started, waiting for hyperparameters from Edge...")

        # Prepare data
        self.prepare_data()

        # Timer: periodic check to decide if data should be sent
        self.timer = self.create_timer(self.pause_sec, self.periodic_check)

    def rossler(self, t, u, a=0.2, b=0.2, c=5.7):
        x, y, z = u
        return [-y - z, x + a * y, b + z * (x - c)]

    def prepare_data(self):
        t_eval = np.arange(0, 200, 0.02)
        sol = solve_ivp(self.rossler, (0, 200), [0.1, 0, 0], t_eval=t_eval)
        data = sol.y.T

        self.X = data[self.shift:self.shift + self.train_len]
        self.Y = data[self.shift + 1:self.shift + self.train_len + 1]
        self.test = data[self.shift + self.train_len:self.shift + self.train_len + self.pred_len]

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.Y_scaled = self.scaler.transform(self.Y)
        self.test_scaled = self.scaler.transform(self.test)

    def handle_hyperparams(self, msg):
        params = json.loads(msg.data)
        if self.logger is None:
            self.logger = Logger("Reservoir", params, self.get_logger())
        self.received_params = True
        self.get_logger().info(f"Hyperparameters received: {params}")

    def handle_training_time(self, msg):
        try:
            self.training_time = float(msg.data)
            if self.logger:
                self.logger.set_training_time(self.training_time)
            self.get_logger().info(f"Training time received: {self.training_time:.3f} seconds")
        except ValueError:
            self.get_logger().error("Error parsing training time")

    def handle_model_size(self, msg):
        model_size = float(msg.data)
        if self.logger:
            self.logger.set_model_size(model_size)
        self.get_logger().info(f"Model size received: {model_size:.6f} MB")

    def periodic_check(self):
        if not self.received_params:
            self.get_logger().info("Waiting for hyperparameters from Edge...")
            return

        if not self.sent and self.run_count < self.warmup_runs + self.eval_runs:
            self.send_data()

        elif self.run_count >= self.warmup_runs + self.eval_runs:
            if self.logger:
                self.logger.summary()

            # Reset before NEXT so that handle_hyperparams creates a fresh Logger instance
            self._reset_for_next_config()

            # Signal NEXT to the Edge node
            try:
                self.sweep_ctrl_pub.publish(String(data="NEXT"))
                self.get_logger().info("Signaled to Edge: NEXT (next hyperparameter row)")
            except Exception as e:
                self.get_logger().error(f"Error sending NEXT: {e}")

            # Do not shut down — wait for new hyperparameters
            return

    def _reset_for_next_config(self):
        # Reset state so the agent waits for new hyperparameters
        self.logger = None
        self.received_params = False
        self.sent = False
        self.send_time = None
        self.run_count = 0
        self.training_time = None
        # Data (X_scaled, Y_scaled, test_scaled) remains unchanged — this is intended
        self.get_logger().info("Agent state reset — waiting for next hyperparameters...")

    def send_data(self):
        if self.publisher.get_subscription_count() > 0:
            msg = Float64MultiArray()
            msg.data = np.concatenate([
                self.X_scaled.flatten(),
                self.Y_scaled.flatten(),
                self.test_scaled.flatten()
            ]).tolist()

            self.publisher.publish(msg)
            self.send_time = time.perf_counter()

            phase = "Warm-up" if self.run_count < self.warmup_runs else "Run"
            self.get_logger().info(f"Rossler data sent ({phase} {self.run_count + 1})")
            self.sent = True
        else:
            self.get_logger().info("Waiting for Edge node...")

    def handle_prediction(self, msg):
        receive_time = time.perf_counter()
        roundtrip_time_s = receive_time - self.send_time if self.send_time is not None else float('nan')

        data = np.array(msg.data, dtype=np.float64)
        n_pred = self.pred_len * 3

        if data.size < n_pred:
            self.get_logger().error(
                f"Received payload too small: {data.size} < expected {n_pred} + timings"
            )
            return

        pred_data = data[:n_pred]
        pred_times_data = data[n_pred:]

        if pred_data.size % 3 != 0:
            self.get_logger().error(f"Prediction data length {pred_data.size} is not a multiple of 3!")
            return

        output_scaled = pred_data.reshape(-1, 3)
        # Back to original space
        output = self.scaler.inverse_transform(output_scaled)

        if output.shape[0] != self.test.shape[0]:
            self.get_logger().error(
                f"Shape mismatch: output has {output.shape[0]} steps, expected {self.test.shape[0]}"
            )
            return

        mse = np.mean((self.test - output) ** 2)

        # Warm-up phase
        if self.run_count < self.warmup_runs and self.logger is not None:
            warmup_index = self.run_count + 1
            avg_pred_ms = np.mean(pred_times_data) * 1e3 if len(pred_times_data) > 0 else float("nan")
            self.logger.log_warmup(warmup_index, roundtrip_time_s)
            self.get_logger().info(
                f"Warm-up {warmup_index} done. "
                f"MSE: {mse:.6f} | "
                f"Roundtrip: {roundtrip_time_s*1000:.2f} ms | "
                f"Avg pred step: {avg_pred_ms:.3f} ms"
            )

        # Evaluation phase
        elif self.logger is not None:
            run_index = self.run_count - self.warmup_runs + 1
            self.logger.log_run(run_index, mse, pred_times_data.tolist(), roundtrip_time_s)
            self.get_logger().info(
                f"Run {run_index} done. MSE: {mse:.6f} | "
                f"Roundtrip: {roundtrip_time_s*1000:.2f} ms | "
                f"Avg pred step: {np.mean(pred_times_data)*1e3:.3f} ms"
            )

        self.run_count += 1
        self.sent = False

def main():
    rclpy.init()
    node = AgentNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
