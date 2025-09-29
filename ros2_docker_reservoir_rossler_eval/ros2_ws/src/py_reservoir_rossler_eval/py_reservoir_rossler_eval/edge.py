import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import jax
import jax.numpy as jnp
import reservoirpy.jax.model
from reservoirpy.jax.nodes import Reservoir, Ridge
from reservoirpy import set_seed
#import joblib
import cloudpickle
import time
import os
import json
from py_reservoir_rossler_eval.hyperparam_io import stream_hyperparams_xlsx

class EdgeReservoirNode(Node):
    def __init__(self):
        super().__init__('edge_reservoir_rossler_node')
        self.set_seed(42)

        jax.default_device(jax.devices()[0])

        # Store hyperparameters
        self.reservoir_params = {
            "units": 500,
            "spectral_radius": 1.6,
            "leaking_rate": 0.3,
            "input_scaling": 0.1
        }
        self.readout_params = {
            "ridge_alpha": 0.0001
        }

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
        self.subscription = self.create_subscription(Float32MultiArray, 'rossler_input', self.handle_input, 10)
        self.publisher = self.create_publisher(Float32MultiArray, 'rossler_output', 10)
        self.param_publisher = self.create_publisher(String, 'rossler_hyperparams', 10)
        self.training_time_pub = self.create_publisher(String, 'rossler_training_time', 10)
        self.model_size_pub = self.create_publisher(String, 'size_model', 10)

        self.get_logger().info("Edge reservoir node ready and waiting for Rössler input.")

        self.create_timer(0.5, self.send_hyperparams_once)

    def load_next_hyperparams(self):
        try:
            row = next(self.hp_iter)
        except StopIteration:
            self.get_logger().info("🎉 No further hyperparameter rows in the Excel file.")
            return

        # Map Excel columns directly onto the existing dicts.
        # Keep fallbacks in case some columns are missing.
        self.reservoir_params = {
            "units":           int(row.get("units", self.reservoir_params.get("units", 500))),
            "spectral_radius": float(row.get("spectral_radius", self.reservoir_params.get("spectral_radius", 1.6))),
            "leaking_rate":    float(row.get("leaking_rate", self.reservoir_params.get("leaking_rate", 0.3))),
            "input_scaling":   float(row.get("input_scaling", self.reservoir_params.get("input_scaling", 0.1))),
        }
        self.readout_params = {
            "ridge_alpha":     float(row.get("ridge_alpha", self.readout_params.get("ridge_alpha", 0.0001)))
        }
        self.sent_params = False  # For send_hyperparams_once()

        # Remove old model to force retraining:
        try:
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
                self.get_logger().info(f"Old model deleted: {self.model_path}")
        except Exception as e:
            self.get_logger().warn(f"Could not delete model: {e}")

    def handle_control(self, msg: String):
        cmd = msg.data.strip().upper()
        if cmd == "NEXT":
            self.get_logger().info("➡️  NEXT received — loading next hyperparameter row.")
            self.load_next_hyperparams()
        elif cmd == "RETRY":
            self.get_logger().info("🔁 RETRY received — re-sending current hyperparameters.")
            self.sent_params = False  # will be sent again on the next timer tick
        else:
            self.get_logger().warn(f"Unknown control command: {cmd}")

    def send_hyperparams_once(self):
        # Nothing to send yet?
        if self.sent_params:
            return

        # Check if the Agent subscriber is ready
        sub_count = self.param_publisher.get_subscription_count()
        if sub_count <= 0:
            # Agent not subscribed yet — try again later
            self.get_logger().info("Waiting for Agent subscriber on 'rossler_hyperparams'...")
            return

        # Now publish safely
        params = {**self.reservoir_params, **self.readout_params}
        param_msg = String()
        param_msg.data = json.dumps(params)
        self.param_publisher.publish(param_msg)
        self.sent_params = True
        self.get_logger().info(f"Hyperparameters sent to Agent (subs={sub_count}).")

    def handle_input(self, msg):
        self.set_seed(42)
        data = jnp.array(msg.data)
        expected_length = (self.train_len + self.train_len + self.pred_len) * 3
        if data.size != expected_length:
            self.get_logger().error(f"Unexpected input data length: {data.size}, expected: {expected_length}")
            return

        X = data[:self.train_len * 3].reshape(self.train_len, 3)
        Y = data[self.train_len * 3:self.train_len * 6].reshape(self.train_len, 3)
        test = data[self.train_len * 6:self.train_len * 6 + self.pred_len * 3].reshape(self.pred_len, 3)

        # Load or train model
        if os.path.exists(self.model_path):
            self.get_logger().info("Model loaded. No training required.")
            #model = joblib.load(self.model_path)
            with open(self.model_path, "rb") as f:
                 model = cloudpickle.load(f)

        else:
            self.get_logger().info("No model found. Starting training...")

            reservoir = Reservoir(
                units=self.reservoir_params["units"],
                sr=self.reservoir_params["spectral_radius"],
                lr=self.reservoir_params["leaking_rate"],
                input_scaling=self.reservoir_params["input_scaling"]
            )
            readout = Ridge(ridge=self.readout_params["ridge_alpha"])
            model = reservoir >> readout

            start_time = time.perf_counter()
            model = model.fit(X, Y, warmup=100, workers=-1)#, reset=True) deprecated reset in respy0.4.1 current resets by default
            training_duration = time.perf_counter() - start_time

            #joblib.dump(model, self.model_path)
            self.get_logger().info(f"saving model")
            model_host = jax.device_get(model)    # convert DeviceArrays -> numpy arrays on host
            with open(self.model_path, "wb") as f:
                cloudpickle.dump(model_host, f)

            # Measure and publish model size and training time
            training_time_msg = String()
            training_time_msg.data = f"{training_duration:.6f}"
            self.training_time_pub.publish(training_time_msg)
            model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
            size_msg = String()
            size_msg.data = f"{model_size_mb:.6f}"
            self.model_size_pub.publish(size_msg)
            self.get_logger().info(f"Training completed and model saved. Duration: {training_duration:.3f} seconds")
            self.get_logger().info(f"Model size published: {model_size_mb:.6f} MB")

        # Autoregressive prediction & timing
        predictions = jnp.array([])
        last_input = X[-1].reshape(1, -1)
        timings = jnp.array([])

        for _ in range(test.shape[0]):
            start = time.perf_counter()
            pred = model.run(last_input, workers=-1)
            end = time.perf_counter()
            timings = jnp.append(timings, end - start)
            predictions = jnp.append(predictions, pred.ravel())
            last_input = pred

        
        avg_time = (end-start)/test.shape[0] #jnp.mean(timings)
        min_time = jnp.min(timings)
        max_time = jnp.max(timings)

        self.get_logger().info(
            f"Prediction time (per step): avg {avg_time*1000:.3f} ms | min {min_time*1000:.3f} ms | max {max_time*1000:.3f} ms"
        )

        # Publish predictions + timings
        pred_array = jnp.array(predictions)
        msg_out = Float32MultiArray()
        msg_out.data = jnp.concatenate([pred_array.flatten(), timings]).tolist()
        self.publisher.publish(msg_out)
        self.get_logger().info(f"Published {len(pred_array)} predictions and {len(timings)} per-step latencies to Agent.")

    def set_seed(self, seed):
        set_seed(seed)

def main():
    rclpy.init()
    node = EdgeReservoirNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
