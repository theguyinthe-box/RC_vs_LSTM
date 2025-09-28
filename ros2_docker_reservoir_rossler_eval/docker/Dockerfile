# Choose the base image
FROM ros:humble

# Install required packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*


RUN pip install scikit-learn asciichartpy pandas openpyxl cloudpickle
    
# Make container cuda compatible with jax
RUN pip install --no-cache-dir jax
RUN pip install -U "jax[cuda12]"

# Make local reservoirpy package available to Python without pip
# Copy the folder into the image and add it to PYTHONPATH so `import reservoirpy` works

# Set working directory
WORKDIR /ros2_ws

# Copy source code into the working directory
COPY ../ros2_ws /ros2_ws

COPY ../reservoirpy /ros2_ws/src/reservoirpy

# Copy entrypoint scripts
COPY ../entrypoints /entrypoints

# Copy local reservoirpy into the image's site-packages so it's available like an installed package
RUN python3 - <<'PY'
import site, shutil, os
src = '/ros2_ws/src/reservoirpy'
dst = os.path.join(site.getsitepackages()[0], 'reservoirpy')
print('Copying', src, 'to', dst)
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
PY
## Copy the local reservoirpy source into the ROS package so it's importable
COPY ../reservoirpy /ros2_ws/src/py_reservoir_rossler_eval/py_reservoir_rossler_eval/reservoirpy

# Build the ROS 2 workspace
RUN . /opt/ros/humble/setup.sh && \
    cd /ros2_ws && \
    colcon build

# Default entrypoint
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
