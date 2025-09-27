# Choose the base image
FROM ros:humble

# Install required packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*


RUN pip install scikit-learn asciichartpy pandas openpyxl
    
# Make container cuda compatible 
RUN pip install -U "jax[cuda12]"

# Make local reservoirpy package available to Python without pip
# Copy the folder into the image and add it to PYTHONPATH so `import reservoirpy` works
COPY ../reservoirpy /opt/reservoirpy
ENV PYTHONPATH=/opt/reservoirpy:${PYTHONPATH}

# Set working directory
WORKDIR /ros2_ws

# Copy source code into the working directory
COPY ../ros2_ws /ros2_ws

# Copy entrypoint scripts
COPY ../entrypoints /entrypoints

# Build the ROS 2 workspace
RUN . /opt/ros/humble/setup.sh && \
    cd /ros2_ws && \
    colcon build

# Default entrypoint
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
