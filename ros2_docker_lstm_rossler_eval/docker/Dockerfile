# Wähle das Basis-Image
FROM ros:humble

# Installiere notwendige Pakete
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch scikit-learn asciichartpy pandas openpyxl
# Arbeitsverzeichnis setzen
WORKDIR /ros2_ws

# Kopiere den Quellcode in das Arbeitsverzeichnis
COPY ../ros2_ws /ros2_ws

# Kopiere die Entry-Point-Skripte
COPY ../entrypoints /entrypoints

# Installiere ROS2 Abhängigkeiten
RUN . /opt/ros/humble/setup.sh && \
    cd /ros2_ws && \
    colcon build

# Standard-Eintragspunkt
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
