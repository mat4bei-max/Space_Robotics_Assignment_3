# Activate with: source env.bash (in a fresh Bash terminal).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Use: source ${0}" >&2
    exit 1
fi
_space_ws="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
if [[ -n "${SPACE_ROBOTICS_WORKSPACE:-}" && "$SPACE_ROBOTICS_WORKSPACE" != "$_space_ws" ]]; then
    echo "Open a fresh terminal before switching ROS workspaces." >&2
    unset _space_ws
    return 1
fi
if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "ROS Humble is missing: /opt/ros/humble/setup.bash" >&2
    unset _space_ws
    return 1
fi
source /opt/ros/humble/setup.bash || return 1
if [[ -f "$_space_ws/.venv/bin/activate" ]]; then
    source "$_space_ws/.venv/bin/activate" || return 1
fi
# Load only this overlay; setup.bash may contain previous workspace chains.
if [[ -f "$_space_ws/install/local_setup.bash" ]]; then
    source "$_space_ws/install/local_setup.bash" || return 1
else
    echo "Workspace is not built yet; ROS base environment is active." >&2
fi
export ROS_LOCALHOST_ONLY=1
export LIBGL_ALWAYS_SOFTWARE=1
export SPACE_ROBOTICS_WORKSPACE="$_space_ws"
cd -- "$_space_ws" || return 1
unset _space_ws
