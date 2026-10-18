#!/usr/bin/env bash
# Targeted ROS Humble TF2 repair for Ubuntu Jammy/amd64.
# Fix: https://github.com/ros2/geometry2/pull/990
# Default: check availability and simulate. Use sudo bash ... --install to apply.
set -euo pipefail

mode=${1:---check}
if [[ $# -gt 1 || ( "$mode" != --check && "$mode" != --install ) ]]; then
  echo "Usage: bash $0 [--check|--install]" >&2
  exit 2
fi
if [[ "$mode" == --install && $EUID -ne 0 ]]; then
  echo "Installation requires sudo; run: sudo bash $0 --install" >&2
  exit 1
fi
source /etc/os-release
if [[ "$ID" != ubuntu || "$VERSION_CODENAME" != jammy || $(dpkg --print-architecture) != amd64 ]]; then
  echo 'This repair is specific to Ubuntu 22.04 amd64.' >&2
  exit 1
fi

packages=(ros-humble-tf2 ros-humble-tf2-ros)
targets=(
  ros-humble-tf2=0.25.24-1jammy.20260915.203746
  ros-humble-tf2-ros=0.25.24-1jammy.20260915.204410
)
needs_upgrade=false
for package in "${packages[@]}"; do
  installed=$(dpkg-query -W -f='${Version}' "$package")
  echo "$package: $installed"
  if dpkg --compare-versions "$installed" lt 0.25.24; then
    needs_upgrade=true
  fi
done
if [[ "$needs_upgrade" == false ]]; then
  echo 'Both packages already contain the fix. Restart ROS processes to load them.'
  exit 0
fi

# Isolated metadata: never add testing to /etc/apt or replace the system lists.
# Keep this directory for troubleshooting; a new one is created on every run.
repair_dir=$(mktemp -d /tmp/rviz-tf2-repair.XXXXXX)
chmod 755 "$repair_dir"
mkdir -p "$repair_dir/lists/partial" "$repair_dir/archives/partial"
echo "Repair metadata: $repair_dir"
cat > "$repair_dir/testing.list" <<'EOF'
deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2-testing/ubuntu jammy main
EOF
apt_options=(
  -o "Dir::Etc::sourcelist=$repair_dir/testing.list"
  -o Dir::Etc::sourceparts=-
  -o "Dir::State::lists=$repair_dir/lists"
  -o "Dir::Cache::archives=$repair_dir/archives"
  -o "Dir::Cache::pkgcache=$repair_dir/pkgcache.bin"
  -o "Dir::Cache::srcpkgcache=$repair_dir/srcpkgcache.bin"
  -o Acquire::Retries=2
  -o Acquire::http::Timeout=30
)
apt-get "${apt_options[@]}" update --error-on=any
apt-get "${apt_options[@]}" --simulate --no-remove install --only-upgrade "${targets[@]}"
if [[ "$mode" == --install ]]; then
  # APT asks for confirmation and verifies downloaded packages against signed metadata.
  apt-get "${apt_options[@]}" --no-remove install --only-upgrade "${targets[@]}"
  for package in "${packages[@]}"; do
    installed=$(dpkg-query -W -f='${Version}' "$package")
    dpkg --compare-versions "$installed" ge 0.25.24
    echo "$package: $installed"
  done
  echo 'Upgrade verified. Restart all ROS processes, including RViz and Nav2.'
fi
