set -x
export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
/usr/bin/Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
while [ -z "$(pidof /usr/bin/Xvfb)" ]; do
  echo "INFO: Waiting for Xvfb to start..."
  sleep 1
done
set +x
echo "INFO: Xvfb started."
