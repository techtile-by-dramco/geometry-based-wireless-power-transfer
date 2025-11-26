# A "new" experiment control system
In the new system, ansible is used to manage the fleet of raspberry pi's that are part of the experiment setup.

* **server** contains all ansible playbooks that can be run from your central management machine
* **inventory** contains the ansible hosts inventory (and groups)
* **tiles** contains scripts that are executed on the ansible hosts (i.e. the raspberry pi's) by means of the ansible playbook ```run-script.yaml```.

General usage example:
```bash
 ansible-playbook -i ansible/inventory/hosts.yaml ansible/server/<playbook>.yaml [--limit <host>] [--extra-vars "var1=value1 var2=value2 ..."]
```

Run ```check-uhd.sh``` on tile A05:
```bash
ansible-playbook -i ansible/inventory/hosts.yaml ansible/general/run-script.yaml --extra-vars="script_path=/home/pi/geometry-based-wireless-power-transfer/ansible/tiles/install_uhd/check-uhd.sh sudo=yes sudo_flags=-E" --limit A05
```

A typical experiment can be divided into 3 phases:
1. A general system setup.
   1. Make sure the rpi OS and software is up-to-date: `update-upgrade.yaml`
   2. Pull the experiment repo and check that all experiment-specific requirements are met: `pull-repo.yaml`

2. Run the experiment (can be an iterative process)
   1. Pull the repo (if changes to certain run settings have been made).
   2. Run the experiment scripts on the rpi's (output can be registered asynchronously): `run-script.yaml`
   3. Don't forget to collect your results

3. Clean-up the pi (optional but highly encouraged): 'clean-home.yaml'

TODO: Detailed description
