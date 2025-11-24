# A "new" experiment control system
In the new system, ansible is used to manage the fleet of raspberry pi's that are part of the experiment setup.

A typical experiment can be divided into 3 phases:
+A general system setup.
-+Make sure the rpi OS and software is up-to-date.
-+Pull the experiment repo and check that all experiment-specific requirements are met.

+Run the experiment (can be an iterative process)
-+Pull the repo (if changes to certain run settings have been made).
-+Run the experiment scripts on the rpi's (output can be registered asynchronously)
-+Don't forget to collect your results

+Clean-up the pi (optional but highly encouraged)

TODO: Detailed description