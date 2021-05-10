# DriveTruth
## 1. Getting Started
Install the latest version of CARLA, either from [their website](https://carla.readthedocs.io/en/0.9.11/) or [directly from the releases page on GitHub](https://github.com/carla-simulator/carla).  Ensure you meet the system requirements and have plenty of hard disk space to collect data.

Ensure you have the appropriate version of Python installed.  At the time of writing, CARLA is meant to work with Python 3.7.  You can use a service like anaconda to ensure you have the correct version of Python.

Finally, install relevant packages for your Python installation, namely numpy, PIL, and OpenCV (cv2).  If any "missing package" errors come up when you try to run, simply install that package.

Finally, within your CARLA installation, create a new folder within the PythonAPI folder and name it whatever you want.  Place the ``collect_dt_data.py`` and ``carla_vehicle_annotator.py`` files in that folder.

To run, simply start the CARLA simulator with one shell [(see here for detailed instructions)](https://carla.readthedocs.io/en/0.9.11/start_quickstart/#running-carla) and in another shell, run ``collect_dt_data.py``.  Parameters are listed below.

## 2. Parameters

The following parameters are available to pass into ``collect_dt_data.py``.
### ``--host``
The IP of the host server (your CARLA instance), by default 127.0.0.1.  Don't change this unless you're sure that CARLA is running on a different IP!

### ``--port, -p``

The TCP port to listen to for CARLA.  Default is 2000.  Again, don't change this unless you're sure CARLA is broadcasting to this port!

### ``--number-of-vehicles, -n``

The number of vehicles the simulation will attempt to spawn, provided the map has enough spawn points.  Defaults to 100.

### ``--number-of-walkers, -w``

Number of pedestrians the simulation will attempt to spawn.  Defaults to 50.

### ``--tm_port, -tm_p``

The port to communicate with CARLA's traffic manager.  Defaults to 8000.  Don't change this unless you're sure Traffic Manager is running on a different port!

### ``--out, -o``

The output location of the generated dataset.  Defaults to ``/output/<timestamp>/`` in the same directory as ``collect_dt_data.py``.

### ``--lidar``

If ``--lidar y`` (case sensitive) is given, the application will export a 3D pointmap of the LIDAR data.

### ``--fps``
FPS at which we run the data collection at.  Every *fps* frames, the simulation will output the data collected in the frame.  Defaults to 5.

### ``--time_limit``
How many seconds we should run a simulation for before resetting.


## 3. In-code Modifications
### Number of segments
The ``segment_num`` and ``target_segment_limit`` variables control how many runs are generated.  The ``time_limit`` parameter controls how long a single run is.  We take ``segment_num`` runs of each map type, alternating between them, until we reach ``target_segment_limit`` for each map type.  One can modify these variables as desired to collect as much or as little data as one wants.

### Maps
By default, maps are classified as "residential" and "highway".  One can reclassify the maps as they wish, passing in different map types through the ``map_types`` variable and defining an array with the map names in them within the ``run_simulation_instance`` function.  Alternatively, one can remove the map name logic or simply pass in an invalid map type.  In this case, DriveTruth will select from a pool of all available maps (automatically detecting user-made maps).

If you add a map and are using map types, be sure to add it to the relevant list under the map type logic so that it will be utilized by the code.

### Weather/Time of Day

By default, the weather is chosen from a list of presets, contained in the variable ``weather_presets``.  Comments above this variable provide tips on how to redefine a custom weather distribution and time of day.

### Cars and Pedestrians

By default, vehicles are generated from the ``blueprints`` variable (which gets all blueprints classified under ``vehicle.*``, which by default is all vehicles in CARLA but will also include any user-created vehicles under that classification).  ``blueprints_p`` does the same for pedestrians.  If one has a specific model in mind, one can alter which blueprints are grabbed to obtain the desired one.

### Tracking new objects
To track a new object, follow the template provided by the traffic signs and traffic lights, labelled by ``TRACKING STEP`` in the code.

First, figure out the numeric semantic tag of the object(s) you would like to track.  The paper lists them, but you can also find a most up-to-date list [in the CARLA docs here](https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera).  If you define a new semantic tag within CARLA, you can pass it in as well.

For tracking step 1, we get the 3D bounding boxes of the object.  simply call ``point_cloud_to_3d_bbox`` on sem_img, optionally specifying a debug filename and debug parameter, but passing in your semantic tags of the object(s) you wish to track in an array.  For example, for traffic signs, the array [12] is passed, with 12 being the semantic tag for traffic lights.

For tracking step 2, follow the code in tracking step 2, but substitute the bbox list with the bbox list you got in tracking step 1, adding the keys to a list and passing it into the snap processing.  When calling the semantic auto annotate function, be sure to change ``gt_class`` to the class you want this object classified as.

Finally, for tracking step 3, simply merge the results into the final dictionary.
## 4. CARLA Modifications

Because DriveTruth is built on top of CARLA, modifications to CARLA are compatible with DriveTruth.  This means that the user can add new maps, vehicles, pedestrians, objects, semantic tags, and more and have DriveTruth work with them, allowing the dataset to be tailored to specific applications.

[The CARLA docs provide many tutorials on modifying the engine](https://carla.readthedocs.io/en/latest/).  Depending on what you do, you may or may not have to modify DriveTruth to account for the new changes. 

- If you add new vehicles or pedestrians, if they are classified under the same blueprints as their default counterparts (``vehicle.*`` or ``pedestrian.*`` respectively), no changes need to be made.  See section 3, "Cars and Pedestrians", if you want to change the distribution of models, or only use a specific model.
- If you add a new map, by default you need to classify it as "residential" or "highway" and add it to the list of maps.  If you do not use map types this is not needed.  See Section 3, "Maps" for how to change how DriveTruth uses maps
- If you add new objects to maps, as long as the map loads no changes are needed.
- If you add new semantic tags, no changes are needed, unless you override the existing semantic tags for traffic signs and traffic lights.  In this case, you may need to update the semantic tags for each within the code, or comment them out if you don't want them. 

