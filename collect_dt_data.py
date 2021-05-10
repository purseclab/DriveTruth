### CARLA Simulator is licensed under the terms of the MIT license
### For a copy, see <https://opensource.org/licenses/MIT>
### For more information about CARLA Simulator, visit https://carla.org/

import glob
import os
import sys
import time
from datetime import datetime
import csv
import random
import traceback

try:
    print("Carla path: ")
    print(os.path.abspath('../carla/dist'))
    print("Version: ")
    print(sys.version_info.major)
    print(sys.version_info.minor)
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    try:
        # Get the other version
        sys.path.append(glob.glob('../carla/dist/carla-0.9.11-py3.7-%s.egg' % (
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        print('carla not found')
        pass

import carla
import argparse
import logging
import random
import queue
import numpy as np
from matplotlib import pyplot as plt
import cv2
import carla_vehicle_annotator as cva
import math


def rotate_point_matrix(yaw, pitch, roll, point_arr):
    yaw_matrix = np.array([[math.cos(yaw), -1 * math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    pitch_matrix = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-1 * math.sin(pitch), 0, math.cos(pitch)]])
    roll_matrix = np.array([[1, 0, 0], [0, math.cos(roll), math.sin(-1 * roll)], [0, math.sin(roll), math.cos(roll)]])
    r = np.matmul(yaw_matrix, np.matmul(pitch_matrix, roll_matrix))
    # Now we use this to compute the resultant point
    point_matrix = np.array([[float(point_arr[0])], [float(point_arr[1])], [float(point_arr[2])]])
#    print("point matrix: ")
#    print(point_matrix)
    resulting_point = np.matmul(r, point_matrix)
    return [resulting_point[0][0], resulting_point[1][0], resulting_point[2][0]]

def point_cloud_to_3d_bbox(semantic_lidar_measurement, sought_semantic_tags, debug_filename = '', debug = False):
    sought_object_ids = []
    # First, go through points and find all of the "sought object" ids
    if debug:
        if not os.path.exists(os.path.dirname(debug_filename)):
            os.makedirs(os.path.dirname(debug_filename))
    #
        debug_file = open(debug_filename, "w")
        debug_file.write("-------------------")
        debug_file.write("Seeking tags:\n")
        debug_file.write(', '.join([str(elem) for elem in sought_semantic_tags]))
        debug_file.write("\n")
    #    debug_file.write("Semantic lidar measurements:\n")
    #    debug_file.write(', '.join([str(detection) for detection in semantic_lidar_measurement]))
        debug_file.write("\n")
        debug_file.write("Raycast IDS:\n")
    for detection in semantic_lidar_measurement:
        if detection.object_idx not in sought_object_ids and detection.object_tag in sought_semantic_tags:
            if debug:
                debug_file.write("Object " + str(detection.object_idx) + " found with tag " + str(detection.object_tag) + "\n")
            sought_object_ids.append(detection.object_idx)
        else:
            if debug:
                debug_file.write("Detection not notable: object " + str(detection.object_idx) + " and tag " + str(detection.object_tag) + "\n")
            else:
                pass
            
    # Now we go through all the points again, getting the largest and smallest values in each axis to get 8 coordinates total, one for each corner
    sought_3d_bboxes = {}
    curr_3d_extents = {}
    for obj in sought_object_ids:
        curr_3d_extents[str(obj)] = {'x_min': None, 'x_max': None, 'y_min': None, 'y_max': None, 'z_min': None, 'z_max': None}
    for detection in semantic_lidar_measurement:
        if detection.object_idx in sought_object_ids and detection.object_tag in sought_semantic_tags: # Second condition prevents getting pole of same object
#            curpoint = detection.point
            lidar_location = semantic_lidar_measurement.transform.location
            lidar_rotation = semantic_lidar_measurement.transform.rotation
            point_arr = [detection.point.x, detection.point.y, -1 * detection.point.z]
            # Convert via rotation
            point_arr = rotate_point_matrix(math.radians(lidar_rotation.yaw), math.radians(lidar_rotation.pitch), math.radians(lidar_rotation.roll), point_arr)
#            print("LIDAR LOCATION (LOOKING AT OBJECT ID " + str(detection.object_idx) + ")")
#            print(lidar_location)
#            print("LIDAR ROTATION")
#            print(lidar_rotation)
#            print("DETECTION POINT COORDINATES")
#            print(detection.point)
#            print("CONVERTED COORDINATES:")
#            print(point_arr)
            curpoint = [lidar_location.x + point_arr[0], lidar_location.y + point_arr[1], lidar_location.z + point_arr[2]]
#            curpoint = [(-1 * detection.point.y) + lidar_location.x, detection.point.x + lidar_location.y, (-1 * detection.point.z) + lidar_location.z]
#            curpoint = transform_lidar_point(lidar_transform, point)
#            print("Curpoint:")
#            print(curpoint)
            if curr_3d_extents[str(detection.object_idx)]['x_min'] is None or curr_3d_extents[str(detection.object_idx)]['x_min'] > curpoint[0]:
                curr_3d_extents[str(detection.object_idx)]['x_min'] = curpoint[0]
            if curr_3d_extents[str(detection.object_idx)]['x_max'] is None or curr_3d_extents[str(detection.object_idx)]['x_max'] < curpoint[0]:
                curr_3d_extents[str(detection.object_idx)]['x_max'] = curpoint[0]
            if curr_3d_extents[str(detection.object_idx)]['y_min'] is None or curr_3d_extents[str(detection.object_idx)]['y_min'] > curpoint[1]:
                curr_3d_extents[str(detection.object_idx)]['y_min'] = curpoint[1]
            if curr_3d_extents[str(detection.object_idx)]['y_max'] is None or curr_3d_extents[str(detection.object_idx)]['y_max'] < curpoint[1]:
                curr_3d_extents[str(detection.object_idx)]['y_max'] = curpoint[1]
            if curr_3d_extents[str(detection.object_idx)]['z_min'] is None or curr_3d_extents[str(detection.object_idx)]['z_min'] > curpoint[2]:
                curr_3d_extents[str(detection.object_idx)]['z_min'] = curpoint[2]
            if curr_3d_extents[str(detection.object_idx)]['z_max'] is None or curr_3d_extents[str(detection.object_idx)]['z_max'] < curpoint[2]:
                curr_3d_extents[str(detection.object_idx)]['z_max'] = curpoint[2]
                
    # Now that everything has an extent, we can convert that into a bounding box
    for obj in sought_object_ids:
        extents = curr_3d_extents[str(obj)]
#        print("Extents for obj " + str(obj))
#        print(extents)
        bbox_center = [(extents['x_max'] + extents['x_min'])/2, (extents['y_max'] + extents['y_min'])/2, (extents['z_max'] + extents['z_min'])/2]
        bbox_extent = [abs(extents['x_max'] - extents['x_min'])/2, abs(extents['y_max'] - extents['y_min'])/2, abs(extents['z_max'] - extents['z_min'])/2]
        cent = carla.Location(bbox_center[0], bbox_center[1], bbox_center[2])
        ext = carla.Vector3D(bbox_extent[0], bbox_extent[1], bbox_extent[2])
        # We now set the center location to 0 because the bounding box calculation expects a point RELATIVE to the object it's attached to
        rel_cent = carla.Location(0, 0, 0)
        bbox = carla.BoundingBox(rel_cent, ext)
        sought_3d_bboxes[str(obj)] = [bbox, cent]
    
    if debug:
        debug_file.write("Returning sought boxes: \n")
        for key, value in sought_3d_bboxes.items():
            debug_file.write(key + "\n")
    #    debug_file.write(sought_3d_bboxes)
        debug_file.close()
    return sought_3d_bboxes


def retrieve_data(sensor_queue, frame, timeout=0.2): # Set to 1/5th of a second
    while True:
        try:
            data = sensor_queue.get(True,timeout)
        except queue.Empty:
            return None
        if data.frame == frame:
            return data


def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=100,
        type=int,
        help='number of vehicles (default: 100)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=50,
        type=int,
        help='number of walkers (default: 50)')

    argparser.add_argument(
        '-tm_p', '--tm_port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')

    argparser.add_argument(
        '-o', '--out',
        metavar='O',
        default=os.getcwd() + '/output/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/',
        help='Default output location')

    argparser.add_argument(
        '--lidar',
        metavar='L',
        default='n',
        help='Y/N dump lidar data.')

    argparser.add_argument(
        '--fps',
        metavar='F',
        default=5,
        help='FPS to run data collection at.')

    argparser.add_argument(
        '--time_limit',
        metavar='T',
        default=200,
        help='How many seconds each data collection should run before the simulation is reset.')

    args = argparser.parse_args()
    
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    
    fps = int(args.fps)
    time_limit = int(args.time_limit) # How many seconds before we re-initialize the whole thing
    segment_num = 100 # How many runs to record before we end
    target_segment_limit = 1000 # How many we should go up to
    # So we run for each map type

    map_types = ["residential", "highway"]
    args_out_og = args.out
    args_out_og2 = args.out
    try:
        for j in range(0, int(target_segment_limit/segment_num)):
            args_out_og = args_out_og2 + "/" + str(j) + "/"
            for mtype in map_types:
                print("Map type: " + mtype)
                for i in range(0, segment_num):
                    args.out = args_out_og + mtype + "_" + str(i) + "/" 
                    print("Running segment " + str(i+1) + "/" + str(segment_num))
                    run_simulation_instance(client, args, mtype, fps, time_limit)
    except Exception:
        traceback.print_exc(file=sys.stdout)
            
    
def run_simulation_instance(client, args, map_type, fps, time_limit):
    vehicles_list = []
    walkers_list = []
    nonvehicles_list = []
    all_id = []
    
    sim_start_time = None
    
    #Multiplier for channels/number of points in semantic lidar.  Higher values mean higher resolution, but slightly longer runtime.  Recommended to leave at default unless you know what you're doing.
    sem_res = 1

    try:
        residential_maps = ['/Game/Carla/Maps/Town01_Opt', '/Game/Carla/Maps/Town02_Opt', '/Game/Carla/Maps/Town03_Opt', '/Game/Carla/Maps/Town05_Opt', '/Game/Carla/Maps/Town06_Opt', '/Game/Carla/Maps/Town07_Opt']
        highway_maps = ['/Game/Carla/Maps/Town04_Opt', '/Game/Carla/Maps/Town06_Opt']
        start_time = time.time()
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        chosen_map = None
        if map_type == "residential":
            chosen_map = random.choice(residential_maps)
        elif map_type == "highway":
            chosen_map = random.choice(highway_maps)
        else:
            maps = client.get_available_maps()
            chosen_map = random.choice(maps)
        world = client.load_world(chosen_map)
        
        # We choose the weather and time of day from a list of available weather presets
        # You can also set weather manually if you so choose.  This also allows you to set a specific distribution and time of day.  See documentation:
        # https://carla.readthedocs.io/en/0.9.3/python_api_tutorial/#changing-the-weather

        weather_presets = [carla.WeatherParameters.ClearNoon, carla.WeatherParameters.CloudyNoon, carla.WeatherParameters.WetNoon, carla.WeatherParameters.WetCloudyNoon, carla.WeatherParameters.MidRainyNoon, carla.WeatherParameters.HardRainNoon, carla.WeatherParameters.SoftRainNoon, carla.WeatherParameters.ClearSunset, carla.WeatherParameters.CloudySunset, carla.WeatherParameters.WetSunset, carla.WeatherParameters.WetCloudySunset, carla.WeatherParameters.MidRainSunset, carla.WeatherParameters.HardRainSunset, carla.WeatherParameters.SoftRainSunset]
        chosen_weather = random.choice(weather_presets)
        world.set_weather(chosen_weather)
        print("Weather: " + str(chosen_weather))
        
        print('\nRUNNING in synchronous mode\n')
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05 # We'll keep this at a 20 FPS rate, just get sensor readings at a different time
            world.apply_settings(settings)
        else:
            synchronous_master = False

        blueprints = world.get_blueprint_library().filter('vehicle.*')
        blueprints_p = world.get_blueprint_library().filter('walker.*')
        
        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        # Spawn ego vehicle
        ego_bp = random.choice(blueprints)
        ego_transform = random.choice(spawn_points)
        ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
        vehicles_list.append(ego_vehicle)
        ego_vehicle.set_autopilot(True)
        print('Ego-vehicle ready')
        
        spawn_points.remove(ego_transform)
        number_of_spawn_points = len(spawn_points)
        
        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points
#            if ratio_num_vehicles < 1:
#                args.number_of_vehicles = number_of_spawn_points * ratio_num_vehicles
#                args.number_of_pedestrians = number_of_spawn_points - args.number_of_vehicles
#            else:
#                args.number_of_pedestrians = number_of_spawn_points * (1/ratio_num_vehicles)
#                args.number_of_vehicles = number_of_spawn_points - args.number_of_pedestrians

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
            spawn_points.pop(0)

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        print('Created %d npc vehicles \n' % len(vehicles_list))

        # --------------
        # Spawn Pedestrians
        # --------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprints_p)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
        
        print('spawned %d walkers \n' % len(walkers_list))
        # -----------------------------
        # Spawn ego vehicle and sensors
        # -----------------------------
        q_list = []
        idx = 0
        
        tick_queue = queue.Queue()
        world.on_tick(tick_queue.put)
        q_list.append(tick_queue)
        tick_idx = idx
        idx = idx+1


        # Spawn RGB camera
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('sensor_tick', str(1/fps))
        cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)
        nonvehicles_list.append(cam)
        cam_queue = queue.Queue()
        cam.listen(cam_queue.put)
        q_list.append(cam_queue)
        cam_idx = idx
        idx = idx+1
        print('RGB camera ready')
        cam_sem_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        cam_sem_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        cam_sem_bp.set_attribute('sensor_tick', str(1/fps))
        cam_sem = world.spawn_actor(cam_sem_bp, cam_sem_transform, attach_to=ego_vehicle)
        nonvehicles_list.append(cam_sem)
        cam_sem_queue = queue.Queue()
        cam_sem.listen(cam_sem_queue.put)
        q_list.append(cam_sem_queue)
        cam_sem_idx = idx
        idx = idx+1

        # Spawn semantic segmentation camera
        
        # Spawn LIDAR sensor
        # We're using a dumbed-down semantic sensor so we don't have to worry about intensity
        # At the same time, it doesn't have as many points as the "real" semantic lidar to save some time
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic') 
        lidar_bp.set_attribute('sensor_tick', str(1/fps))
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '1120000')
        lidar_bp.set_attribute('upper_fov', '40')
        lidar_bp.set_attribute('lower_fov', '-40')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
        nonvehicles_list.append(lidar)
        lidar_queue = queue.Queue()
        lidar.listen(lidar_queue.put)
        q_list.append(lidar_queue)
        lidar_idx = idx
        idx = idx+1
        print('LIDAR ready')
        
        if args.lidar == 'y': # We spawn an addition lidar sensor for lidar collection
            t_lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
            t_lidar_bp.set_attribute('sensor_tick', str(1/fps))
            t_lidar_bp.set_attribute('channels', '64')
            t_lidar_bp.set_attribute('points_per_second', '1120000')
            t_lidar_bp.set_attribute('upper_fov', '40')
            t_lidar_bp.set_attribute('lower_fov', '-40')
            t_lidar_bp.set_attribute('range', '100')
            t_lidar_bp.set_attribute('rotation_frequency', '20')
            t_lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            t_lidar = world.spawn_actor(t_lidar_bp, t_lidar_transform, attach_to=ego_vehicle)
            nonvehicles_list.append(t_lidar)
            t_lidar_queue = queue.Queue()
            t_lidar.listen(t_lidar_queue.put)
            q_list.append(t_lidar_queue)
            t_lidar_idx = idx
            idx = idx+1
            print('TRUE LIDAR ready')

        # Spawn semantic lidar
        lidar_sem_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        lidar_sem_bp.set_attribute('sensor_tick', str(1/fps))
        lidar_sem_bp.set_attribute('channels', str(1024 * float(sem_res)))
        lidar_sem_bp.set_attribute('points_per_second', str(17920000 * float(sem_res))) # Both are set to extremely high resolutions to get objects better
        lidar_sem_bp.set_attribute('upper_fov', '90')
        lidar_sem_bp.set_attribute('lower_fov', '-90')
        lidar_sem_bp.set_attribute('range', '100')
        lidar_sem_bp.set_attribute('rotation_frequency', '20')
        lidar_sem_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        lidar_sem = world.spawn_actor(lidar_sem_bp, lidar_sem_transform, attach_to=ego_vehicle)
        nonvehicles_list.append(lidar_sem)
        lidar_sem_queue = queue.Queue()
        lidar_sem.listen(lidar_sem_queue.put)
        q_list.append(lidar_sem_queue)
        lidar_sem_idx = idx
        idx = idx+1
        print("Semantic Segmentation camera ready")
        end_time = (time.time() - start_time)
        print("Initialization time: %.2f seconds" % end_time)
        start_sim_time = time.time()
        # Begin the loop
        time_sim = 0
        overall_time_sim = 0
        print("MAP:")
        print(chosen_map)
        while overall_time_sim <= time_limit:
            # Extract the available data
            nowFrame = world.tick()

            # Check whether it's time for sensor to capture data
            if time_sim >= (1/fps): # Capture at fps rate
                data = [retrieve_data(q,nowFrame) for q in q_list]
                assert all(x.frame == nowFrame for x in data if x is not None)
#                print("Frame: " + str(nowFrame))
                # Skip if any sensor data is not available
                if None in data:
                    continue
                
                vehicles_raw = world.get_actors().filter('vehicle.*')
                walkers_raw = world.get_actors().filter('walker.*')
                snap = data[tick_idx]
                rgb_img = data[cam_idx]
                lidar_img = data[lidar_idx]
                sem_img = data[lidar_sem_idx]
                cam_sem_img = data[cam_sem_idx]
                
                        
                # Save all of the sensors for debug
                cva.show_lidar(cva.filter_lidar(lidar_img, cam, 100), cam, rgb_img, path=args.out)
                cva.show_sem_lidar(cva.filter_lidar(sem_img, cam, 100), cam, rgb_img, path=args.out)

                # Attach additional information to the snapshot
                vehicles = cva.snap_processing(vehicles_raw, snap, veh_check=ego_vehicle.id)
                walkers = cva.snap_processing(walkers_raw, snap)
                
                # Get header information
                ego_cur_speed = ego_vehicle.get_velocity()
                ego_cur_position = ego_vehicle.get_location()
                ego_veh_control = ego_vehicle.get_control()
                ego_throttle = str(ego_veh_control.throttle)
                ego_steer = str(ego_veh_control.steer)
                ego_brake = str(ego_veh_control.brake)
                ego_handbrake = str(ego_veh_control.hand_brake)
                ego_reverse = str(ego_veh_control.reverse)
                ego_manual_gear_shift = str(ego_veh_control.manual_gear_shift)
                ego_gear = str(ego_veh_control.gear)
                cam_rotation = cam.get_transform().rotation
                # Calculating visible bounding boxes
#                filtered_out,_ = cva.auto_annotate_lidar(vehicles, cam, lidar_img, show_img = rgb_img, json_path = 'vehicle_class_json_file.txt')
                vehicle_data = cva.auto_annotate_lidar_process(vehicles, cam, lidar_img, ego_cur_speed, ego_cur_position, max_dist = 100, min_detect = 5, show_img = None, gt_class = "Vehicle")
                walker_data = cva.auto_annotate_lidar_process(walkers, cam, lidar_img, ego_cur_speed, ego_cur_position, max_dist = 100, min_detect = 5, show_img = None, gt_class = "Pedestrian")
                # We can get the rest of the data from the semantic lidar
                debug_fname = str(args.out) + 'debug/frame_' + str(nowFrame) + '_signs.txt'
                debug_fname2 = str(args.out) + 'debug/frame_' + str(nowFrame) + '_lights.txt'
                ## !!TRACKING STEP 1
                traffic_signs_bboxes = point_cloud_to_3d_bbox(sem_img, [12], debug_fname)
                ## !!END TRACKING STEP 1
                traffic_lights_bboxes = point_cloud_to_3d_bbox(sem_img, [18], debug_fname2)
                
                # !!TRACKING STEP 2
                if traffic_signs_bboxes: #Not empty
                    traffic_sign_ids = []
                    for key in traffic_signs_bboxes.keys():
                        traffic_sign_ids.append(int(key))
                    traffic_signs = cva.snap_processing_manual_bbox(traffic_sign_ids, snap, traffic_signs_bboxes)
                    traffic_signs_data = cva.semantic_auto_annotate(traffic_signs, cam, lidar_img, ego_cur_speed, ego_cur_position, max_dist = 100, min_detect = 5, show_img = None, gt_class = "Traffic Sign")
                else:
                    traffic_signs_data = {}
                # !!END TRACKING STEP 2
                
                if traffic_lights_bboxes:
                    traffic_light_ids = []
                    for key in traffic_lights_bboxes.keys():
                        traffic_light_ids.append(int(key))
                    traffic_lights = cva.snap_processing_manual_bbox(traffic_light_ids, snap, traffic_lights_bboxes)
                    traffic_lights_data = cva.semantic_auto_annotate(traffic_lights, cam, lidar_img, ego_cur_speed, ego_cur_position, max_dist = 100, min_detect = 5, show_img = None, gt_class = "Traffic Light")
                else:
                    traffic_lights_data = {}
                
                # Merge all of the data into a single dictionary
                ## !!TRACKING STEP 3
                out_data = {**vehicle_data, **walker_data, **traffic_signs_data, **traffic_lights_data}
                ## !!END TRACKING STEP 3
                # Save the results
                cva.save_output_img(rgb_img, out_data, path=args.out, save_patched=True)
                cam_sem_img.save_to_disk(args.out + "out_sem_cam/%06d.png" % cam_sem_img.frame, carla.ColorConverter.CityScapesPalette)                
                # Now we save the pertinent information, in a special folder
                dirname = str(args.out) + 'frame_' + str(nowFrame) + '/'
                if not os.path.exists(dirname):
                    os.makedirs(os.path.dirname(dirname))
                # Dump lidar measurements
                if args.lidar == 'y':
                    # Below is a method to save to CSV
                    # However, this is time consuming
                    # Uncomment it if you want your lidar data in CSV, but are OK with longer runtime
#                    t_lidar_img = data[t_lidar_idx]
#                    csv_columns_li = ['x', 'y', 'z', 'intensity']
#                    csv_file_li = str(dirname) + 'lidar_' + str(nowFrame) + '.csv'
#                    lidar_dump = []
#                    for point in t_lidar_img:
#                        lidar_out = {'x': point.point.x, 'y': point.point.y, 'z': point.point.z, 'intensity': point.intensity}
#                        lidar_dump.append(lidar_out)
#                        try:
#                            with open(csv_file_li, 'w', newline='') as csvfile:
#                                writer = csv.DictWriter(csvfile, fieldnames=csv_columns_li)
#                                writer.writeheader()
#                                for data in lidar_dump:
#                                    writer.writerow(data)
#                        except IOError:
#                            print("I/O error")
                    t_lidar_img = data[t_lidar_idx]
                    ply_file = str(dirname) + 'lidar_' + str(nowFrame) + '.ply'
                    t_lidar_img.save_to_disk(ply_file)
                    
                # First save header info in a txt file
                headerfile = open(str(dirname) + 'frame_' + str(nowFrame) + '.txt', "w")
                headerfile.write("Ego Vehicle Speed: " + "[" + str(ego_cur_speed.x) + ", " + str(ego_cur_speed.y) + ", " + str(ego_cur_speed.z) + "]\n")
                headerfile.write("Ego Vehicle Position: " + "[" + str(ego_cur_position.x) + ", " + str(ego_cur_position.y) + ", " + str(ego_cur_position.z) + "]\n")
                headerfile.write("Ego Vehicle Throttle: " + ego_throttle + "\n")
                headerfile.write("Ego Vehicle Steer: " + ego_steer + "\n")
                headerfile.write("Ego Vehicle Brake: " + ego_brake + "\n")
                headerfile.write("Ego Vehicle Handbrake: " + ego_handbrake + "\n")
                headerfile.write("Ego Vehicle Reverse: " + ego_reverse + "\n")
                headerfile.write("Ego Vehicle Manual Gear Shift: " + ego_manual_gear_shift + "\n")
                headerfile.write("Ego Vehicle Gear: " + ego_gear + "\n")
                headerfile.write("Camera Pitch (Degrees): " + str(cam_rotation.pitch) + "\n")
                headerfile.write("Camera Yaw (Degrees): " + str(cam_rotation.yaw) + "\n")
                headerfile.write("Camera Roll (Degrees): " + str(cam_rotation.roll) + "\n")
                headerfile.close()
                # Then save a csv file with all of the values
                csv_columns = ['id', 'bbox', 'location', 'class', 'rel_velocity', 'distance']
                csv_file = str(dirname) + 'frame_' + str(nowFrame) + '.csv'
                # Reprocess the out data with the extra ID parameter, with everything in one
                out_data_csv = []
                for key, value in out_data.items():
                    cur_out = {**{'id': key}, **value}
                    out_data_csv.append(cur_out)
                
                try:
                    with open(csv_file, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                        writer.writeheader()
                        for data in out_data_csv:
                            writer.writerow(data)
                except IOError:
                    print("I/O error")
                    
                time_sim = 0
            time_sim = time_sim + settings.fixed_delta_seconds
            overall_time_sim = overall_time_sim + settings.fixed_delta_seconds

    finally:
        if start_sim_time is not None:
            end_sim_time = (time.time() - start_sim_time)
            print("Simulation runtime: %.2f seconds" % end_sim_time)
        try: 
            cam.stop()
            cam_sem.stop()
            lidar.stop()
            lidar_sem.stop()
            if args.lidar == 'y':
                t_lidar.stop()
        except: 
            print('Sensors has not been initiated')
        
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        print('destroying %d nonvehicles' % len(nonvehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in nonvehicles_list])
        
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])


        time.sleep(0.5)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
