### Functions to convert 3D point cloud to 2D bbox adapted from code by Mukhlas Adib and CARLA's client_bounding_boxes.py

### CARLA Simulator and client_bounding_boxes.py are licensed under the terms of the MIT license
### For a copy, see <https://opensource.org/licenses/MIT>
### For more information about CARLA Simulator, visit https://carla.org/

import numpy as np
import PIL
from PIL import Image
from PIL import ImageDraw
import json
import pickle
import os
import glob
import sys
import cv2
import carla


# Special function that computes attributes based purely on the semantically-calculated bounding boxes
def semantic_auto_annotate(objects, camera, lidar_data, ego_velocity, ego_location, max_dist = 100, min_detect = 5, show_img = None, gt_class = None):
    filtered_data = filter_lidar(lidar_data, camera, max_dist)
    if show_img != None:
        show_lidar(filtered_data, camera, show_img)

    ### Delete this section if object_idx issue has been fixed in CARLA
    filtered_data = np.array([p for p in filtered_data if p.object_idx != 0])
    filtered_data = get_points_id(filtered_data, objects, camera, max_dist)
    ###
    
    visible_id, idx_counts = np.unique([p.object_idx for p in filtered_data], return_counts=True)
    visible_objects = [v for v in objects if v.id in visible_id]
    visible_objects= [v for v in objects if idx_counts[(visible_id == v.id).nonzero()[0]] >= min_detect] # min_detect controls the minimum number of lidar points to "see" an object, allowing occluded or distant objects to remain untracked
    # Now we have a dictionary storing the object
    annotated_dict = {}
    for o in visible_objects:
        object_bbox = get_2d_bb(o, camera, True) # We set the hasWorldCoord to true, because we do have a separate location for these objects!
        # Process this into left, up, right, bottom
        object_bbox = [object_bbox[0][0], object_bbox[0][1], object_bbox[1][0], object_bbox[1][1]]
        if gt_class is not None:
            object_class = gt_class
        else:
            object_class = "None"
        # Get relative velocity
        object_velocity = o.get_velocity()
        object_relative_velocity = object_velocity - ego_velocity
        # Get distance (This is mainly where the function differs from the below one)
        object_location = o.loc_given
        object_distance = object_location.distance(ego_location)
        annotated_dict[str(o.id)] = {'bbox': object_bbox, 'location': object_location, 'class': object_class, 'rel_velocity': object_relative_velocity, 'distance': object_distance}

    return annotated_dict
# Collect the data in a single frame
def auto_annotate_lidar_process(objects, camera, lidar_data, ego_velocity, ego_location, max_dist = 100, min_detect = 5, show_img = None, gt_class = None):
    filtered_data = filter_lidar(lidar_data, camera, max_dist)
    if show_img != None:
        show_lidar(filtered_data, camera, show_img)

    ### Delete this section if object_idx issue has been fixed in CARLA
    filtered_data = np.array([p for p in filtered_data if p.object_idx != 0])
    filtered_data = get_points_id(filtered_data, objects, camera, max_dist)
    ###
    
    visible_id, idx_counts = np.unique([p.object_idx for p in filtered_data], return_counts=True)
    visible_objects = [v for v in objects if v.id in visible_id]
    visible_objects= [v for v in objects if idx_counts[(visible_id == v.id).nonzero()[0]] >= min_detect]
    # Now we have a dictionary storing the object
    annotated_dict = {}
    for o in visible_objects:
        object_bbox = get_2d_bb(o, camera)
        # Process this into left, up, right, bottom
        object_bbox = [object_bbox[0][0], object_bbox[0][1], object_bbox[1][0], object_bbox[1][1]]
        if gt_class is not None:
            object_class = gt_class
        else:
            object_class = "None"
        # Get relative velocity
        object_velocity = o.get_velocity()
        object_relative_velocity = object_velocity - ego_velocity
        # Get distance
        object_location = o.get_transform().location
        object_distance = object_location.distance(ego_location)
        annotated_dict[str(o.id)] = {'bbox': object_bbox, 'location': object_location, 'class': object_class, 'rel_velocity': object_relative_velocity, 'distance': object_distance}

    return annotated_dict


### From Mukhlas Adib, this function can debug occlusion filtering via depth image
def auto_annotate_debug(vehicles, camera, depth_img, depth_show=False, max_dist=100, depth_margin=-1, patch_ratio=0.5, resize_ratio=0.5, json_path=None):
    vehicles = filter_angle_distance(vehicles, camera, max_dist)
    bounding_boxes_2d = [get_2d_bb(vehicle, camera) for vehicle in vehicles]
    if json_path is not None:
        vehicle_class = get_vehicle_class(vehicles, json_path)
    else:
        vehicle_class = []
    filtered_out, removed_out, depth_area, depth_show = filter_occlusion_bbox(bounding_boxes_2d, vehicles, camera, depth_img, vehicle_class, depth_show, depth_margin, patch_ratio, resize_ratio)
    return filtered_out, removed_out, depth_area, depth_show



### Get camera intrinsic matrix 'k'
def get_camera_intrinsic(sensor):
    VIEW_WIDTH = int(sensor.attributes['image_size_x'])
    VIEW_HEIGHT = int(sensor.attributes['image_size_y'])
    VIEW_FOV = int(float(sensor.attributes['fov']))
    calibration = np.identity(3)
    calibration[0, 2] = VIEW_WIDTH / 2.0
    calibration[1, 2] = VIEW_HEIGHT / 2.0
    calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
    return calibration

### Extract bounding box vertices of vehicle
def create_bb_points(vehicle):
    cords = np.zeros((8, 4))
    extent = vehicle.bounding_box.extent
    cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
    cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
    cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
    cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
    cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
    cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
    cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
    cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
    return cords

### Get transformation matrix from carla.Transform object
def get_matrix(transform):
    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix    

### Transform coordinate from vehicle reference to world reference
def vehicle_to_world(cords, vehicle, hasWorldCoords=False):
    bb_transform = carla.Transform(vehicle.bounding_box.location)
    bb_vehicle_matrix = get_matrix(bb_transform)
    if hasWorldCoords:
        v_trans = carla.Transform(vehicle.loc_given, vehicle.get_transform().rotation) # I set it to the given location but keep the rotation 
    else:
        v_trans = vehicle.get_transform()
    vehicle_world_matrix = get_matrix(v_trans)
    bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
    world_cords = np.dot(bb_world_matrix, np.transpose(cords))
    return world_cords

### Transform coordinate from world reference to sensor reference
def world_to_sensor(cords, sensor):
    sensor_world_matrix = get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    sensor_cords = np.dot(world_sensor_matrix, cords)
    return sensor_cords

### Transform coordinate from vehicle reference to sensor reference
def vehicle_to_sensor(cords, vehicle, sensor, hasWorldCoords = False):
    world_cord = vehicle_to_world(cords, vehicle, hasWorldCoords)
    sensor_cord = world_to_sensor(world_cord, sensor)
    return sensor_cord

### Summarize bounding box creation and project the poins in sensor image
def get_bounding_box(vehicle, sensor, hasWorldCoords=False):
    camera_k_matrix = get_camera_intrinsic(sensor)
    bb_cords = create_bb_points(vehicle)
    cords_x_y_z = vehicle_to_sensor(bb_cords, vehicle, sensor, hasWorldCoords)[:3, :]
    # Trying to see if not negating the z axis (which I already account for) works better
    if hasWorldCoords:
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], cords_x_y_z[2, :], cords_x_y_z[0, :]])
    else:
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
    bbox = np.transpose(np.dot(camera_k_matrix, cords_y_minus_z_x))
    camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
    return camera_bbox

### Draw 2D bounding box (4 vertices) from 3D bounding box (8 vertices) in image
### 2D bounding box is represented by two corner points
def p3d_to_p2d_bb(p3d_bb):
    min_x = np.amin(p3d_bb[:,0])
    min_y = np.amin(p3d_bb[:,1])
    max_x = np.amax(p3d_bb[:,0])
    max_y = np.amax(p3d_bb[:,1])
    p2d_bb = np.array([[min_x,min_y] , [max_x,max_y]])
    return p2d_bb

### Summarize 2D bounding box creation
def get_2d_bb(vehicle, sensor, hasWorldCoords=False):
    p3d_bb = get_bounding_box(vehicle, sensor, hasWorldCoords)
    p2d_bb = p3d_to_p2d_bb(p3d_bb)
    return p2d_bb
    
### Use these functions to remove invisible vehicles
### Get numpy 2D array of vehicles' location and rotation from world reference, also locations from sensor reference
def get_list_transform(vehicles_list, sensor):
    t_list = []
    for vehicle in vehicles_list:
        v = vehicle.get_transform()
        transform = [v.location.x , v.location.y , v.location.z , v.rotation.roll , v.rotation.pitch , v.rotation.yaw]
        t_list.append(transform)
    t_list = np.array(t_list).reshape((len(t_list),6))
    
    transform_h = np.concatenate((t_list[:,:3],np.ones((len(t_list),1))),axis=1)
    sensor_world_matrix = get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    transform_s = np.dot(world_sensor_matrix, transform_h.T).T
    
    return t_list , transform_s

### Remove vehicles that are not in the FOV of the sensor
def filter_angle(vehicles_list, v_transform, v_transform_s, sensor):
    attr_dict = sensor.attributes
    VIEW_FOV = float(attr_dict['fov'])
    v_angle = np.arctan2(v_transform_s[:,1],v_transform_s[:,0]) * 180 / np.pi

    selector = np.array(np.absolute(v_angle) < (int(VIEW_FOV)/2))
    vehicles_list_f = [v for v, s in zip(vehicles_list, selector) if s]
    v_transform_f = v_transform[selector[:,0],:]
    v_transform_s_f = v_transform_s[selector[:,0],:]
    return vehicles_list_f , v_transform_f , v_transform_s_f

### Remove vehicles that have distance > max_dist from the sensor
def filter_distance(vehicles_list, v_transform, v_transform_s, sensor, max_dist=100):
    s = sensor.get_transform()
    s_transform = np.array([s.location.x , s.location.y , s.location.z])
    dist2 = np.sum(np.square(v_transform[:,:3] - s_transform), axis=1)
    selector = dist2 < (max_dist**2)
    vehicles_list_f = [v for v, s in zip(vehicles_list, selector) if s]
    v_transform_f = v_transform[selector,:]
    v_transform_s_f = v_transform_s[selector,:] 
    return vehicles_list_f , v_transform_f , v_transform_s_f

### Remove vehicles that are occluded from the sensor view based on one point depth measurement
### NOT USED by default because of the unstable result
def filter_occlusion_1p(vehicles_list, v_transform, v_transform_s, sensor, depth_img, depth_margin=2.0):
    camera_k_matrix = get_camera_intrinsic(sensor)
    CAM_W = int(sensor.attributes['image_size_x'])
    CAM_H = int(sensor.attributes['image_size_y'])
    pos_x_y_z = v_transform_s.T
    pos_y_minus_z_x = np.concatenate([pos_x_y_z[1, :], -pos_x_y_z[2, :]-0.0, pos_x_y_z[0, :]])
    img_pos = np.transpose(np.dot(camera_k_matrix, pos_y_minus_z_x))
    camera_pos = np.concatenate([img_pos[:, 0] / img_pos[:, 2], img_pos[:, 1] / img_pos[:, 2], img_pos[:, 2]], axis=1)

    u_arr = np.array(camera_pos[:,0]).flatten()
    v_arr = np.array(camera_pos[:,1]).flatten()
    dist = np.array(v_transform_s[:,0]).flatten()

    depth_patches = []
    v_depth = []
    for u, v in zip(list(u_arr),list(v_arr)):
        if u<=CAM_W and v<=CAM_H:
            v_depth.append(depth_img[int(v),int(u)])
            depth_a = np.array([[int(u)-3,int(v)-3] , [int(u)+3,int(v)+3]])
            depth_patches.append(depth_a)
        else:
            v_depth.append(0)
    v_depth = np.array(v_depth)
    
    selector = (dist-v_depth) < depth_margin
    vehicles_list_f = [v for v, s in zip(vehicles_list, selector) if s]
    v_transform_f = v_transform[selector,:]
    v_transform_s_f = v_transform_s[selector,:]
    return vehicles_list_f , v_transform_f , v_transform_s_f, depth_patches

### Apply angle and distance filters in one function
def filter_angle_distance(vehicles_list, sensor, max_dist=100):
    vehicles_transform , vehicles_transform_s = get_list_transform(vehicles_list, sensor)
    vehicles_list , vehicles_transform , vehicles_transform_s = filter_distance(vehicles_list, vehicles_transform, vehicles_transform_s, sensor, max_dist)
    vehicles_list , vehicles_transform , vehicles_transform_s = filter_angle(vehicles_list, vehicles_transform, vehicles_transform_s, sensor)
    return vehicles_list

### Apply occlusion filter based on resized bounding box depth values
def filter_occlusion_bbox(bounding_boxes, vehicles, sensor, depth_img, v_class=None, depth_capture=False, depth_margin=-1, patch_ratio=0.5, resize_ratio=0.5):
    filtered_bboxes = []
    filtered_vehicles = []
    filtered_v_class = []
    filtered_out = {}
    removed_bboxes = []
    removed_vehicles = []
    removed_v_class = []
    removed_out = {}
    selector = []
    patches = []
    patch_delta = []
    _, v_transform_s = get_list_transform(vehicles, sensor)
    
    for v, vs, bbox in zip(vehicles,v_transform_s,bounding_boxes):
        dist = vs[:,0]
        if depth_margin < 0:
            depth_margin = (v.bounding_box.extent.x**2+v.bounding_box.extent.y**2)**0.5 + 0.25
        uc = int((bbox[0,0]+bbox[1,0])/2)
        vc = int((bbox[0,1]+bbox[1,1])/2)
        wp = int((bbox[1,0]-bbox[0,0])*resize_ratio/2)
        hp = int((bbox[1,1]-bbox[0,1])*resize_ratio/2)
        u1 = uc-wp
        u2 = uc+wp
        v1 = vc-hp
        v2 = vc+hp
        depth_patch = np.array(depth_img[v1:v2,u1:u2])
        dist_delta = dist-depth_patch
        s_patch = np.array(dist_delta < depth_margin)
        s = np.sum(s_patch) > s_patch.shape[0]*patch_ratio
        selector.append(s)
        patches.append(np.array([[u1,v1],[u2,v2]]))
        patch_delta.append(dist_delta)
    
    for bbox,v,s in zip(bounding_boxes,vehicles,selector):
        if s:
            filtered_bboxes.append(bbox)
            filtered_vehicles.append(v)
        else:
            removed_bboxes.append(bbox)
            removed_vehicles.append(v)
    filtered_out['bbox']=filtered_bboxes
    filtered_out['vehicles']=filtered_vehicles
    removed_out['bbox']=removed_bboxes
    removed_out['vehicles']=removed_vehicles
        
    if v_class is not None:
        for cls,s in zip(v_class,selector):
            if s:
                filtered_v_class.append(cls)
            else:
                removed_v_class.append(cls)
        filtered_out['class']=filtered_v_class
        removed_out['class']=removed_v_class
    
    if depth_capture:
        depth_debug(patches, depth_img, sensor)
        for i,matrix in enumerate(patch_delta):
            print("\nvehicle "+ str(i) +": \n" + str(matrix))
        depth_capture = False
        
    return filtered_out, removed_out, patches, depth_capture

### Display area in depth image where measurement values are taken
def depth_debug(depth_patches, depth_img, sensor):
    CAM_W = int(sensor.attributes['image_size_x'])
    CAM_H = int(sensor.attributes['image_size_y'])
    #depth_img = depth_img/1000*255
    depth_img = np.log10(depth_img)
    depth_img = depth_img*255/4
    depth_img
    depth_3ch = np.zeros((CAM_H,CAM_W,3))
    depth_3ch[:,:,0] = depth_3ch[:,:,1] = depth_3ch[:,:,2] = depth_img
    depth_3ch = np.uint8(depth_3ch)
    image = Image.fromarray(depth_3ch, 'RGB')
    img_draw = ImageDraw.Draw(image)  
    for crop in depth_patches:
        u1 = int(crop[0,0])
        v1 = int(crop[0,1])
        u2 = int(crop[1,0])
        v2 = int(crop[1,1])
        crop_bbox = [(u1,v1),(u2,v2)]
        img_draw.rectangle(crop_bbox, outline ="red")
    image.show()

### Filter out lidar points that are outside camera FOV
def filter_lidar(lidar_data, camera, max_dist):
    CAM_W = int(camera.attributes['image_size_x'])
    CAM_H = int(camera.attributes['image_size_y'])
    CAM_HFOV = float(camera.attributes['fov'])
    CAM_VFOV = np.rad2deg(2*np.arctan(np.tan(np.deg2rad(CAM_HFOV/2))*CAM_H/CAM_W))
    lidar_points = np.array([[p.point.y,-p.point.z,p.point.x] for p in lidar_data])
    
    dist2 = np.sum(np.square(lidar_points), axis=1).reshape((-1))
    p_angle_h = np.absolute(np.arctan2(lidar_points[:,0],lidar_points[:,2]) * 180 / np.pi).reshape((-1))
    p_angle_v = np.absolute(np.arctan2(lidar_points[:,1],lidar_points[:,2]) * 180 / np.pi).reshape((-1))

    selector = np.array(np.logical_and(dist2 < (max_dist**2), np.logical_and(p_angle_h < (CAM_HFOV/2), p_angle_v < (CAM_VFOV/2))))
    filtered_lidar = [pt for pt, s in zip(lidar_data, selector) if s]
    return filtered_lidar

### Save camera image with projected lidar points for debugging purpose
def show_lidar(lidar_data, camera, carla_img, path=''):
    lidar_np = np.array([[p.point.y,-p.point.z,p.point.x] for p in lidar_data])
    cam_k = get_camera_intrinsic(camera)

    # Project LIDAR 3D to Camera 2D
    lidar_2d = np.transpose(np.dot(cam_k,np.transpose(lidar_np)))
    lidar_2d = (lidar_2d/lidar_2d[:,2].reshape((-1,1))).astype(int)

    # Visualize the result
    c_scale = []
    for pts in lidar_data:
        if pts.object_idx == 0: c_scale.append(255)
        else: c_scale.append(0)

    carla_img.convert(carla.ColorConverter.Raw)
    img_bgra = np.array(carla_img.raw_data).reshape((carla_img.height,carla_img.width,4))
    img_rgb = np.zeros((carla_img.height,carla_img.width,3))
    img_rgb[:,:,0] = img_bgra[:,:,2]
    img_rgb[:,:,1] = img_bgra[:,:,1]
    img_rgb[:,:,2] = img_bgra[:,:,0]
    img_rgb = np.uint8(img_rgb)

    for p,c in zip(lidar_2d,c_scale):
        c = int(c)
        cv2.circle(img_rgb,tuple(p[:2]),1,(c,c,c),-1)
    filename = path + 'out_lidar_img/%06d.jpg' % carla_img.frame
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename, img_rgb)

#Identical to show_lidar but with a different path and color method
def show_sem_lidar(lidar_data, camera, carla_img, path = ''):
    color_conversion = {'0': (0, 0, 0), '1': (70, 70, 70), '2': (100, 40, 40), '3': (55, 90, 80), '4': (220, 20, 60), '5': (153, 153, 153), '6': (157, 234, 50), '7': (128, 64, 128), '8': (244, 35, 232), '9': (107, 142, 35), '10': (0, 0, 142), '11': (102, 102, 156), '12': (220, 220, 0), '13': (70, 130, 180), '14': (81, 0, 81), '15': (150, 100, 100), '16': (230, 150, 140), '17': (180, 165, 180), '18': (250, 170, 30), '19': (110, 190, 160), '20': (170, 120, 50), '21': (45, 60, 150), '22': (145, 170, 100)}
    lidar_np = np.array([[p.point.y,-p.point.z,p.point.x] for p in lidar_data])
    cam_k = get_camera_intrinsic(camera)

    # Project LIDAR 3D to Camera 2D
    lidar_2d = np.transpose(np.dot(cam_k,np.transpose(lidar_np)))
    lidar_2d = (lidar_2d/lidar_2d[:,2].reshape((-1,1))).astype(int)

    # Visualize the result
    c_scale = []
    for pts in lidar_data:
        c_scale.append(color_conversion[str(pts.object_tag)])

    carla_img.convert(carla.ColorConverter.Raw)
    img_bgra = np.array(carla_img.raw_data).reshape((carla_img.height,carla_img.width,4))
    img_rgb = np.zeros((carla_img.height,carla_img.width,3))
    img_rgb[:,:,0] = img_bgra[:,:,2]
    img_rgb[:,:,1] = img_bgra[:,:,1]
    img_rgb[:,:,2] = img_bgra[:,:,0]
    img_rgb = np.uint8(img_rgb)

    for p,c in zip(lidar_2d,c_scale):
        cv2.circle(img_rgb,tuple(p[:2]),1,c,-1)
    filename = path + 'out_lidar_sem_img/%06d.jpg' % carla_img.frame
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename, img_rgb)

### Add actor ID of the vehcile hit by the lidar points
### Only used before the object_id issue of semantic lidar solved
def get_points_id(lidar_points, vehicles, camera, max_dist):
    vehicles_f = filter_angle_distance(vehicles, camera, max_dist)
    fixed_lidar_points = []
    for p in lidar_points:
        sensor_world_matrix = get_matrix(camera.get_transform())
        pw = np.dot(sensor_world_matrix, [[p.point.x],[p.point.y],[p.point.z],[1]])
        pw = carla.Location(pw[0,0],pw[1,0],pw[2,0])
        for v in vehicles_f:
            if v.bounding_box.contains(pw, v.get_transform()):
                p.object_idx = v.id
                break
        fixed_lidar_points.append(p)
    return fixed_lidar_points
        

### Use this function to save just the rgb image (with and without bounding box) in a specified path format 
def save_output_img(carla_img, out_data, cc_rgb=carla.ColorConverter.Raw, path='', save_patched=False):
    # Convert class to color
    class_to_color_dict = {"Vehicle": (255, 0, 0), "Pedestrian": (0, 0, 255), "Traffic Sign": (220, 200, 0), "Traffic Light": (250, 170, 30)}
    carla_img.save_to_disk(path + 'out_rgb_raw/%06d.png' % carla_img.frame, cc_rgb)
    if save_patched:
        carla_img.convert(cc_rgb)
        img_bgra = np.array(carla_img.raw_data).reshape((carla_img.height,carla_img.width,4))
        img_rgb = np.zeros((carla_img.height,carla_img.width,3))

        img_rgb[:,:,0] = img_bgra[:,:,2]
        img_rgb[:,:,1] = img_bgra[:,:,1]
        img_rgb[:,:,2] = img_bgra[:,:,0]
        img_rgb = np.uint8(img_rgb)
        image = Image.fromarray(img_rgb, 'RGB')
        img_draw = ImageDraw.Draw(image)
        for obj in out_data.values():
            crop = obj['bbox']
            if obj['class'] in class_to_color_dict.keys():
                color_str = class_to_color_dict[obj['class']]
            else:
                color_str = "black"
            u1 = int(crop[0])
            v1 = int(crop[1])
            u2 = int(crop[2])
            v2 = int(crop[3])
            crop_bbox = [(u1,v1),(u2,v2)]
            img_draw.rectangle(crop_bbox, outline =color_str)
        filename = path + 'out_rgb_bbox/%06d.png' % carla_img.frame
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        image.save(filename)


### Use this function to convert depth image (carla.Image) to a depth map in meter
def extract_depth(depth_img):
    depth_img.convert(carla.ColorConverter.Depth)
    depth_meter = np.array(depth_img.raw_data).reshape((depth_img.height,depth_img.width,4))[:,:,0] * 1000 / 255
    return depth_meter

### Use this function to get vehicle's snapshots that can be processed by auto_annotate() function.
def snap_processing(vehiclesActor, worldSnap, veh_check=None):
    vehicles = [] 
    for v in vehiclesActor:
        vid = v.id
        if veh_check is not None and v.id == veh_check:
            continue
        vsnap = worldSnap.find(vid)
        if vsnap is None:
            continue
        vsnap.bounding_box = v.bounding_box
        vsnap.type_id = v.type_id
        vehicles.append(vsnap)
    return vehicles

def snap_processing_manual_bbox(ids, worldSnap, bboxes):
    actors = []
    for v in ids:
        vid = v
        vsnap = worldSnap.find(vid)
        if vsnap is None:
            continue
        vsnap.bounding_box = bboxes[str(vid)][0]
        # Add this extra attribute "loc_given" to have the center
        vsnap.loc_given = bboxes[str(vid)][1]
        vsnap.type_id = "None" # We handle classes differently, we use this so we can handle static non-id'd objects like traffic lights
        actors.append(vsnap)
    return actors
