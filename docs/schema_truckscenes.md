TruckScenes schema
==========
This document describes the database schema used in MAN TruckScenes.
All annotations and meta data (including calibration, taxonomy, vehicle coordinates etc.) are covered in a relational database.
The database tables are listed below.
Every row can be identified by its unique primary key `token`.
Foreign keys such as `sample_token` may be used to link to the `token` of the table `sample`.

attribute
---------
An attribute is a property of an instance that can change while the category remains the same.
Example: a vehicle being parked/stopped/moving, and whether or not a bicycle has a rider.
```
attribute {
   "token":                   <str> -- Unique record identifier.
   "name":                    <str> -- Attribute name.
   "description":             <str> -- Attribute description.
}
```

calibrated_sensor
---------
Definition of a particular sensor (lidar/radar/camera) as calibrated on a particular vehicle.
All extrinsic parameters are given with respect to the ego vehicle body frame.
All camera images come undistorted and rectified.
```
calibrated_sensor {
   "token":                   <str> -- Unique record identifier.
   "sensor_token":            <str> -- Foreign key pointing to the sensor type.
   "translation":             <float> [3] -- Coordinate system origin in meters: x, y, z.
   "rotation":                <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
   "camera_intrinsic":        <float> [3, 3] -- Intrinsic camera calibration. Empty for sensors that are not cameras.
}
```

category
---------
Taxonomy of object categories (e.g. vehicle, human). 
Subcategories are delineated by a period (e.g. `human.pedestrian.adult`).
```
category {
   "token":                   <str> -- Unique record identifier.
   "name":                    <str> -- Category name. Subcategories indicated by period.
   "description":             <str> -- Category description.
   "index":                   <int> -- The index of the label used for efficiency reasons.
}
```

ego_motion_cabin
---------
Ego vehicle cabin motion at a particular timestamp. Given with respect to vehicle coordinate system.
The cabin movement can be different from the chassis movement.
```
ego_motion_cabin {
   "token":                   <str> -- Unique record identifier.
   "timestamp":               <int> -- Unix time stamp.
   "vx":                      <float> -- Velocity in x direction given in meters per second (m/s).
   "vy":                      <float> -- Velocity in y direction given in meters per second (m/s).
   "vz":                      <float> -- Velocity in z direction given in meters per second (m/s).
   "ax":                      <float> -- Acceleration in x direction given in meters per second squared (m/s^2).
   "ay":                      <float> -- Acceleration in y direction given in meters per second squared (m/s^2).
   "az":                      <float> -- Acceleration in z direction given in meters per second squared (m/s^2).
   "yaw":                     <float> -- Yaw angle around the z axis given in rad.
   "pitch":                   <float> -- Pitch angle around the y axis given in rad.
   "roll":                    <float> -- Roll angle around the x axis given in rad.
   "yaw_rate":                <float> -- Yaw rate around the z axis given in rad per second.
   "pitch_rate":              <float> -- Pitch rate around the z axis given in rad per second.
   "roll_rate":               <float> -- Roll rate around the z axis given in rad per second.
}
```

ego_motion_chassis
---------
Ego vehicle chassis motion at a particular timestamp. Given with respect to vehicle coordinate system.
The cabin movement can be different from the chassis movement.
```
ego_motion_chassis {
   "token":                   <str> -- Unique record identifier.
   "timestamp":               <int> -- Unix time stamp.
   "vx":                      <float> -- Velocity in x direction given in meters per second (m/s).
   "vy":                      <float> -- Velocity in y direction given in meters per second (m/s).
   "vz":                      <float> -- Velocity in z direction given in meters per second (m/s).
   "ax":                      <float> -- Acceleration in x direction given in meters per second squared (m/s^2).
   "ay":                      <float> -- Acceleration in y direction given in meters per second squared (m/s^2).
   "az":                      <float> -- Acceleration in z direction given in meters per second squared (m/s^2).
   "yaw":                     <float> -- Yaw angle around the z axis given in rad.
   "pitch":                   <float> -- Pitch angle around the y axis given in rad.
   "roll":                    <float> -- Roll angle around the x axis given in rad.
   "yaw_rate":                <float> -- Yaw rate around the z axis given in rad per second.
   "pitch_rate":              <float> -- Pitch rate around the z axis given in rad per second.
   "roll_rate":               <float> -- Roll rate around the z axis given in rad per second.
}
```

ego_pose
---------
Ego vehicle pose at a particular timestamp. Given with respect to global coordinate system in UTM-WGS84 coordinates mapped to cell U32.
```
ego_pose {
   "token":                   <str> -- Unique record identifier.
   "translation":             <float> [3] -- Coordinate system origin in meters: x, y, z. Note that z is always 0.
   "rotation":                <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
   "timestamp":               <int> -- Unix time stamp.
}
```

instance
---------
An object instance, e.g. particular vehicle.
This table is an enumeration of all object instances we observed.
Note that instances are not tracked across scenes.
```
instance {
   "token":                   <str> -- Unique record identifier.
   "category_token":          <str> -- Foreign key pointing to the object category.
   "nbr_annotations":         <int> -- Number of annotations of this instance.
   "first_annotation_token":  <str> -- Foreign key. Points to the first annotation of this instance.
   "last_annotation_token":   <str> -- Foreign key. Points to the last annotation of this instance.
}
```

sample
---------
A sample is an annotated keyframe at 2 Hz.
The data is collected at (approximately) the same timestamp as sample_data marked as keyframes.
```
sample {
   "token":                   <str> -- Unique record identifier.
   "timestamp":               <int> -- Unix time stamp.
   "scene_token":             <str> -- Foreign key pointing to the scene.
   "next":                    <str> -- Foreign key. Sample that follows this in time. Empty if end of scene.
   "prev":                    <str> -- Foreign key. Sample that precedes this in time. Empty if start of scene.
}
```

sample_annotation
---------
A bounding box defining the position of an object seen in a sample.
All location data is given with respect to the global coordinate system.
```
sample_annotation {
   "token":                   <str> -- Unique record identifier.
   "sample_token":            <str> -- Foreign key. NOTE: this points to a sample NOT a sample_data since annotations are done on the sample level taking all relevant sample_data into account.
   "instance_token":          <str> -- Foreign key. Which object instance is this annotating. An instance can have multiple annotations over time.
   "attribute_tokens":        <str> [n] -- Foreign keys. List of attributes for this annotation. Attributes can change over time, so they belong here, not in the instance table.
   "visibility_token":        <str> -- Foreign key. Visibility may also change over time. If no visibility is annotated, the token is an empty string.
   "translation":             <float> [3] -- Bounding box location in meters as center_x, center_y, center_z.
   "size":                    <float> [3] -- Bounding box size in meters as width, length, height.
   "rotation":                <float> [4] -- Bounding box orientation as quaternion: w, x, y, z.
   "num_lidar_pts":           <int> -- Number of lidar points in this box. Points are counted during the lidar sweep identified with this sample.
   "num_radar_pts":           <int> -- Number of radar points in this box. Points are counted during the radar sweep identified with this sample. This number is summed across all radar sensors without any invalid point filtering.
   "next":                    <str> -- Foreign key. Sample annotation from the same object instance that follows this in time. Empty if this is the last annotation for this object.
   "prev":                    <str> -- Foreign key. Sample annotation from the same object instance that precedes this in time. Empty if this is the first annotation for this object.
}
```

sample_data
---------
A sensor data e.g. image, point cloud or radar return. 
For sample_data with is_key_frame=True, the time-stamps should be very close to the sample it points to.
For non key-frames the sample_data points to the sample that follows closest in time.
```
sample_data {
   "token":                   <str> -- Unique record identifier.
   "sample_token":            <str> -- Foreign key. Sample to which this sample_data is associated.
   "ego_pose_token":          <str> -- Foreign key.
   "calibrated_sensor_token": <str> -- Foreign key.
   "filename":                <str> -- Relative path to data-blob on disk.
   "fileformat":              <str> -- Data file format.
   "width":                   <int> -- If the sample data is an image, this is the image width in pixels.
   "height":                  <int> -- If the sample data is an image, this is the image height in pixels.
   "timestamp":               <int> -- Unix time stamp.
   "is_key_frame":            <bool> -- True if sample_data is part of key_frame, else False.
   "next":                    <str> -- Foreign key. Sample data from the same sensor that follows this in time. Empty if end of scene.
   "prev":                    <str> -- Foreign key. Sample data from the same sensor that precedes this in time. Empty if start of scene.
}
```

scene
---------
A scene is a 20s long sequence of consecutive frames. 
Multiple scenes can come from the same measurement drive. 
Note that object identities (instance tokens) are not preserved across scenes.
```
scene {
   "token":                   <str> -- Unique record identifier.
   "name":                    <str> -- Short string identifier.
   "description":             <str> -- List of scene tags according to seven distinct categories separated by semicolon.
   "log_token":               <str> -- Foreign key. Always empty.
   "nbr_samples":             <int> -- Number of samples in this scene.
   "first_sample_token":      <str> -- Foreign key. Points to the first sample in scene.
   "last_sample_token":       <str> -- Foreign key. Points to the last sample in scene.
}
```

sensor
---------
A specific sensor type.
```
sensor {
   "token":                   <str> -- Unique record identifier.
   "channel":                 <str> -- Sensor channel name.
   "modality":                <str> {camera, lidar, radar} -- Sensor modality. Supports category(ies) in brackets.
}
```

visibility
---------
The visibility of an instance is the fraction of annotation visible in all 4 images. Binned into 4 bins 0-40%, 40-60%, 60-80% and 80-100%.
```
visibility {
   "token":                   <str> -- Unique record identifier.
   "level":                   <int> -- Visibility level.
   "description":             <str> -- Description of visibility level.
}
```




Copied and adapted from [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)