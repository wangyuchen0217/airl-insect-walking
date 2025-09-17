import math
import numpy as np
# import csv
import os
import pandas as pd

def sysCall_init():
    sim = require('sim')

    self.logging = False

    # CSV file
    
    # self.time_step = 0
    self.csv_row_count = 0
    # self.csv_file = "/home/yuchen/airl-insect-walking/env/Animal06_110919_00_31.csv"
    self.csv_file = "/home/yuchen/airl-insect-walking/env/ds_loopsm.csv"
    self.df = pd.read_csv(self.csv_file)
    self.df_final_line = len(self.df['LF_CTr'])
    print(self.df_final_line)
    
    self.leg_names = ['/FL','/ML','/HL','/FR','/MR','/HR'] # type: ignore
    self.joint_names = ['/m1','/m2','/m3'] # type: ignore
    
    self.FL_joints = []
    self.ML_joints = []
    self.HL_joints = []
    self.FR_joints = []
    self.MR_joints = []
    self.HR_joints = []

    self.FL_foot_contact_sensor = sim.getObject('/forceSensor_FL')
    self.ML_foot_contact_sensor = sim.getObject('/forceSensor_ML')
    self.HL_foot_contact_sensor = sim.getObject('/forceSensor_HL')
    self.FR_foot_contact_sensor = sim.getObject('/forceSensor_FR')
    self.MR_foot_contact_sensor = sim.getObject('/forceSensor_MR')
    self.HR_foot_contact_sensor = sim.getObject('/forceSensor_HR')

    self.foot_contact_sensors = [self.FL_foot_contact_sensor,
                                 self.ML_foot_contact_sensor,
                                 self.HL_foot_contact_sensor,
                                 self.FR_foot_contact_sensor,
                                 self.MR_foot_contact_sensor,
                                 self.HR_foot_contact_sensor]
    
    # Define Object Handle

    self.abdomen = sim.getObject('/abdomen')
    self.head = sim.getObject('/head')
    for i in range(3):
        self.FL_joints.append(sim.getObject(self.joint_names[i] + '_FL'))
        self.ML_joints.append(sim.getObject(self.joint_names[i] + '_ML'))
        self.HL_joints.append(sim.getObject(self.joint_names[i] + '_HL'))
        self.FR_joints.append(sim.getObject(self.joint_names[i] + '_FR'))
        self.MR_joints.append(sim.getObject(self.joint_names[i] + '_MR'))
        self.HR_joints.append(sim.getObject(self.joint_names[i] + '_HR'))
    
    self.foots = []
    self.foot_names = ['/foot_FL', '/foot_ML', '/foot_HL', '/foot_FR', '/foot_MR', '/foot_HR']
    for i in range(6):
        self.foots.append(sim.getObject(self.foot_names[i]))

    self.IMU_robot = sim.getObject('/IMU_robot')
    self.IMU_ref = sim.getObject('/IMU_ref')


    # ====================================================================================== #
    self.sim_time = 0
    self.robot_positions = [0,0,0]
    # self.foots_positions = [0,0,0]

    self.FR_joints_init_position_deg = [30, 9.5, -60]
    self.MR_joints_init_position_deg = [0 ,  -2.5, -60]
    self.HR_joints_init_position_deg = [-40, 9.5,-60]
    self.FL_joints_init_position_deg = [30, 9.5, -60]
    self.ML_joints_init_position_deg = [0 ,  -2.5, -60]
    self.HL_joints_init_position_deg = [-40, 9.5,-60]

    
    self.FL_joints_target = [0, 0, 0]
    self.ML_joints_target = [0, 0, 0]
    self.HL_joints_target = [0, 0, 0]
    self.FR_joints_target = [0, 0, 0]
    self.MR_joints_target = [0, 0, 0]
    self.HR_joints_target = [0, 0, 0]

    self.FL_joints_postions = [0, 0, 0]
    self.ML_joints_postions = [0, 0, 0]
    self.HL_joints_postions = [0, 0, 0]
    self.FR_joints_postions = [0, 0, 0]
    self.MR_joints_postions = [0, 0, 0]
    self.HR_joints_postions = [0, 0, 0]
    

    self.FL_joints_csv = [0, 0, 0]
    self.ML_joints_csv = [0, 0, 0]
    self.HL_joints_csv = [0, 0, 0]
    self.FR_joints_csv = [0, 0, 0]
    self.MR_joints_csv = [0, 0, 0]
    self.HR_joints_csv = [0, 0, 0]

    self.FL_joints_csv_direction = [1, 1, 1]
    self.ML_joints_csv_direction = [1, 1, 1]
    self.HL_joints_csv_direction = [1, 1, 1]
    self.FR_joints_csv_direction = [1, 1, 1]
    self.MR_joints_csv_direction = [1, 1, 1]
    self.HR_joints_csv_direction = [1, 1, 1]

    self.FL_joints_csv_offset = [0, 0, -180]
    self.ML_joints_csv_offset = [0, 0, -180]
    self.HL_joints_csv_offset = [0, 0, -180]
    self.FR_joints_csv_offset = [0, 0, -180]
    self.MR_joints_csv_offset = [0, 0, -180]
    self.HR_joints_csv_offset = [0, 0, -180]
    
    self.FL_joints_prev_pos = [0, 0, 0]
    self.ML_joints_prev_pos = [0, 0, 0]
    self.HL_joints_prev_pos = [0, 0, 0]
    self.FR_joints_prev_pos = [0, 0, 0]
    self.MR_joints_prev_pos = [0, 0, 0]
    self.HR_joints_prev_pos = [0, 0, 0]


    self.FL_joints_init_direction = [-1,  1, 1]
    self.ML_joints_init_direction = [-1,  1, 1]
    self.HL_joints_init_direction = [-1,  1, 1]
    self.FR_joints_init_direction = [1,  -1, -1]
    self.MR_joints_init_direction = [1,  -1, -1]
    self.HR_joints_init_direction = [1,  -1, -1]

    self.FL_force_sensor_values_xyz = [0,0,0]
    self.ML_force_sensor_values_xyz = [0,0,0]
    self.HL_force_sensor_values_xyz = [0,0,0]
    self.FR_force_sensor_values_xyz = [0,0,0]
    self.MR_force_sensor_values_xyz = [0,0,0]
    self.HR_force_sensor_values_xyz = [0,0,0]
    self.force_sensors_values_xyz = [self.FL_force_sensor_values_xyz ,self.ML_force_sensor_values_xyz ,self.HL_force_sensor_values_xyz ,self.FR_force_sensor_values_xyz ,self.MR_force_sensor_values_xyz ,self.HR_force_sensor_values_xyz]
    self.force_sensors_values_z_filter = [0, 0, 0, 0, 0, 0]

    # ================================================= #
    # set Joint torque
    for i in range(3):
        sim.setJointTargetForce(self.FL_joints[i], 2)
        sim.setJointTargetForce(self.ML_joints[i], 2)
        sim.setJointTargetForce(self.HL_joints[i], 2)
        sim.setJointTargetForce(self.FR_joints[i], 2)
        sim.setJointTargetForce(self.MR_joints[i], 2)
        sim.setJointTargetForce(self.HR_joints[i], 2)


    # ================================================== #
    # set CSV
    self.test_data = {'joe':[]}

    self.data_list = {
        'sim_time': [],
        'body_x': [],
        'body_y': [],
        'body_z': [],
        'body_roll': [],
        'body_pitch': [],
        'body_yaw': [],
        'motor_cmd_FL_TC': [],
        'motor_cmd_FL_CF': [],
        'motor_cmd_FL_FT': [],
        'motor_cmd_ML_TC': [],
        'motor_cmd_ML_CF': [],
        'motor_cmd_ML_FT': [],
        'motor_cmd_HL_TC': [],
        'motor_cmd_HL_CF': [],
        'motor_cmd_HL_FT': [],
        'motor_cmd_FR_TC': [],
        'motor_cmd_FR_CF': [],
        'motor_cmd_FR_FT': [],
        'motor_cmd_MR_TC': [],
        'motor_cmd_MR_CF': [],
        'motor_cmd_MR_FT': [],
        'motor_cmd_HR_TC': [],
        'motor_cmd_HR_CF': [],
        'motor_cmd_HR_FT': [],

        'motor_pos_FL_TC': [],
        'motor_pos_FL_CF': [],
        'motor_pos_FL_FT': [],
        'motor_pos_ML_TC': [],
        'motor_pos_ML_CF': [],
        'motor_pos_ML_FT': [],
        'motor_pos_HL_TC': [],
        'motor_pos_HL_CF': [],
        'motor_pos_HL_FT': [],
        'motor_pos_FR_TC': [],
        'motor_pos_FR_CF': [],
        'motor_pos_FR_FT': [],
        'motor_pos_MR_TC': [],
        'motor_pos_MR_CF': [],
        'motor_pos_MR_FT': [],
        'motor_pos_HR_TC': [],
        'motor_pos_HR_CF': [],
        'motor_pos_HR_FT': [],

        'force_FL': [],
        'force_ML': [],
        'force_HL': [],
        'force_FR': [],
        'force_MR': [],
        'force_HR': [],

        'force_FL_x': [],
        'force_FL_y': [],
        'force_FL_z': [],
        'force_ML_x': [],
        'force_ML_y': [],
        'force_ML_z': [],
        'force_HL_x': [],
        'force_HL_y': [],
        'force_HL_z': [],
        'force_FR_x': [],
        'force_FR_y': [],
        'force_FR_z': [],
        'force_MR_x': [],
        'force_MR_y': [],
        'force_MR_z': [],
        'force_HR_x': [],
        'force_HR_y': [],
        'force_HR_z': [],

        'FL_foot_traj_x': [],
        'FL_foot_traj_y': [],
        'FL_foot_traj_z': [],
        'ML_foot_traj_x': [],
        'ML_foot_traj_y': [],
        'ML_foot_traj_z': [],
        'HL_foot_traj_x': [],
        'HL_foot_traj_y': [],
        'HL_foot_traj_z': [],
        'FR_foot_traj_x': [],
        'FR_foot_traj_y': [],
        'FR_foot_traj_z': [],
        'MR_foot_traj_x': [],
        'MR_foot_traj_y': [],
        'MR_foot_traj_z': [],
        'HR_foot_traj_x': [],
        'HR_foot_traj_y': [],
        'HR_foot_traj_z': []
        }

    # ================= Graph ================= #
    self.graph_ThC = sim.getObject('/graph_ThC')
    self.FL_ThC = sim.addGraphStream(self.graph_ThC, 'FL_ThC','m', 0, [1,0,0])
    self.ML_ThC = sim.addGraphStream(self.graph_ThC, 'ML_ThC','m', 0, [0,1,0])
    self.HL_ThC = sim.addGraphStream(self.graph_ThC, 'HL_ThC','m', 0, [0,0,1])
    self.FR_ThC = sim.addGraphStream(self.graph_ThC, 'FR_ThC','m', 0, [1,1,0])
    self.MR_ThC = sim.addGraphStream(self.graph_ThC, 'MR_ThC','m', 0, [0,1,1])
    self.HR_ThC = sim.addGraphStream(self.graph_ThC, 'HR_ThC','m', 0, [1,0,1])

    self.graph_CTr = sim.getObject('/graph_CTr')
    self.FL_CTr = sim.addGraphStream(self.graph_CTr, 'FL_CTr','m', 0, [1,0,0])
    self.ML_CTr = sim.addGraphStream(self.graph_CTr, 'ML_CTr','m', 0, [0,1,0])
    self.HL_CTr = sim.addGraphStream(self.graph_CTr, 'HL_CTr','m', 0, [0,0,1])
    self.FR_CTr = sim.addGraphStream(self.graph_CTr, 'FR_CTr','m', 0, [1,1,0])
    self.MR_CTr = sim.addGraphStream(self.graph_CTr, 'MR_CTr','m', 0, [0,1,1])
    self.HR_CTr = sim.addGraphStream(self.graph_CTr, 'HR_CTr','m', 0, [1,0,1])

    self.graph_FTi = sim.getObject('/graph_FTi')
    self.FL_FTi = sim.addGraphStream(self.graph_FTi, 'FL_FTi','m', 0, [1,0,0])
    self.ML_FTi = sim.addGraphStream(self.graph_FTi, 'ML_FTi','m', 0, [0,1,0])
    self.HL_FTi = sim.addGraphStream(self.graph_FTi, 'HL_FTi','m', 0, [0,0,1])
    self.FR_FTi = sim.addGraphStream(self.graph_FTi, 'FR_FTi','m', 0, [1,1,0])
    self.MR_FTi = sim.addGraphStream(self.graph_FTi, 'MR_FTi','m', 0, [0,1,1])
    self.HR_FTi = sim.addGraphStream(self.graph_FTi, 'HR_FTi','m', 0, [1,0,1])

    self.graph_force = sim.getObject('/graph_force')
    self.FL_force = sim.addGraphStream(self.graph_force, 'FL_force','m', 0, [1,0,0])
    self.ML_force = sim.addGraphStream(self.graph_force, 'ML_force','m', 0, [0,1,0])
    self.HL_force = sim.addGraphStream(self.graph_force, 'HL_force','m', 0, [0,0,1])
    self.FR_force = sim.addGraphStream(self.graph_force, 'FR_force','m', 0, [1,1,0])
    self.MR_force = sim.addGraphStream(self.graph_force, 'MR_force','m', 0, [0,1,1])
    self.HR_force = sim.addGraphStream(self.graph_force, 'HR_force','m', 0, [1,0,1])

    self.graph_foot = sim.getObject('/graph_foot')
    self.FL_foot = sim.addGraphStream(self.graph_foot, 'FL_foot','m', 0, [1,0,0])
    self.ML_foot = sim.addGraphStream(self.graph_foot, 'ML_foot','m', 0, [0,1,0])
    self.HL_foot = sim.addGraphStream(self.graph_foot, 'HL_foot','m', 0, [0,0,1])
    self.FR_foot = sim.addGraphStream(self.graph_foot, 'FR_foot','m', 0, [1,1,0])
    self.MR_foot = sim.addGraphStream(self.graph_foot, 'MR_foot','m', 0, [0,1,1])
    self.HR_foot = sim.addGraphStream(self.graph_foot, 'HR_foot','m', 0, [1,0,1])

    self.graph_pose = sim.getObject('/graph_pose')
    # self.x = sim.addGraphStream(self.graph_pose, 'x','m', 0, [1,0,0])
    # self.y = sim.addGraphStream(self.graph_pose, 'y','m', 0, [0,1,0])
    self.z = sim.addGraphStream(self.graph_pose, 'z','m', 0, [0,0,1])
    self.roll = sim.addGraphStream(self.graph_pose, 'roll','m', 0, [1,1,0])
    self.pitch = sim.addGraphStream(self.graph_pose, 'pitch','m', 0, [0,1,1])
    self.yaw = sim.addGraphStream(self.graph_pose, 'yaw','m', 0, [1,0,1])



    pass
# ============================================================================ #
# ============================================================================ #
#                               sysCall_actuation                              #
# ============================================================================ #
# ============================================================================ #


def sysCall_actuation():
    
    self.test_data['joe'].append(self.csv_row_count)
    # ===================================== #
    #            Read motor command         #
    # ===================================== #
    csv_to_motor()

    # ===================================== #
    #            Command to robot           #
    # ===================================== #
    for i in range(3):
        # #------------------- Standing position -------------------#
        # self.FR_joints_target[i] = math.radians(self.FR_joints_init_position_deg[i]) * self.FR_joints_init_direction[i] 
        # self.MR_joints_target[i] = math.radians(self.MR_joints_init_position_deg[i]) * self.MR_joints_init_direction[i] 
        # self.HR_joints_target[i] = math.radians(self.HR_joints_init_position_deg[i]) * self.HR_joints_init_direction[i] 
        # self.FL_joints_target[i] = math.radians(self.FL_joints_init_position_deg[i]) * self.FL_joints_init_direction[i] 
        # self.ML_joints_target[i] = math.radians(self.ML_joints_init_position_deg[i]) * self.ML_joints_init_direction[i] 
        # self.HL_joints_target[i] = math.radians(self.HL_joints_init_position_deg[i]) * self.HL_joints_init_direction[i] 

        #------------------- Control -------------------#
        self.FL_joints_target[i] = 0 + (math.radians(self.FL_joints_csv[i] * self.FL_joints_csv_direction[i]) + math.radians(self.FL_joints_csv_offset[i])) * self.FL_joints_init_direction[i]
        self.ML_joints_target[i] = 0 + (math.radians(self.ML_joints_csv[i] * self.ML_joints_csv_direction[i]) + math.radians(self.ML_joints_csv_offset[i])) * self.ML_joints_init_direction[i]
        self.HL_joints_target[i] = 0 + (math.radians(self.HL_joints_csv[i] * self.HL_joints_csv_direction[i]) + math.radians(self.HL_joints_csv_offset[i])) * self.HL_joints_init_direction[i]
        self.FR_joints_target[i] = 0 + (math.radians(self.FR_joints_csv[i] * self.FR_joints_csv_direction[i]) + math.radians(self.FR_joints_csv_offset[i])) * self.FR_joints_init_direction[i]
        self.MR_joints_target[i] = 0 + (math.radians(self.MR_joints_csv[i] * self.MR_joints_csv_direction[i]) + math.radians(self.MR_joints_csv_offset[i])) * self.MR_joints_init_direction[i]
        self.HR_joints_target[i] = 0 + (math.radians(self.HR_joints_csv[i] * self.HR_joints_csv_direction[i]) + math.radians(self.HR_joints_csv_offset[i])) * self.HR_joints_init_direction[i]

        # # # # Fix a certain joint
        # self.FL_joints_target[2] = math.radians(-90)
        # self.ML_joints_target[2] = math.radians(-90)
        # self.HL_joints_target[2] = math.radians(-90)
        # self.FR_joints_target[2] = math.radians(90)
        # self.MR_joints_target[2] = math.radians(90)
        # self.HR_joints_target[2] = math.radians(90)

        # # # # For smooth movement
        # self.FL_joints_prev_pos[i] = self.FL_joints_target[i]
        # self.ML_joints_prev_pos[i] = self.ML_joints_target[i]
        # self.HL_joints_prev_pos[i] = self.HL_joints_target[i]
        # self.FR_joints_prev_pos[i] = self.FR_joints_target[i]
        # self.MR_joints_prev_pos[i] = self.MR_joints_target[i]
        # self.HR_joints_prev_pos[i] = self.HR_joints_target[i]

        # # # # drive to the motor 
        sim.setJointTargetPosition(self.FL_joints[i], self.FL_joints_target[i] ) 
        sim.setJointTargetPosition(self.ML_joints[i], self.ML_joints_target[i] ) 
        sim.setJointTargetPosition(self.HL_joints[i], self.HL_joints_target[i] ) 
        sim.setJointTargetPosition(self.FR_joints[i], self.FR_joints_target[i] ) 
        sim.setJointTargetPosition(self.MR_joints[i], self.MR_joints_target[i] ) 
        sim.setJointTargetPosition(self.HR_joints[i], self.HR_joints_target[i] ) 

    pass
# ============================================================================ #
# ============================================================================ #
#                               sysCall_sensing                              #
# ============================================================================ #
# ============================================================================ #
def sysCall_sensing():
    self.sim_time = sim.getSimulationTime()
    # ===================================== #
    #               IMU moment              #
    # ===================================== #
    self.robot_positions = sim.getObjectPosition(self.head)
    # print(self.IMU_ref, self.robot_positions)
    sim.setObjectPosition(self.IMU_ref, [self.robot_positions[0], self.robot_positions[1], 0.4])
    self.robot_orientations = sim.getObjectOrientation(self.IMU_robot, self.IMU_ref)


    # ===================================== #
    #               Foot contact            #
    # ===================================== #
    for i in range(6):
        result, forceVector, torqueVector = sim.readForceSensor(self.foot_contact_sensors[i])
        self.force_sensors_values_xyz[i] = forceVector
        self.force_sensors_values_z_filter[i] = max(0, np.sqrt((forceVector[0])**2 + (forceVector[1])**2 + (forceVector[2])**2) - 0.2)


    FL_foot_traj = sim.getObjectPosition(self.foots[0])
    ML_foot_traj = sim.getObjectPosition(self.foots[1])
    HL_foot_traj = sim.getObjectPosition(self.foots[2])
    FR_foot_traj = sim.getObjectPosition(self.foots[3])
    MR_foot_traj = sim.getObjectPosition(self.foots[4])
    HR_foot_traj = sim.getObjectPosition(self.foots[5])

    

    # ===================================== #
    #               motor position          #
    # ===================================== #
    for i in range(3):
        self.FL_joints_postions[i] = sim.getJointPosition(self.FL_joints[i])
        self.ML_joints_postions[i] = sim.getJointPosition(self.ML_joints[i])
        self.HL_joints_postions[i] = sim.getJointPosition(self.HL_joints[i])
        self.FR_joints_postions[i] = sim.getJointPosition(self.FR_joints[i])
        self.MR_joints_postions[i] = sim.getJointPosition(self.MR_joints[i])
        self.HR_joints_postions[i] = sim.getJointPosition(self.HR_joints[i])


    # ================ graph data ================== #
    sim.setGraphStreamValue(self.graph_ThC, self.FL_ThC, self.FL_joints_postions[0])
    sim.setGraphStreamValue(self.graph_ThC, self.ML_ThC, self.ML_joints_postions[0])
    sim.setGraphStreamValue(self.graph_ThC, self.HL_ThC, self.HL_joints_postions[0])
    sim.setGraphStreamValue(self.graph_ThC, self.FR_ThC, self.FR_joints_postions[0])
    sim.setGraphStreamValue(self.graph_ThC, self.MR_ThC, self.MR_joints_postions[0])
    sim.setGraphStreamValue(self.graph_ThC, self.HR_ThC, self.HR_joints_postions[0])

    sim.setGraphStreamValue(self.graph_CTr, self.FL_CTr, self.FL_joints_postions[1])
    sim.setGraphStreamValue(self.graph_CTr, self.ML_CTr, self.ML_joints_postions[1])
    sim.setGraphStreamValue(self.graph_CTr, self.HL_CTr, self.HL_joints_postions[1])
    sim.setGraphStreamValue(self.graph_CTr, self.FR_CTr, self.FR_joints_postions[1])
    sim.setGraphStreamValue(self.graph_CTr, self.MR_CTr, self.MR_joints_postions[1])
    sim.setGraphStreamValue(self.graph_CTr, self.HR_CTr, self.HR_joints_postions[1])

    sim.setGraphStreamValue(self.graph_FTi, self.FL_FTi, self.FL_joints_postions[2])
    sim.setGraphStreamValue(self.graph_FTi, self.ML_FTi, self.ML_joints_postions[2])
    sim.setGraphStreamValue(self.graph_FTi, self.HL_FTi, self.HL_joints_postions[2])
    sim.setGraphStreamValue(self.graph_FTi, self.FR_FTi, self.FR_joints_postions[2])
    sim.setGraphStreamValue(self.graph_FTi, self.MR_FTi, self.MR_joints_postions[2])
    sim.setGraphStreamValue(self.graph_FTi, self.HR_FTi, self.HR_joints_postions[2])

    sim.setGraphStreamValue(self.graph_force, self.FL_force, self.force_sensors_values_z_filter[0])
    sim.setGraphStreamValue(self.graph_force, self.ML_force, self.force_sensors_values_z_filter[1])
    sim.setGraphStreamValue(self.graph_force, self.HL_force, self.force_sensors_values_z_filter[2])
    sim.setGraphStreamValue(self.graph_force, self.FR_force, self.force_sensors_values_z_filter[3])
    sim.setGraphStreamValue(self.graph_force, self.MR_force, self.force_sensors_values_z_filter[4])
    sim.setGraphStreamValue(self.graph_force, self.HR_force, self.force_sensors_values_z_filter[5])

    sim.setGraphStreamValue(self.graph_foot, self.FL_foot, FL_foot_traj[2])
    sim.setGraphStreamValue(self.graph_foot, self.ML_foot, ML_foot_traj[2])
    sim.setGraphStreamValue(self.graph_foot, self.HL_foot, HL_foot_traj[2])
    sim.setGraphStreamValue(self.graph_foot, self.FR_foot, FR_foot_traj[2])
    sim.setGraphStreamValue(self.graph_foot, self.MR_foot, MR_foot_traj[2])
    sim.setGraphStreamValue(self.graph_foot, self.HR_foot, HR_foot_traj[2])

    # sim.setGraphStreamValue(self.graph_pose, self.x, self.robot_positions[0])
    # sim.setGraphStreamValue(self.graph_pose, self.y, self.robot_positions[1])
    sim.setGraphStreamValue(self.graph_pose, self.z, self.robot_positions[2])
    sim.setGraphStreamValue(self.graph_pose, self.roll, self.robot_orientations[0])
    sim.setGraphStreamValue(self.graph_pose, self.pitch, self.robot_orientations[1])
    sim.setGraphStreamValue(self.graph_pose, self.yaw, self.robot_orientations[2])

    # ================ logging data ================== #
    if self.logging:
        self.data_list['sim_time'].append(self.sim_time)
        self.data_list['body_x'].append(self.robot_positions[0])
        self.data_list['body_y'].append(self.robot_positions[1])
        self.data_list['body_z'].append(self.robot_positions[2])
        self.data_list['body_roll'].append(self.robot_orientations[0])
        self.data_list['body_pitch'].append(self.robot_orientations[1])
        self.data_list['body_yaw'].append( self.robot_orientations[2])
        self.data_list['motor_cmd_FL_TC'].append(self.FL_joints_target[0])
        self.data_list['motor_cmd_FL_CF'].append(self.FL_joints_target[1])
        self.data_list['motor_cmd_FL_FT'].append(self.FL_joints_target[2])
        self.data_list['motor_cmd_ML_TC'].append(self.ML_joints_target[0])
        self.data_list['motor_cmd_ML_CF'].append(self.ML_joints_target[1])
        self.data_list['motor_cmd_ML_FT'].append(self.ML_joints_target[2])
        self.data_list['motor_cmd_HL_TC'].append(self.HL_joints_target[0])
        self.data_list['motor_cmd_HL_CF'].append(self.HL_joints_target[1])
        self.data_list['motor_cmd_HL_FT'].append(self.HL_joints_target[2])
        self.data_list['motor_cmd_FR_TC'].append(self.FR_joints_target[0])
        self.data_list['motor_cmd_FR_CF'].append(self.FR_joints_target[1])
        self.data_list['motor_cmd_FR_FT'].append(self.FR_joints_target[2])
        self.data_list['motor_cmd_MR_TC'].append(self.MR_joints_target[0])
        self.data_list['motor_cmd_MR_CF'].append(self.MR_joints_target[1])
        self.data_list['motor_cmd_MR_FT'].append(self.MR_joints_target[2])
        self.data_list['motor_cmd_HR_TC'].append(self.HR_joints_target[0])
        self.data_list['motor_cmd_HR_CF'].append(self.HR_joints_target[1])
        self.data_list['motor_cmd_HR_FT'].append(self.HR_joints_target[2])

        self.data_list['motor_pos_FL_TC'].append(self.FL_joints_postions[0])
        self.data_list['motor_pos_FL_CF'].append(self.FL_joints_postions[1])
        self.data_list['motor_pos_FL_FT'].append(self.FL_joints_postions[2])
        self.data_list['motor_pos_ML_TC'].append(self.ML_joints_postions[0])
        self.data_list['motor_pos_ML_CF'].append(self.ML_joints_postions[1])
        self.data_list['motor_pos_ML_FT'].append(self.ML_joints_postions[2])
        self.data_list['motor_pos_HL_TC'].append(self.HL_joints_postions[0])
        self.data_list['motor_pos_HL_CF'].append(self.HL_joints_postions[1])
        self.data_list['motor_pos_HL_FT'].append(self.HL_joints_postions[2])
        self.data_list['motor_pos_FR_TC'].append(self.FR_joints_postions[0])
        self.data_list['motor_pos_FR_CF'].append(self.FR_joints_postions[1])
        self.data_list['motor_pos_FR_FT'].append(self.FR_joints_postions[2])
        self.data_list['motor_pos_MR_TC'].append(self.MR_joints_postions[0])
        self.data_list['motor_pos_MR_CF'].append(self.MR_joints_postions[1])
        self.data_list['motor_pos_MR_FT'].append(self.MR_joints_postions[2])
        self.data_list['motor_pos_HR_TC'].append(self.HR_joints_postions[0])
        self.data_list['motor_pos_HR_CF'].append(self.HR_joints_postions[1])
        self.data_list['motor_pos_HR_FT'].append(self.HR_joints_postions[2])

        self.data_list['force_FL'].append(self.force_sensors_values_z_filter[0])
        self.data_list['force_ML'].append(self.force_sensors_values_z_filter[1])
        self.data_list['force_HL'].append(self.force_sensors_values_z_filter[2])
        self.data_list['force_FR'].append(self.force_sensors_values_z_filter[3])
        self.data_list['force_MR'].append(self.force_sensors_values_z_filter[4])
        self.data_list['force_HR'].append(self.force_sensors_values_z_filter[5])

        self.data_list['force_FL_x'].append(self.force_sensors_values_xyz[0][0])
        self.data_list['force_FL_y'].append(self.force_sensors_values_xyz[0][1])
        self.data_list['force_FL_z'].append(self.force_sensors_values_xyz[0][2])
        self.data_list['force_ML_x'].append(self.force_sensors_values_xyz[1][0])
        self.data_list['force_ML_y'].append(self.force_sensors_values_xyz[1][1])
        self.data_list['force_ML_z'].append(self.force_sensors_values_xyz[1][2])
        self.data_list['force_HL_x'].append(self.force_sensors_values_xyz[2][0])
        self.data_list['force_HL_y'].append(self.force_sensors_values_xyz[2][1])
        self.data_list['force_HL_z'].append(self.force_sensors_values_xyz[2][2])
        self.data_list['force_FR_x'].append(self.force_sensors_values_xyz[3][0])
        self.data_list['force_FR_y'].append(self.force_sensors_values_xyz[3][1])
        self.data_list['force_FR_z'].append(self.force_sensors_values_xyz[3][2])
        self.data_list['force_MR_x'].append(self.force_sensors_values_xyz[4][0])
        self.data_list['force_MR_y'].append(self.force_sensors_values_xyz[4][1])
        self.data_list['force_MR_z'].append(self.force_sensors_values_xyz[4][2])
        self.data_list['force_HR_x'].append(self.force_sensors_values_xyz[5][0])
        self.data_list['force_HR_y'].append(self.force_sensors_values_xyz[5][1])
        self.data_list['force_HR_z'].append(self.force_sensors_values_xyz[5][2])


        self.data_list['FL_foot_traj_x'].append(FL_foot_traj[0])
        self.data_list['FL_foot_traj_y'].append(FL_foot_traj[1])
        self.data_list['FL_foot_traj_z'].append(FL_foot_traj[2])
        self.data_list['ML_foot_traj_x'].append(ML_foot_traj[0])
        self.data_list['ML_foot_traj_y'].append(ML_foot_traj[1])
        self.data_list['ML_foot_traj_z'].append(ML_foot_traj[2])
        self.data_list['HL_foot_traj_x'].append(HL_foot_traj[0])
        self.data_list['HL_foot_traj_y'].append(HL_foot_traj[1])
        self.data_list['HL_foot_traj_z'].append(HL_foot_traj[2])
        self.data_list['FR_foot_traj_x'].append(FR_foot_traj[0])
        self.data_list['FR_foot_traj_y'].append(FR_foot_traj[1])
        self.data_list['FR_foot_traj_z'].append(FR_foot_traj[2])
        self.data_list['MR_foot_traj_x'].append(MR_foot_traj[0])
        self.data_list['MR_foot_traj_y'].append(MR_foot_traj[1])
        self.data_list['MR_foot_traj_z'].append(MR_foot_traj[2])
        self.data_list['HR_foot_traj_x'].append(HR_foot_traj[0])
        self.data_list['HR_foot_traj_y'].append(HR_foot_traj[1])
        self.data_list['HR_foot_traj_z'].append(HR_foot_traj[2])

    pass
# ============================================================================ #
# ============================================================================ #
#                               sysCall_cleanup                              #
# ============================================================================ #
# ============================================================================ #

def sysCall_cleanup():
    # do some clean-up here
    if self.logging:
        save_data = pd.DataFrame(self.data_list)
        save_data.to_csv('/home/yuchen/airl-insect-walking/expert29.csv', index=False)

    pass


# ============================================================================ #
# ============================================================================ #
# ============================================================================ #
#                                   Functions                                  #
# ============================================================================ #
# ============================================================================ #
# ============================================================================ #


# See the user manual or the available code snippets for additional callback functions and details

def csv_to_motor():

    # ------------original csv reading method ------------- #
    # if self.csv_row_count < 1371 or self.csv_row_count > 2070:
    #     self.csv_row_count = 1371

    # ------------downsampled csv reading method ------------- #
    # if self.csv_row_count < 2 or self.csv_row_count > 64:
    #     self.csv_row_count = 1

    # if self.csv_row_count > self.df_final_line-3:
    #     sim.stopSimulation()
    # else:
    #     self.csv_row_count += 1

    # ------------no looping method ------------- #
    if self.csv_row_count < 2:
        self.csv_row_count = 2
    
    if self.csv_row_count > 64:
        sim.stopSimulation()
    else:
        self.csv_row_count += 1


    #     self.time_step += 1

    # if self.time_step == 1:
    #     noise = np.random.uniform(-5, 5, size=18)
    #     print("Adding noise to the joints: ", noise)
    # else:
    #     noise = np.zeros(18)

    print(self.csv_row_count)


    self.FL_joints_csv[1] = self.df['LF_CTr'][self.csv_row_count] #+ noise[0]
    self.ML_joints_csv[1] = self.df['LM_CTr'][self.csv_row_count] #+ noise[1]	
    self.HL_joints_csv[1] = self.df['LH_CTr'][self.csv_row_count] #+ noise[2]	
    self.FR_joints_csv[1] = self.df['RF_CTr'][self.csv_row_count] #+ noise[3]	
    self.MR_joints_csv[1] = self.df['RM_CTr'][self.csv_row_count] #+ noise[4]	
    self.HR_joints_csv[1] = self.df['RH_CTr'][self.csv_row_count] #+ noise[5]	
    self.FL_joints_csv[0] = self.df['LF_ThC'][self.csv_row_count] #+ noise[6]	
    self.ML_joints_csv[0] = self.df['LM_ThC'][self.csv_row_count] #+ noise[7]	
    self.HL_joints_csv[0] = self.df['LH_ThC'][self.csv_row_count] #+ noise[8]	
    self.FR_joints_csv[0] = self.df['RF_ThC'][self.csv_row_count] #+ noise[9]	
    self.MR_joints_csv[0] = self.df['RM_ThC'][self.csv_row_count] #+ noise[10]	
    self.HR_joints_csv[0] = self.df['RH_ThC'][self.csv_row_count] #+ noise[11]	
    self.FL_joints_csv[2] = self.df['LF_FTi'][self.csv_row_count] #+ noise[12]	
    self.ML_joints_csv[2] = self.df['LM_FTi'][self.csv_row_count] #+ noise[13]	
    self.HL_joints_csv[2] = self.df['LH_FTi'][self.csv_row_count] #+ noise[14]	
    self.FR_joints_csv[2] = self.df['RF_FTi'][self.csv_row_count] #+ noise[15]	
    self.MR_joints_csv[2] = self.df['RM_FTi'][self.csv_row_count] #+ noise[16]	
    self.HR_joints_csv[2] = self.df['RH_FTi'][self.csv_row_count] #+ noise[17]

    # print(row)


def force_filter(force_val):
    force = np.tanh(relu(-force_val))
    return force
    
def relu(val):
    if val < 0:
        return 0
    else:
        return val
    
def sigmoid(val):
    return 1/(1 + np.exp(-val)) 