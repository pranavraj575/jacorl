#!/usr/bin/env python
###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2019 Kinova inc. All rights reserved.
#
# This software may be modified and distributed 
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import rospy
import time

from kortex_driver.srv import *
from kortex_driver.msg import *


class ExampleFullArmMovement:
    def __init__(self):
        try:
            rospy.init_node('example_full_arm_movement_python')

            self.HOME_ACTION_IDENTIFIER = 2

            # Get node params
            self.robot_name = rospy.get_param('~robot_name', "my_gen3")
            self.degrees_of_freedom = rospy.get_param("/" + self.robot_name + "/degrees_of_freedom", 6)
            self.is_gripper_present = rospy.get_param("/" + self.robot_name + "/is_gripper_present", False)

            rospy.loginfo("Using robot_name " + self.robot_name + " , robot has " + str(self.degrees_of_freedom) + " degrees of freedom and is_gripper_present is " + str(self.is_gripper_present))

            # Init the action topic subscriber
            self.action_topic_sub = rospy.Subscriber("/" + self.robot_name + "/action_topic", ActionNotification, self.cb_action_topic)
            self.last_action_notif_type = None

            # Init the services
            clear_faults_full_name = '/' + self.robot_name + '/base/clear_faults'
            rospy.wait_for_service(clear_faults_full_name)
            self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

            read_action_full_name = '/' + self.robot_name + '/base/read_action'
            rospy.wait_for_service(read_action_full_name)
            self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)

            execute_action_full_name = '/' + self.robot_name + '/base/execute_action'
            rospy.wait_for_service(execute_action_full_name)
            self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)

            set_cartesian_reference_frame_full_name = '/' + self.robot_name + '/control_config/set_cartesian_reference_frame'
            rospy.wait_for_service(set_cartesian_reference_frame_full_name)
            self.set_cartesian_reference_frame = rospy.ServiceProxy(set_cartesian_reference_frame_full_name, SetCartesianReferenceFrame)

            send_gripper_command_full_name = '/' + self.robot_name + '/base/send_gripper_command'
            rospy.wait_for_service(send_gripper_command_full_name)
            self.send_gripper_command = rospy.ServiceProxy(send_gripper_command_full_name, SendGripperCommand)

            activate_publishing_of_action_notification_full_name = '/' + self.robot_name + '/base/activate_publishing_of_action_topic'
            rospy.wait_for_service(activate_publishing_of_action_notification_full_name)
            self.activate_publishing_of_action_notification = rospy.ServiceProxy(activate_publishing_of_action_notification_full_name, OnNotificationActionTopic)
        
            get_product_configuration_full_name = '/' + self.robot_name + '/base/get_product_configuration'
            rospy.wait_for_service(get_product_configuration_full_name)
            self.get_product_configuration = rospy.ServiceProxy(get_product_configuration_full_name, GetProductConfiguration)

            validate_waypoint_list_full_name = '/' + self.robot_name + '/base/validate_waypoint_list'
            rospy.wait_for_service(validate_waypoint_list_full_name)
            self.validate_waypoint_list = rospy.ServiceProxy(validate_waypoint_list_full_name, ValidateWaypointList)
        except:
            self.is_init_success = False
        else:
            self.is_init_success = True

    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event
    

    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if (self.last_action_notif_type == ActionEvent.ACTION_END):
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif (self.last_action_notif_type == ActionEvent.ACTION_ABORT):
                rospy.loginfo("Received ACTION_ABORT notification")
                return False
            else:
                time.sleep(0.01)

    def example_subscribe_to_a_robot_notification(self):
        # Activate the publishing of the ActionNotification
        req = OnNotificationActionTopicRequest()
        rospy.loginfo("Activating the action notifications...")
        try:
            self.activate_publishing_of_action_notification(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call OnNotificationActionTopic")
            return False
        else:
            rospy.loginfo("Successfully activated the Action Notifications!")

        rospy.sleep(1.0)
        return True

    def example_clear_faults(self):
        try:
            self.clear_faults()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        else:
            rospy.loginfo("Cleared the faults successfully")
            rospy.sleep(2.5)
            return True
    def move_arm(self,angles):
        # moves robot arm to the angles, requires a list of 6 (list of #dof)
        self.last_action_notif_type = None
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        try:
            res = self.read_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ReadAction")
            return False
        else:
            # What we just read is the input of the ExecuteAction service
            req = ExecuteActionRequest()
            req.input = res.output
            rospy.loginfo("action:"+str(angles))
            try:
                i=0
                for obj in (req.input.oneof_action_parameters.reach_joint_angles[0].joint_angles.joint_angles):
                    
                    obj.value=angles[i] # now robot thinks "angles" is the home position
                    # yes this is janky
                    i+=1
                self.execute_action(req)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ExecuteAction")
                return False
            else:
                return self.wait_for_action_end_or_abort()



    def move_fingy(self, value):
        # Initialize the request
        # Close the gripper
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        rospy.loginfo("Sending the gripper command...")

        # Call the service 
        try:
            self.send_gripper_command(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SendGripperCommand")
            return False
        else:
            time.sleep(0.5)
            return True

    def main(self):
        # For testing purposes
        success = self.is_init_success
        try:
            rospy.delete_param("/kortex_examples_test_results/full_arm_movement_python")
        except:
            pass

        if success:
            #*******************************************************************************
            # Make sure to clear the robot's faults else it won't move if it's already in fault
            success &= self.example_clear_faults()
            #*******************************************************************************
            
            #*******************************************************************************
            # Activate the action notifications
            success &= self.example_subscribe_to_a_robot_notification()
            #*******************************************************************************

            #*******************************************************************************
            # Move the robot to the Home position with an Action
            success &= self.example_home_the_robot()
            #*******************************************************************************

            #*******************************************************************************
            # Example of gripper command
            # Let's fully open the gripper
            if self.is_gripper_present:
                success &= self.example_send_gripper_command(0.0)
            else:
                rospy.logwarn("No gripper is present on the arm.")  
            success &= self.example_send_cartesian_pose()
            success &= self.example_send_gripper_command(1.0)
            success &= self.example_send_gripper_command(0.0)
            success &= self.example_send_gripper_command(1.0)
            success &= self.example_send_gripper_command(0.0)
            success &= self.example_send_gripper_command(1.0)
            success &= self.example_send_gripper_command(0.0)
            success &= self.example_send_joint_angles()
            #*******************************************************************************

            #*******************************************************************************
            # Set the reference frame to "Mixed"
            success &= self.example_set_cartesian_reference_frame()

            # Example of cartesian pose
            # Let's make it move in Z
            success &= self.example_send_cartesian_pose()
            #*******************************************************************************

            #*******************************************************************************
            # Example of angular position
            # Let's send the arm to vertical position
            success &= self.example_send_joint_angles()
            #*******************************************************************************

            #*******************************************************************************
            # Example of gripper command
            # Let's close the gripper at 50%
            if self.is_gripper_present:
                success &= self.example_send_gripper_command(0.5)
            else:
                rospy.logwarn("No gripper is present on the arm.")    
            #*******************************************************************************
        
            #*******************************************************************************
            # Move the robot to the Home position with an Action
            success &= self.example_home_the_robot()
            #*******************************************************************************

            #*******************************************************************************
            # Example of waypoint
            # Let's move the arm
            success &= self.example_cartesian_waypoint_action()

            #*******************************************************************************
            # Move the robot to the Home position with an Action
            success &= self.example_home_the_robot()
            #*******************************************************************************

        # For testing purposes
        rospy.set_param("/kortex_examples_test_results/full_arm_movement_python", success)

        if not success:
            rospy.logerr("The example encountered an error.")


if __name__ == "__main__":
    ex = ExampleFullArmMovement()
    ex.main()
