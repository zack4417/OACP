import threading
import rospy
from consensus_bphto.msg import States, Controls  

class MinimalSubscriber:
    def __init__(self):   
        # Subscriptions
        self.subscription = rospy.Subscriber('ego_vehicle_cmds', Controls, self.listener_callback)
        
        # Publishers 
        self.publisher_ = rospy.Publisher('ego_vehicle_vel_bound', States, queue_size=10, latch=True)

        # Initialize velocity boundaries with defaults
        self.vxc_min = 0.0
        self.vxc_max = 8.0

        # Start the input thread for updating boundaries
        self.input_thread = threading.Thread(target=self.update_velocity_boundaries)
        self.input_thread.daemon = True  # Ensure the thread exits when the program stops
        self.input_thread.start()

    def listener_callback(self, msg): 
        # rospy.loginfo("Received ego_vehicle_cmds message")
        pass

    def update_velocity_boundaries(self):
        """
        Continuously prompt the user for vxc_min and vxc_max values.
        Runs in a separate thread.
        """
        while not rospy.is_shutdown():
            try:
                # Get user input for the velocity boundaries
                new_vxc_min = float(input("Enter new vxc_min: "))
                new_vxc_max = float(input("Enter new vxc_max: "))

                # Validate the input to ensure vxc_max > vxc_min
                if new_vxc_max > new_vxc_min:
                    self.vxc_min = new_vxc_min
                    self.vxc_max = new_vxc_max
                    rospy.loginfo(f"Updated velocity boundaries: vxc_min={self.vxc_min}, vxc_max={self.vxc_max}")
                else:
                    rospy.logwarn("Invalid input: vxc_max must be greater than vxc_min.")
            except ValueError:
                rospy.logwarn("Invalid input. Please enter numeric values.")

    def timer_callback(self, event):
        """
        Publish the current velocity boundaries at a regular interval.
        """
        msg = States() 
        msg.vxc_min = self.vxc_min
        msg.vxc_max = self.vxc_max
        self.publisher_.publish(msg)
        # rospy.loginfo(f"Published velocity boundaries: vxc_min={self.vxc_min}, vxc_max={self.vxc_max}")

def main():
    rospy.init_node('ev_vel_test', anonymous=True)
    try:
        # Instantiate the MinimalSubscriber class
        minimal_subscriber = MinimalSubscriber()
        
        # Set a timer for periodic publishing
        rospy.Timer(rospy.Duration(1), minimal_subscriber.timer_callback)  # 1-second interval

        # Keep the ROS node running
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()