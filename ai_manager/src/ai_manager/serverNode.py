import rospy
from ai_manager.srv import GetBestPrediction, GetBestPredictionResponse


def handle_prediction(req):

    print("liaison établie")

if __name__ == "__main__":

    rospy.init_node("serverNode")
    s = rospy.Service("get_best_prediction_test", GetBestPrediction, handle_prediction)
    print("liaison prête")
    rospy.spin()

