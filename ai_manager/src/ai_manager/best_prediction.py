#!/usr/bin/env python


import rospy
from ai_manager.srv import GetBestPrediction, GetBestPredictionResponse

def get_best_prediction(req):
    predictions = rospy.get_param("/list_of_predictions")
    rospy.loginfo("%s is %s", rospy.resolve_name('/list_of_predictions'), predictions)
    rospy.set_param('list_of_predictions', [])
    return GetBestPredictionResponse(predictions[0][1], predictions[0][2])


if __name__ == '__main__':
    try:
        rospy.init_node('best_prediction')
        s = rospy.Service('best_prediction_service', GetBestPrediction, get_best_prediction)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass