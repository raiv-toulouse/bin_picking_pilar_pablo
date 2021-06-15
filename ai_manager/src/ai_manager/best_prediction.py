#!/usr/bin/env python


import rospy
import random
from ai_manager.srv import GetBestPrediction, GetBestPredictionResponse
from ai_manager.msg import Prediction
from ai_manager.msg import ListOfPredictions

class NodeBestPrediction:
    def __init__(self):
        self.predictions = []
        msg_list_pred = ListOfPredictions()
        rospy.init_node('best_prediction')
        s = rospy.Service('best_prediction_service', GetBestPrediction, self.get_best_prediction)
        pub = rospy.Publisher('predictions', ListOfPredictions, queue_size=10)
        while not rospy.is_shutdown():
            msg = Prediction()
            msg.proba = random.random()
            msg.x = random.randint(0, 640)
            msg.y = random.randint(0, 480)
            self.predictions.append(msg)
            msg_list_pred.predictions = self.predictions
            pub.publish(msg_list_pred)
            rospy.sleep(0.1)

    def get_best_prediction(self, req):
        last = len(self.predictions)
        return GetBestPredictionResponse(self.predictions[last-1])


if __name__ == '__main__':
    try:
        n = NodeBestPrediction()
    except rospy.ROSInterruptException:
        pass