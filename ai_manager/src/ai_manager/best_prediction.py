#!/usr/bin/env python

import rospy
import random
import math
from ai_manager.srv import GetBestPrediction, GetBestPredictionResponse
from ai_manager.msg import Prediction
from ai_manager.msg import ListOfPredictions

INVALIDATION_RADIUS = 150  # When a prediction is selected, we invalidate all the previous predictions in this radius


class NodeBestPrediction:
    """
    This node is both a service and a publisher.
    * Service best_prediction_service : use a GetBestPrediction message (no input message and a ListOfPredictions output message)
    When this service is called, return the current best prediction and invalidate all the predictions in its neighborhood.
    * Publisher : publish on the 'predictions' topic a ListOfPredictions message
    """
    def __init__(self):
        self.predictions = []
        msg_list_pred = ListOfPredictions()
        rospy.init_node('best_prediction')
        s = rospy.Service('best_prediction_service', GetBestPrediction, self._get_best_prediction)
        pub = rospy.Publisher('predictions', ListOfPredictions, queue_size=10)
        while not rospy.is_shutdown():
            msg = Prediction()
            msg.proba = random.random()
            msg.x = random.randint(0, 640)
            msg.y = random.randint(0, 480)
            self.predictions.append(msg)
            msg_list_pred.predictions = self.predictions
            pub.publish(msg_list_pred)
            rospy.sleep(0.001)

    def _get_best_prediction(self, req):
        # Find best prediction
        best_pred = self.predictions[0]
        for pred in self.predictions:
            if pred.proba > best_pred.proba:
                best_pred = pred
        self._invalidate_neighborhood(best_pred.x, best_pred.y)
        return GetBestPredictionResponse(best_pred)

    def _invalidate_neighborhood(self, x, y):
        """
        Invalidate (remove) all the predictions in a circle of radius INVALIDATION_RADIUS centered in (x,y)
        :param x:
        :param y:
        :return:
        """
        self.predictions = [pred for pred in self.predictions if self._distance(pred, x, y) > INVALIDATION_RADIUS]
        
    def _distance(self, pred, px, py):
        """
        Compute the distance between the prediction point (pred.x, pred.y) and the pick location (px, py)
        :param pred: 
        :param px: 
        :param py: 
        :return: 
        """
        dx = (pred.x - px) * (pred.x - px) 
        dy = (pred.y - py) * (pred.y - py) 
        return math.sqrt(dx + dy)

if __name__ == '__main__':
    try:
        n = NodeBestPrediction()
    except rospy.ROSInterruptException:
        pass