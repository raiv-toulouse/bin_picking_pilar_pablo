# coding: utf-8
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt

class Reglage(QWidget):
    def __init__(self,min,max,text,val=0):
        super(Reglage,self).__init__()
        #loadUi('reglage.ui', self)
        # Définition de l'interface
        hLayout = QHBoxLayout()
        self.label = QLabel()
        hLayout.addWidget(self.label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMaximum(9999)
        hLayout.addWidget(self.slider)
        self.edit = QLineEdit()
        self.edit.setMaxLength(6)
        hLayout.addWidget(self.edit)
        # initialisation
        self.min = min
        self.max = max
        self.a = (max - min) / 9999.0  # Coeff de valeur = a * slidder + b (avec minSlidder = 0 et maxSlidder = 9999)
        self.b = min
        self.slider.setSliderPosition(val)
        self.label.setText(text)
        self.edit.setText(str(val))
        self.slider.setSliderPosition(self._invConvert(val))
        self.slider.valueChanged.connect(self.changerValeurEdit)
        self.edit.editingFinished.connect(self.changerValeurSlider)
        self.setLayout(hLayout)

    def value(self):
        return float(self.edit.text())

    def majEditEtSlider(self,value):
        '''
        Met à jour la valeur du slider et du edit d'après la valeur reçue
        @param value: une réel dans [min,max]
        @return:
        '''
        self.edit.blockSignals(True)  # On bloque les signaux émits lors d'un changement le temps de faire la màj sur les 2
        self.slider.blockSignals(True)  # pour éviter des signaux-slots cycliques
        self.edit.setText('{:.3}'.format(value))
        self.slider.setSliderPosition(self._invConvert(value))
        self.edit.blockSignals(False)
        self.slider.blockSignals(False)

    def _convert(self,pos):
        '''
        Méthode privée permettant de convertir une position d'un slider en sa valeur dans [min,max]
        @param pos:
        @return:
        '''
        return pos * self.a + self.b

    def _invConvert(self,val):
        '''
        Méthode privée permettant de convertir une valeur dans [min,max] en une position sur le slider
        @param val:
        @return:
        '''
        return int((val - self.b) / float(self.a))

    def changerValeurEdit(self):
        val = self.slider.value()
        self.edit.setText('{:.3}'.format(self._convert(val)))

    def changerValeurSlider(self):
        try:
            val = float(self.edit.text())
            self.slider.setSliderPosition(self._invConvert(val))
        except ValueError:
            QMessageBox.error(self, "Error", "Must be a number")



if __name__ == '__main__':
      # Programme principal
      app = QApplication([])
      ihm = Reglage(-3.14,3.14,'tyty',1)
      ihm.show()
      app.exec_()

