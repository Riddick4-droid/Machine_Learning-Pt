{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "138b17c6-b09e-4864-8a58-74666f7acec5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4e7e6681-654c-4424-b19b-c1ff0eb83334",
   "metadata": {},
   "outputs": [],
   "source": [
    "class NN_model(object):\n",
    "    def __init__(self):\n",
    "        path = os.getcwd()+'/model_exercise.pkl'\n",
    "        file = open(path,'rb')\n",
    "        self.model = pickle.load(file)\n",
    "    def predict(self,age,job,education,default,balance,housing,loan,day_of_week,month,duration,campaign,pdays,previous,married,single):\n",
    "        X = [[age,job,education,default, balance, housing, loan, day_of_week, month, duration, campaign, pdays, previous,married, single]]\n",
    "        return self.model.predict(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f49e4613-f923-4f60-92d8-a8cca1021316",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'C:\\\\Users\\\\LENOVO\\\\Desktop\\\\desktop docs\\\\SUPERVISEDLEARNING'"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "os.getcwd()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
