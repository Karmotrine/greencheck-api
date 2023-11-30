from const import *
from lib.Processing import *
from lib.Model import Model
from lib.Analysis import *
import pprint

image = r'dataset\finalized\validation\blb\40.jpg'

model = Model(ModelType.BaseModel)
model.load()
pprint.pprint(model.predict(image))