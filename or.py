from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import pandas as pd 


def main(data, modelName, plotName, eta, epochs):
    df_OR = pd.DataFrame(data)

    X, y = prepare_data(df_OR)
    ETA = 0.1
    EPOCHS = 10

    model_OR = Perceptron(eta = eta, epochs = epochs)
    model_OR.fit(X, y)
    _= model_OR.total_loss()

    model_OR.save(filename = modelName, model_dir="model")
    save_plot(df_OR, model_OR, filename=plotName)

if __name__ == "__main__":

    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [0,1,1,1]
    }

    ETA = 0.3
    EPOCHS = 10
    main(data=OR, modelName="or.pkl", plotName="or.png", eta=ETA, epochs=EPOCHS)

df_OR = pd.DataFrame(OR)

X, y = prepare_data(df_OR)
ETA = 0.1
EPOCHS = 10

model_OR = Perceptron(eta = ETA, epochs = EPOCHS)
model_OR.fit(X, y)
_= model_OR.total_loss()

model_OR.save(filename = "or.pkl", model_dir="model_or")
save_plot(df_OR, model_OR, filename="or.png")