Horse Racing Predictor
This project is a neural network-based model designed to predict the top 2, 3, or 4 horses likely to place in a race. It's built using features extracted from horse racing data. 
The project is functional, but is far from complete.

Features
Data Processing: Handles CSV files containing race data, including feature engineering for inputs like horse performance history, track conditions, etc.
Neural Network Model: A trained model that outputs predictions for top placements, also a seperate model which attempts to predict winners
Prediction Modes: Configurable to predict top 2, 3, or 4 horses per race depending on the size of the race.

Model Details
Architecture: Feedforward neural network
Training: Uses supervised learning on historical race results.
Evaluation: Basic metrics like accuracy for top-N predictions (e.g., hit rate for top 3).

Limitations
The project is incomplete: No full deployment, only runs locally
Performance may vary based on data quality and race specifics.
No real-time data integration.

Future Work
Integrate live data feeds.
Improve feature selection and model interpretability.

How It Works
First I pulled 12 months of racecard data from a reliable API. I also pulled horse specific data, dam specific data, jockey specific data and owner specific data.
I then merged these together so they were in one file. For example, to merge a jockeys distance stas to a horse, I merged where jockey_id and distance matched. I did this as neural network training requires 
all data to be in one file.
Once this was done I then created my features to enhance my models predictability in predicting results. This part was really fun as it allowed me to be creative with the data I had access to.
After I had all my features built and my data inside one file, I then moved onto training my model. This was probably the most challenging aspect of the whole project, not because it was difficult,
but it was a lot of trial and error. I got there in the end though and it produced some very good results!
My last contribution to this project was adding a script where I could predict a daily racecard. This was so I could test the model actually worked. And it did! In the months where I was testing it, it had multiple horses
with odds over 10/1 win the race!

Conclusion
I loved my time building this project, but decided to pursue a different project. I'd like to pick this back up again in the future and try and improve it as much as I can.
