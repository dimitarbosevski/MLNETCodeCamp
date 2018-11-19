using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TitanicTest1.Models;

namespace TitanicTest1
{
    class Program
    {
        static void Main(string[] args)
        {
            //The Microsoft.ML.MLContext is a starting point for all ML.NET operations
            MLContext mlContext = new MLContext();

             //2. Location of the DATA file and the location where ZIP model will be generated
            string dataPath = "d:\\MLDir\\titanic.data";
            string trainedModelPath = "d:\\MLDir\\titanic.zip";

             //3. Lets load the file by calling TextReaded and then reading from CSV file               
            var textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
               {
                    new TextLoader.Column("Label", DataKind.R4, 1),  //r4 and r8 = single and double precision floating-point
                    new TextLoader.Column("Pclass", DataKind.R4, 2),
                    new TextLoader.Column("Sex", DataKind.R4, 4),
                    new TextLoader.Column("Age", DataKind.R4, 5),
                    new TextLoader.Column("SibSp", DataKind.R4, 6),
                    new TextLoader.Column("Parch", DataKind.R4, 7),
                    new TextLoader.Column("Ticket", DataKind.Text, 8),
                    new TextLoader.Column("Fare", DataKind.R4, 9),
                    new TextLoader.Column("Cabin", DataKind.Text, 10),
                    new TextLoader.Column("Embarked", DataKind.Text, 11),
                }
            });
            var fullData = textLoader.Read(dataPath);

             //4. Split the set on Train and Test data  
            (IDataView trainingDataView, IDataView testingDataView) = mlContext.Clustering.TrainTestSplit(fullData, testFraction: 0.3);

            //5. BinaryClassification needed since we have Survived or Not survived labels
            var trainer = mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent(features: "Features");

             //6. Creating the pipeline by appending transofrms and the trainer at the end
            var dataProcessPipeline = mlContext.Transforms.DropColumns("Cabin", "Ticket", "Embarked")
                .Append(mlContext.Transforms.ReplaceMissingValues("Age", null, NAReplaceTransform.ColumnInfo.ReplacementMode.Mean))
                .Append(mlContext.Transforms.Concatenate("Features", "Age", "Sex", "Pclass", "SibSp", "Parch", "Fare"))
                .Append(trainer);

            var trainedModel = dataProcessPipeline.Fit(trainingDataView);


            //7. Check agains the testing data and evaluate to get the accuracy.
            var predictions = trainedModel.Transform(testingDataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, label: "Label", score: "Score");
            Console.WriteLine("Accuracy: " + metrics.Accuracy);

             //8. Save the model as zip file
            using (var fs = new FileStream(trainedModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(trainedModel, fs);

              // --------------------------------------- NEW APPLICATION ------------------------------------
            //9. Test with one sample text 
            var sampleData = new Passenger()
            {
                Age = 23,
                Embarked = "S",
                Fare = 223,
                Parch = 3,
                Pclass = 1,
                Sex = 1,
                SibSp = 2
            };

             //10. Load the model, and do the prediction.
            using (var stream = new FileStream(trainedModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                var resultModel = TransformerChain.LoadFrom(mlContext, stream);
                var predictionFunction = resultModel.MakePredictionFunction<Passenger, PredictedData>(mlContext);
                var prediction = predictionFunction.Predict(sampleData);
                Console.WriteLine("Survived: " + prediction.IsSurvived);
            }


            Console.ReadLine();
        }
    }
}
