public class Main {
    public static void main(String[] args) {
        try {
            CSVReader.Data trainData = CSVReader.readCSV("data/train_data.csv", 3);
            double[][] trainFeatures = trainData.features;
            double[] trainTargets = trainData.targets;

            FeatureStandardizer standardizer = new FeatureStandardizer();
            standardizer.fit(trainFeatures);
            double[][] standardizedTrainFeatures = standardizer.transform(trainFeatures);

            LinearRegression model = new LinearRegression(0.01, 1000, 10);
            model.fit(standardizedTrainFeatures, trainTargets, 0.8);


            CSVReader.Data testData = CSVReader.readCSV("data/test_data.csv", 3);
            double[][] testFeatures = testData.features;
            double[] testTargets = testData.targets;
            double[][] standardizedTestFeatures = standardizer.transform(testFeatures);

            double[] predictions = model.predict(standardizedTestFeatures);

            double mse = model.calculateMSE(predictions, testTargets);
            System.out.println("MSE SCORE " + mse);
        }
        catch (Exception e) {
            System.err.println("Error reading CSV file.");
        }
    }
}