public class LinearRegression {
    private final double learningRate;
    private final int iterations;
    private int noOfTrainingPoints;
    private int noOfFeatures;
    private double[] weights;
    private double bias;
    private double[][] trainFeatures;
    private double[] trainTargets;
    private double[][] validationFeatures;
    private double[] validationTargets;
    private int patience;
    private double lambda;

    public LinearRegression(double learningRate, int iterations) {
        this.learningRate = learningRate;
        this.iterations = iterations;
        this.patience = 7;
        this.lambda = 0.001;
    }

    public LinearRegression(double learningRate, int iterations, int patience, double lambda) {
        this.learningRate = learningRate;
        this.iterations = iterations;
        this.patience = patience;
        this.lambda = lambda;
    }

    public LinearRegression fit(double[][] features, double[] targets) {
        return fit(features, targets, 0.8);
    }

    public LinearRegression fit(double[][] features, double[] targets, double trainRatio) {
        if (features == null || features.length == 0 || targets == null || targets.length == 0 || features.length != targets.length) {
            throw new IllegalArgumentException(
                    "Must contain at least one feature and target and length of features must be equal to length of targets.");
        }

        if (trainRatio <= 0 || trainRatio >= 1) {
            throw new IllegalArgumentException("Train ratio must be between 0 and 1.");
        }

        splitData(features, targets, trainRatio);

        this.noOfTrainingPoints = trainFeatures.length;
        this.noOfFeatures = trainFeatures[0].length;

        this.weights = new double[this.noOfFeatures];
        this.bias = 0;

        int noImprovementCount = 0;
        double bestValidationError = Double.MAX_VALUE;
        for (int i = 0; i < this.iterations; i++) {
            updateWeights();

            double[] validationPredictions = predict(this.validationFeatures);
            double validationError = calculateMSE(validationPredictions, this.validationTargets);

            if  (validationError < bestValidationError) {
                bestValidationError = validationError;
                noImprovementCount = 0;
            }
            else  {
                noImprovementCount++;
            }

            if (noImprovementCount == this.patience) {
                System.out.println("No more improvement after " + noImprovementCount + " iterations.");
                break;
            }
        }
        return this;
    }

    private void splitData(double[][] features, double[] targets, double trainRatio) {
        int totalSamples = features.length;
        int trainSize = (int) (totalSamples * trainRatio);
        int validationSize = totalSamples - trainSize;

        this.trainFeatures = new double[trainSize][features[0].length];
        this.trainTargets = new double[trainSize];
        this.validationFeatures = new double[validationSize][features[0].length];
        this.validationTargets = new double[validationSize];

        for (int i = 0; i < trainSize; i++) {
            System.arraycopy(features[i], 0, this.trainFeatures[i], 0, features[i].length);
            this.trainTargets[i] = targets[i];
        }

        for (int i = 0; i < validationSize; i++) {
            System.arraycopy(features[trainSize + i], 0, this.validationFeatures[i], 0, features[trainSize + i].length);
            this.validationTargets[i] = targets[trainSize + i];
        }
    }

    private void updateWeights() {
        double[] predictions = predict(this.trainFeatures);

        double[] dWeights = new double[this.noOfFeatures];
        for (int j = 0; j < this.noOfFeatures; j++) {
            double sum = 0.0;
            for (int i = 0; i < this.noOfTrainingPoints; i++) {
                sum += this.trainFeatures[i][j] * (this.trainTargets[i] - predictions[i]);
            }
            dWeights[j] = -2 * sum / this.noOfTrainingPoints;
        }

        double dBias = 0;
        for (int i = 0; i < this.noOfTrainingPoints; i++) {
            dBias += (this.trainTargets[i] - predictions[i]);
        }
        dBias = -2 * dBias / this.noOfTrainingPoints;

        for (int j = 0; j < this.noOfFeatures; j++) {
            weights[j] = weights[j] * (1 - learningRate * lambda) - learningRate * dWeights[j];
        }
        this.bias = this.bias - this.learningRate * dBias;
    }

    public double[] predict(double[][] features) {
        double[] results = new double[features.length];

        for (int i = 0; i < features.length; i++) {
            double sum = 0;
            for (int j = 0; j < features[i].length; j++) {
                sum += features[i][j] * this.weights[j];
            }
            results[i] = sum + this.bias;
        }
        return results;
    }

    public double calculateMSE(double[] predictions, double[] targets) {
        double sum = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            double error = targets[i] - predictions[i];
            sum += error * error;
        }
        return sum / predictions.length;
    }
}