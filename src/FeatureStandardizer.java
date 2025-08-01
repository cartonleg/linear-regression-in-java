public class FeatureStandardizer {
    private double[] means;
    private double[] standardDeviations;
    private boolean fitted = false;
    private int numOfFeatures;

    public void fit(double[][] features) {
        if (features == null || features.length == 0 || features[0].length == 0) {
            throw new IllegalArgumentException("Features array cannot be null or empty");
        }

        int numSamples = features.length;
        this.numOfFeatures = features[0].length;
        this.means = new double[numOfFeatures];
        this.standardDeviations = new double[numOfFeatures];

        for (int j = 0; j < this.numOfFeatures; j++) {
            double sum = 0.0;
            for (int i = 0; i < numSamples; i++) {
                sum += features[i][j];
            }
            this.means[j] = sum / numSamples;
        }

        for (int j = 0; j < numOfFeatures; j++) {
            double sumSquaredDiffs = 0.0;
            for (int i = 0; i < numSamples; i++) {
                double diff = features[i][j] - means[j];
                sumSquaredDiffs += diff * diff;
            }
            this.standardDeviations[j] = Math.sqrt(sumSquaredDiffs / numSamples);

            // this is only a temporary solution as the better solution would be to delete the column all together
            // but since we know our data won't have any features with the std of 0 then this just saves time
            if (this.standardDeviations[j] == 0) {
                this.standardDeviations[j] = 1.0;
            }
        }

        this.fitted = true;
    }

    public double[][] transform(double[][] features) {
        if (!fitted) {
            throw new IllegalArgumentException("Must fit before transforming.");
        }

        if (features == null || features.length == 0) {
            throw new IllegalArgumentException("Features array cannot be empty.");
        }

        if (features[0].length != this.numOfFeatures) {
            throw new IllegalArgumentException("Number of features must match fitted data.");
        }

        int numSamples = features.length;
        double[][] standardized = new double[numSamples][this.numOfFeatures];

        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < this.numOfFeatures; j++) {
                standardized[i][j] = (features[i][j]-this.means[j])/this.standardDeviations[j];
            }
        }
        return standardized;
    }

    public double[][] fitTransform(double[][] features) {
        fit(features);
        return transform(features);
    }
}
