import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CSVReader {

    public static class Data {
        public final double[][] features;
        public final double[] targets;

        public Data(double[][] features, double[] targets) {
            this.features = features;
            this.targets = targets;
        }
    }

    public static Data readCSV(String filename, int numFeatures) throws IOException {
        List<double[]> featuresList = new ArrayList<>();
        List<Double> targetsList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            boolean isFirstLine = true;

            while ((line = br.readLine()) != null) {
                if (isFirstLine) {
                    String[] parts = line.split(",");
                    try {
                        Double.parseDouble(parts[0].trim());
                        isFirstLine = false;
                    } catch (NumberFormatException e) {
                        isFirstLine = false;
                        continue;
                    }
                }

                String[] values = line.split(",");

                if (values.length < numFeatures + 1) {
                    System.err.println("Warning: Line has fewer than " + (numFeatures + 1) + " columns, skipping: " + line);
                    continue;
                }

                try {
                    double[] features = new double[numFeatures];
                    for (int i = 0; i < numFeatures; i++) {
                        features[i] = Double.parseDouble(values[i].trim());
                    }

                    double target = Double.parseDouble(values[values.length - 1].trim());

                    featuresList.add(features);
                    targetsList.add(target);

                } catch (NumberFormatException e) {
                    System.err.println("Warning: Could not parse line, skipping: " + line);
                }
            }
        }

        double[][] features = featuresList.toArray(new double[featuresList.size()][]);
        double[] targets = targetsList.stream().mapToDouble(Double::doubleValue).toArray();

        return new Data(features, targets);
    }
}